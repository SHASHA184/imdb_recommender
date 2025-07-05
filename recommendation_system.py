#!/usr/bin/env python3
"""
IMDb Movie Recommendation System
Provides multiple recommendation algorithms for movies and TV shows
"""

import pandas as pd
import polars as pl
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import time


class IMDbRecommendationSystem:
    """
    A comprehensive movie/TV show recommendation system using IMDb data
    """

    def __init__(self, data_path: str = ".", use_polars: bool = True):
        """
        Initialize the recommendation system

        Args:
            data_path: Path to directory containing IMDb TSV files
            use_polars: Whether to use Polars for faster processing
        """
        self.data_path = Path(data_path)
        self.use_polars = use_polars
        self.titles_df = None
        self.ratings_df = None
        self.names_df = None
        self.principals_df = None
        self.crew_df = None

        # Recommendation models
        self.content_features = None
        self.similarity_matrix = None
        self.title_to_index = None
        self.index_to_title = None

        # Load data
        self._load_data()

    def _load_data(self):
        """Load IMDb data files"""
        print("🔄 Loading IMDb data...")
        start_time = time.time()

        if self.use_polars:
            self._load_with_polars()
        else:
            self._load_with_pandas()

        print(f"✅ Data loaded in {time.time() - start_time:.2f} seconds")
        print(f"📊 Loaded {len(self.titles_df)} titles")

    def _load_with_polars(self):
        """Load data using Polars (faster)"""
        # Load main files
        self.titles_df = pl.read_csv(
            self.data_path / "title.basics.tsv",
            separator="\t",
            null_values=["\\N"],
            quote_char=None,
            ignore_errors=True,
        )

        self.ratings_df = pl.read_csv(
            self.data_path / "title.ratings.tsv",
            separator="\t",
            null_values=["\\N"],
            quote_char=None,
            ignore_errors=True,
        )

        self.names_df = pl.read_csv(
            self.data_path / "name.basics.tsv",
            separator="\t",
            null_values=["\\N"],
            quote_char=None,
            ignore_errors=True,
        )

        self.crew_df = pl.read_csv(
            self.data_path / "title.crew.tsv",
            separator="\t",
            null_values=["\\N"],
            quote_char=None,
            ignore_errors=True,
        )

        # Load principals (may be large, so handle with care)
        try:
            self.principals_df = pl.read_csv(
                self.data_path / "title.principals.tsv",
                separator="\t",
                null_values=["\\N"],
                quote_char=None,
                ignore_errors=True,
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not load principals data: {e}")
            self.principals_df = None

    def _load_with_pandas(self):
        """Load data using Pandas"""
        # Load main files
        self.titles_df = pd.read_csv(
            self.data_path / "title.basics.tsv",
            sep="\t",
            low_memory=False,
            na_values=["\\N"],
        )

        self.ratings_df = pd.read_csv(
            self.data_path / "title.ratings.tsv", sep="\t", na_values=["\\N"]
        )

        self.names_df = pd.read_csv(
            self.data_path / "name.basics.tsv",
            sep="\t",
            low_memory=False,
            na_values=["\\N"],
        )

        self.crew_df = pd.read_csv(
            self.data_path / "title.crew.tsv",
            sep="\t",
            low_memory=False,
            na_values=["\\N"],
        )

        # Load principals (may be large)
        try:
            self.principals_df = pd.read_csv(
                self.data_path / "title.principals.tsv",
                sep="\t",
                low_memory=False,
                na_values=["\\N"],
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not load principals data: {e}")
            self.principals_df = None

    def build_content_features(self):
        """Build content-based feature matrix"""
        print("🔄 Building content features...")
        start_time = time.time()

        if self.use_polars:
            # Join titles with ratings
            content_df = (
                self.titles_df.join(self.ratings_df, on="tconst", how="left")
                .filter(
                    pl.col("titleType").is_in(["movie", "tvSeries", "tvMiniSeries"])
                )
                .filter(pl.col("genres").is_not_null())
                .filter(pl.col("primaryTitle").is_not_null())
            )

            # Convert to pandas for sklearn compatibility
            content_df = content_df.to_pandas()
        else:
            # Filter and join data
            content_df = pd.merge(
                self.titles_df[
                    self.titles_df["titleType"].isin(
                        ["movie", "tvSeries", "tvMiniSeries"]
                    )
                ],
                self.ratings_df,
                on="tconst",
                how="left",
            )
            content_df = content_df.dropna(subset=["genres", "primaryTitle"])

        # Fill missing values
        content_df["averageRating"] = content_df["averageRating"].fillna(0)
        content_df["numVotes"] = content_df["numVotes"].fillna(0)
        content_df["startYear"] = content_df["startYear"].fillna(0)
        content_df["runtimeMinutes"] = content_df["runtimeMinutes"].fillna(0)

        # Create feature vectors
        # 1. Genre features using TF-IDF
        genre_vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(","), lowercase=False, token_pattern=None
        )
        genre_features = genre_vectorizer.fit_transform(content_df["genres"].fillna(""))

        # 2. Numerical features
        numerical_features = content_df[
            ["averageRating", "numVotes", "startYear", "runtimeMinutes"]
        ].values

        # Scale numerical features
        scaler = StandardScaler()
        numerical_features = scaler.fit_transform(numerical_features)

        # Combine features
        from scipy.sparse import hstack

        self.content_features = hstack([genre_features, numerical_features])

        # Create mapping between titles and indices
        self.title_to_index = {
            title: idx for idx, title in enumerate(content_df["primaryTitle"])
        }
        self.index_to_title = {idx: title for title, idx in self.title_to_index.items()}

        # Store metadata
        self.content_metadata = content_df[
            [
                "tconst",
                "primaryTitle",
                "titleType",
                "startYear",
                "genres",
                "averageRating",
                "numVotes",
            ]
        ].copy()

        print(f"✅ Content features built in {time.time() - start_time:.2f} seconds")
        print(f"📊 Feature matrix shape: {self.content_features.shape}")

    def compute_similarity_matrix(self):
        """Compute similarity matrix for content-based recommendations"""
        if self.content_features is None:
            self.build_content_features()

        print("🔄 Computing similarity matrix...")
        start_time = time.time()

        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(self.content_features)

        print(
            f"✅ Similarity matrix computed in {time.time() - start_time:.2f} seconds"
        )

    def get_content_based_recommendations(
        self,
        title: str,
        n_recommendations: int = 10,
        min_rating: float = 6.0,
        min_votes: int = 1000,
    ) -> List[Dict]:
        """
        Get content-based recommendations for a given title

        Args:
            title: Title to get recommendations for
            n_recommendations: Number of recommendations to return
            min_rating: Minimum rating threshold
            min_votes: Minimum number of votes threshold

        Returns:
            List of recommended titles with metadata
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # Find title index
        if title not in self.title_to_index:
            # Try partial matching
            matches = [
                t for t in self.title_to_index.keys() if title.lower() in t.lower()
            ]
            if not matches:
                return []
            title = matches[0]
            print(f"🔍 Using closest match: {title}")

        title_idx = self.title_to_index[title]

        # Get similarity scores
        similarity_scores = self.similarity_matrix[title_idx]

        # Get top similar titles
        similar_indices = np.argsort(similarity_scores)[::-1]

        recommendations = []
        for idx in similar_indices[1:]:  # Skip the title itself
            if len(recommendations) >= n_recommendations:
                break

            metadata = self.content_metadata.iloc[idx]

            # Apply filters
            if (
                metadata["averageRating"] >= min_rating
                and metadata["numVotes"] >= min_votes
            ):

                recommendations.append(
                    {
                        "title": metadata["primaryTitle"],
                        "year": (
                            int(metadata["startYear"])
                            if pd.notna(metadata["startYear"])
                            else None
                        ),
                        "type": metadata["titleType"],
                        "genres": metadata["genres"],
                        "rating": round(metadata["averageRating"], 1),
                        "votes": int(metadata["numVotes"]),
                        "similarity_score": round(similarity_scores[idx], 3),
                    }
                )

        return recommendations

    def get_popular_recommendations(
        self,
        title_type: str = "movie",
        genre: Optional[str] = None,
        year_range: Optional[Tuple[int, int]] = None,
        n_recommendations: int = 10,
        min_votes: int = 10000,
    ) -> List[Dict]:
        """
        Get popular recommendations based on ratings and votes

        Args:
            title_type: Type of content ("movie", "tvSeries", etc.)
            genre: Genre filter (optional)
            year_range: Year range filter (start_year, end_year)
            n_recommendations: Number of recommendations
            min_votes: Minimum votes threshold

        Returns:
            List of popular titles with metadata
        """
        if self.use_polars:
            # Filter using Polars
            filtered_df = (
                self.titles_df.join(self.ratings_df, on="tconst", how="inner")
                .filter(pl.col("titleType") == title_type)
                .filter(pl.col("numVotes") >= min_votes)
            )

            # Apply genre filter
            if genre:
                filtered_df = filtered_df.filter(
                    pl.col("genres").str.contains(genre, literal=False)
                )

            # Apply year range filter
            if year_range:
                start_year, end_year = year_range
                filtered_df = filtered_df.filter(
                    (pl.col("startYear") >= start_year)
                    & (pl.col("startYear") <= end_year)
                )

            # Sort by rating and get top recommendations
            top_titles = (
                filtered_df.sort("averageRating", descending=True)
                .head(n_recommendations)
                .to_pandas()
            )

        else:
            # Filter using Pandas
            filtered_df = pd.merge(
                self.titles_df[self.titles_df["titleType"] == title_type],
                self.ratings_df,
                on="tconst",
                how="inner",
            )

            filtered_df = filtered_df[filtered_df["numVotes"] >= min_votes]

            # Apply genre filter
            if genre:
                filtered_df = filtered_df[
                    filtered_df["genres"].str.contains(genre, na=False, case=False)
                ]

            # Apply year range filter
            if year_range:
                start_year, end_year = year_range
                filtered_df = filtered_df[
                    (filtered_df["startYear"] >= start_year)
                    & (filtered_df["startYear"] <= end_year)
                ]

            # Sort by rating and get top recommendations
            top_titles = filtered_df.nlargest(n_recommendations, "averageRating")

        # Format recommendations
        recommendations = []
        for _, row in top_titles.iterrows():
            recommendations.append(
                {
                    "title": row["primaryTitle"],
                    "year": (
                        int(row["startYear"]) if pd.notna(row["startYear"]) else None
                    ),
                    "type": row["titleType"],
                    "genres": row["genres"],
                    "rating": round(row["averageRating"], 1),
                    "votes": int(row["numVotes"]),
                    "popularity_score": round(
                        row["averageRating"] * np.log(row["numVotes"]), 2
                    ),
                }
            )

        return recommendations

    def get_recommendations_by_genre(
        self,
        genre: str,
        title_type: str = "movie",
        n_recommendations: int = 10,
        min_rating: float = 7.0,
        min_votes: int = 5000,
    ) -> List[Dict]:
        """
        Get recommendations for a specific genre

        Args:
            genre: Genre to filter by
            title_type: Type of content
            n_recommendations: Number of recommendations
            min_rating: Minimum rating threshold
            min_votes: Minimum votes threshold

        Returns:
            List of genre-specific recommendations
        """
        return self.get_popular_recommendations(
            title_type=title_type,
            genre=genre,
            n_recommendations=n_recommendations,
            min_votes=min_votes,
        )

    def get_recommendations_by_year(
        self,
        year: int,
        title_type: str = "movie",
        n_recommendations: int = 10,
        min_rating: float = 7.0,
        min_votes: int = 5000,
    ) -> List[Dict]:
        """
        Get recommendations for a specific year

        Args:
            year: Year to filter by
            title_type: Type of content
            n_recommendations: Number of recommendations
            min_rating: Minimum rating threshold
            min_votes: Minimum votes threshold

        Returns:
            List of year-specific recommendations
        """
        return self.get_popular_recommendations(
            title_type=title_type,
            year_range=(year, year),
            n_recommendations=n_recommendations,
            min_votes=min_votes,
        )

    def search_titles(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for titles by name

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching titles
        """
        if self.use_polars:
            results = (
                self.titles_df.join(self.ratings_df, on="tconst", how="left")
                .filter(pl.col("primaryTitle").str.contains(query, literal=False))
                .filter(
                    pl.col("titleType").is_in(["movie", "tvSeries", "tvMiniSeries"])
                )
                .sort("numVotes", descending=True, nulls_last=True)
                .head(limit)
                .to_pandas()
            )
        else:
            results = pd.merge(
                self.titles_df[
                    self.titles_df["primaryTitle"].str.contains(
                        query, case=False, na=False
                    )
                ],
                self.ratings_df,
                on="tconst",
                how="left",
            )
            results = results[
                results["titleType"].isin(["movie", "tvSeries", "tvMiniSeries"])
            ]
            results = results.sort_values(
                "numVotes", ascending=False, na_position="last"
            ).head(limit)

        # Format results
        formatted_results = []
        for _, row in results.iterrows():
            formatted_results.append(
                {
                    "title": row["primaryTitle"],
                    "year": (
                        int(row["startYear"]) if pd.notna(row["startYear"]) else None
                    ),
                    "type": row["titleType"],
                    "genres": row["genres"] if pd.notna(row["genres"]) else "Unknown",
                    "rating": (
                        round(row["averageRating"], 1)
                        if pd.notna(row["averageRating"])
                        else None
                    ),
                    "votes": int(row["numVotes"]) if pd.notna(row["numVotes"]) else 0,
                }
            )

        return formatted_results

    def get_hybrid_recommendations(
        self,
        title: str,
        n_recommendations: int = 10,
        content_weight: float = 0.7,
        popularity_weight: float = 0.3,
        min_rating: float = 6.0,
        min_votes: int = 1000,
    ) -> List[Dict]:
        """
        Get hybrid recommendations combining content-based and popularity-based approaches

        Args:
            title: Title to get recommendations for
            n_recommendations: Number of recommendations
            content_weight: Weight for content-based score
            popularity_weight: Weight for popularity score
            min_rating: Minimum rating threshold
            min_votes: Minimum votes threshold

        Returns:
            List of hybrid recommendations
        """
        # Get content-based recommendations
        content_recs = self.get_content_based_recommendations(
            title, n_recommendations * 2, min_rating, min_votes
        )

        # Get metadata for the source title
        source_metadata = None
        if self.content_metadata is not None:
            source_matches = self.content_metadata[
                self.content_metadata["primaryTitle"].str.contains(
                    title, case=False, na=False
                )
            ]
            if not source_matches.empty:
                source_metadata = source_matches.iloc[0]

        # Get popular recommendations from same genre if available
        popular_recs = []
        if source_metadata is not None and pd.notna(source_metadata["genres"]):
            genres = source_metadata["genres"].split(",")
            for genre in genres[:2]:  # Use first 2 genres
                popular_recs.extend(
                    self.get_popular_recommendations(
                        title_type=source_metadata["titleType"],
                        genre=genre.strip(),
                        n_recommendations=n_recommendations,
                        min_votes=min_votes,
                    )
                )

        # Combine and score recommendations
        all_recommendations = {}

        # Add content-based recommendations
        for rec in content_recs:
            title_key = rec["title"]
            all_recommendations[title_key] = rec.copy()
            all_recommendations[title_key]["content_score"] = rec["similarity_score"]
            all_recommendations[title_key]["popularity_score"] = 0

        # Add popularity-based recommendations
        for rec in popular_recs:
            title_key = rec["title"]
            if title_key in all_recommendations:
                all_recommendations[title_key]["popularity_score"] = rec[
                    "popularity_score"
                ]
            else:
                all_recommendations[title_key] = rec.copy()
                all_recommendations[title_key]["content_score"] = 0
                all_recommendations[title_key]["popularity_score"] = rec[
                    "popularity_score"
                ]

        # Calculate hybrid scores
        for title_key in all_recommendations:
            rec = all_recommendations[title_key]
            content_score = rec.get("content_score", 0)
            popularity_score = rec.get("popularity_score", 0)

            # Normalize scores
            content_score_norm = content_score / max(
                1,
                max([r.get("content_score", 0) for r in all_recommendations.values()]),
            )
            popularity_score_norm = popularity_score / max(
                1,
                max(
                    [r.get("popularity_score", 0) for r in all_recommendations.values()]
                ),
            )

            rec["hybrid_score"] = (
                content_weight * content_score_norm
                + popularity_weight * popularity_score_norm
            )

        # Sort by hybrid score and return top recommendations
        sorted_recs = sorted(
            all_recommendations.values(), key=lambda x: x["hybrid_score"], reverse=True
        )

        return sorted_recs[:n_recommendations]

    def print_recommendations(
        self, recommendations: List[Dict], title: str = "Recommendations"
    ):
        """Pretty print recommendations"""
        print(f"\n🎬 {title}")
        print("=" * 80)

        for i, rec in enumerate(recommendations, 1):
            year = f"({rec['year']})" if rec["year"] else ""
            rating = f"⭐ {rec['rating']}/10" if rec["rating"] else "No rating"
            votes = f"{rec['votes']:,} votes" if rec["votes"] else "No votes"

            print(f"{i:2d}. {rec['title']} {year}")
            print(f"    Type: {rec['type']} | Genres: {rec['genres']}")
            print(f"    {rating} | {votes}")

            # Show additional scores if available
            if "similarity_score" in rec:
                print(f"    Similarity: {rec['similarity_score']}")
            if "popularity_score" in rec:
                print(f"    Popularity: {rec['popularity_score']}")
            if "hybrid_score" in rec:
                print(f"    Hybrid Score: {rec['hybrid_score']:.3f}")
            print()


def main():
    """Example usage of the recommendation system"""
    # Initialize the recommendation system
    print("🎬 IMDb Recommendation System")
    print("=" * 50)

    recommender = IMDbRecommendationSystem()

    # Example 1: Content-based recommendations
    print("\n1. Content-Based Recommendations for 'The Dark Knight'")
    dark_knight_recs = recommender.get_content_based_recommendations(
        "The Dark Knight", n_recommendations=5, min_rating=7.0, min_votes=50000
    )
    recommender.print_recommendations(dark_knight_recs, "Similar to 'The Dark Knight'")

    # Example 2: Popular recommendations by genre
    print("\n2. Popular Action Movies")
    action_recs = recommender.get_recommendations_by_genre(
        "Action", n_recommendations=5, min_rating=7.5, min_votes=100000
    )
    recommender.print_recommendations(action_recs, "Top Action Movies")

    # Example 3: Search functionality
    print("\n3. Search Results for 'Avengers'")
    search_results = recommender.search_titles("Avengers", limit=5)
    recommender.print_recommendations(search_results, "Search: 'Avengers'")

    # Example 4: Hybrid recommendations
    print("\n4. Hybrid Recommendations for 'Inception'")
    hybrid_recs = recommender.get_hybrid_recommendations(
        "Inception", n_recommendations=5, content_weight=0.6, popularity_weight=0.4
    )
    recommender.print_recommendations(
        hybrid_recs, "Hybrid Recommendations for 'Inception'"
    )


if __name__ == "__main__":
    main()
