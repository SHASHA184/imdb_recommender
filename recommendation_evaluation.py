#!/usr/bin/env python3
"""
IMDb Recommendation System Evaluation
Tools for measuring recommendation quality and system performance
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import time
from recommendation_system import IMDbRecommendationSystem
from collections import defaultdict
import statistics


class RecommendationEvaluator:
    """
    Evaluates the quality of recommendations
    """

    def __init__(self, recommender: IMDbRecommendationSystem):
        self.recommender = recommender
        self.evaluation_results = {}

    def evaluate_diversity(self, recommendations: List[Dict]) -> float:
        """
        Evaluate diversity of recommendations based on genres
        Returns a score between 0-1 (higher is more diverse)
        """
        if not recommendations:
            return 0.0

        # Extract all genres from recommendations
        all_genres = set()
        for rec in recommendations:
            if rec.get("genres"):
                genres = rec["genres"].split(",")
                all_genres.update([g.strip() for g in genres])

        # Diversity score based on number of unique genres
        max_possible_genres = 20  # Approximate max genres in IMDb
        diversity_score = min(len(all_genres) / max_possible_genres, 1.0)

        return diversity_score

    def evaluate_novelty(self, recommendations: List[Dict]) -> float:
        """
        Evaluate novelty of recommendations (how non-obvious they are)
        Returns a score between 0-1 (higher is more novel)
        """
        if not recommendations:
            return 0.0

        # Calculate novelty based on inverse popularity
        novelty_scores = []
        for rec in recommendations:
            votes = rec.get("votes", 0)
            if votes > 0:
                # Log scale - less popular movies get higher novelty scores
                novelty = 1 - min(
                    np.log(votes) / np.log(2000000), 1.0
                )  # 2M votes as max
                novelty_scores.append(max(novelty, 0))
            else:
                novelty_scores.append(0.5)  # Default for movies with no votes

        return np.mean(novelty_scores) if novelty_scores else 0.0

    def evaluate_quality(self, recommendations: List[Dict]) -> Dict[str, float]:
        """
        Evaluate overall quality of recommendations
        Returns various quality metrics
        """
        if not recommendations:
            return {
                "avg_rating": 0.0,
                "avg_votes": 0.0,
                "diversity": 0.0,
                "novelty": 0.0,
                "coverage": 0.0,
            }

        # Average rating
        ratings = [rec.get("rating", 0) for rec in recommendations if rec.get("rating")]
        avg_rating = np.mean(ratings) if ratings else 0.0

        # Average votes (popularity)
        votes = [rec.get("votes", 0) for rec in recommendations if rec.get("votes")]
        avg_votes = np.mean(votes) if votes else 0.0

        # Diversity
        diversity = self.evaluate_diversity(recommendations)

        # Novelty
        novelty = self.evaluate_novelty(recommendations)

        # Coverage (how many different types/years are covered)
        types = set(rec.get("type", "") for rec in recommendations)
        years = set(rec.get("year", 0) for rec in recommendations if rec.get("year"))
        coverage = (len(types) + len(years)) / (3 + 50)  # Normalize by typical ranges

        return {
            "avg_rating": round(avg_rating, 2),
            "avg_votes": round(avg_votes, 0),
            "diversity": round(diversity, 3),
            "novelty": round(novelty, 3),
            "coverage": round(coverage, 3),
            "quality_score": round(
                (avg_rating / 10 + diversity + novelty + coverage) / 4, 3
            ),
        }

    def benchmark_performance(
        self, test_movies: List[str], n_recommendations: int = 10
    ) -> Dict:
        """
        Benchmark the performance of different recommendation methods
        """
        print("🔄 Running performance benchmark...")

        results = {
            "content_based": {"times": [], "qualities": []},
            "popular": {"times": [], "qualities": []},
            "hybrid": {"times": [], "qualities": []},
        }

        for movie in test_movies:
            print(f"  Testing with '{movie}'...")

            # Content-based recommendations
            start_time = time.time()
            try:
                content_recs = self.recommender.get_content_based_recommendations(
                    movie, n_recommendations=n_recommendations
                )
                content_time = time.time() - start_time
                content_quality = self.evaluate_quality(content_recs)

                results["content_based"]["times"].append(content_time)
                results["content_based"]["qualities"].append(content_quality)
            except Exception as e:
                print(f"    ❌ Content-based failed: {e}")

            # Popular recommendations (same genre)
            start_time = time.time()
            try:
                # Get genre from the movie if possible
                search_results = self.recommender.search_titles(movie, limit=1)
                if search_results and search_results[0].get("genres"):
                    genre = search_results[0]["genres"].split(",")[0].strip()
                    popular_recs = self.recommender.get_recommendations_by_genre(
                        genre, n_recommendations=n_recommendations
                    )
                else:
                    popular_recs = self.recommender.get_popular_recommendations(
                        n_recommendations=n_recommendations
                    )

                popular_time = time.time() - start_time
                popular_quality = self.evaluate_quality(popular_recs)

                results["popular"]["times"].append(popular_time)
                results["popular"]["qualities"].append(popular_quality)
            except Exception as e:
                print(f"    ❌ Popular recommendations failed: {e}")

            # Hybrid recommendations
            start_time = time.time()
            try:
                hybrid_recs = self.recommender.get_hybrid_recommendations(
                    movie, n_recommendations=n_recommendations
                )
                hybrid_time = time.time() - start_time
                hybrid_quality = self.evaluate_quality(hybrid_recs)

                results["hybrid"]["times"].append(hybrid_time)
                results["hybrid"]["qualities"].append(hybrid_quality)
            except Exception as e:
                print(f"    ❌ Hybrid recommendations failed: {e}")

        # Calculate average results
        summary = {}
        for method in results:
            if results[method]["times"]:
                avg_time = np.mean(results[method]["times"])
                avg_quality = {}

                # Average quality metrics
                quality_keys = results[method]["qualities"][0].keys()
                for key in quality_keys:
                    values = [q[key] for q in results[method]["qualities"]]
                    avg_quality[key] = np.mean(values)

                summary[method] = {
                    "avg_time": round(avg_time, 3),
                    "avg_quality": avg_quality,
                }

        return summary

    def analyze_recommendation_patterns(self, test_movies: List[str]) -> Dict:
        """
        Analyze patterns in recommendations to understand system behavior
        """
        print("🔄 Analyzing recommendation patterns...")

        patterns = {
            "genre_distribution": defaultdict(int),
            "year_distribution": defaultdict(int),
            "rating_distribution": defaultdict(int),
            "common_recommendations": defaultdict(int),
        }

        all_recommendations = []

        for movie in test_movies:
            try:
                recs = self.recommender.get_content_based_recommendations(
                    movie, n_recommendations=10
                )
                all_recommendations.extend(recs)

                for rec in recs:
                    # Genre distribution
                    if rec.get("genres"):
                        for genre in rec["genres"].split(","):
                            patterns["genre_distribution"][genre.strip()] += 1

                    # Year distribution
                    if rec.get("year"):
                        decade = (rec["year"] // 10) * 10
                        patterns["year_distribution"][decade] += 1

                    # Rating distribution
                    if rec.get("rating"):
                        rating_range = f"{int(rec['rating'])}-{int(rec['rating'])+1}"
                        patterns["rating_distribution"][rating_range] += 1

                    # Common recommendations
                    patterns["common_recommendations"][rec["title"]] += 1

            except Exception as e:
                print(f"    ❌ Error analyzing '{movie}': {e}")

        # Convert to sorted lists for better readability
        for key in patterns:
            patterns[key] = dict(
                sorted(patterns[key].items(), key=lambda x: x[1], reverse=True)
            )

        return patterns

    def generate_evaluation_report(self, test_movies: List[str]) -> str:
        """
        Generate a comprehensive evaluation report
        """
        print("📊 Generating evaluation report...")

        # Performance benchmark
        performance = self.benchmark_performance(test_movies)

        # Pattern analysis
        patterns = self.analyze_recommendation_patterns(test_movies)

        # Generate report
        report = []
        report.append("🎬 IMDb Recommendation System - Evaluation Report")
        report.append("=" * 60)

        # Performance section
        report.append("\n📈 PERFORMANCE BENCHMARK")
        report.append("-" * 30)

        for method, results in performance.items():
            report.append(f"\n{method.upper()} RECOMMENDATIONS:")
            report.append(f"  ⏱️  Average Time: {results['avg_time']:.3f} seconds")
            report.append(
                f"  ⭐ Average Rating: {results['avg_quality']['avg_rating']:.1f}/10"
            )
            report.append(
                f"  🎭 Diversity Score: {results['avg_quality']['diversity']:.3f}"
            )
            report.append(
                f"  🆕 Novelty Score: {results['avg_quality']['novelty']:.3f}"
            )
            report.append(
                f"  🏆 Overall Quality: {results['avg_quality']['quality_score']:.3f}"
            )

        # Pattern analysis section
        report.append("\n\n📊 RECOMMENDATION PATTERNS")
        report.append("-" * 30)

        # Top genres
        report.append("\n🎭 Most Recommended Genres:")
        for genre, count in list(patterns["genre_distribution"].items())[:10]:
            report.append(f"  {genre}: {count} recommendations")

        # Year distribution
        report.append("\n📅 Most Recommended Decades:")
        for decade, count in list(patterns["year_distribution"].items())[:10]:
            report.append(f"  {decade}s: {count} recommendations")

        # Rating distribution
        report.append("\n⭐ Rating Distribution:")
        for rating_range, count in patterns["rating_distribution"].items():
            report.append(f"  {rating_range}: {count} recommendations")

        # Common recommendations
        report.append("\n🔄 Most Frequently Recommended Movies:")
        for title, count in list(patterns["common_recommendations"].items())[:10]:
            report.append(f"  {title}: {count} times")

        # Recommendations section
        report.append("\n\n💡 RECOMMENDATIONS FOR IMPROVEMENT")
        report.append("-" * 30)

        # Analysis-based recommendations
        best_method = max(
            performance.keys(),
            key=lambda x: performance[x]["avg_quality"]["quality_score"],
        )
        report.append(f"• Best performing method: {best_method.upper()}")

        avg_diversity = np.mean(
            [performance[m]["avg_quality"]["diversity"] for m in performance]
        )
        if avg_diversity < 0.3:
            report.append("• Consider improving genre diversity in recommendations")

        avg_novelty = np.mean(
            [performance[m]["avg_quality"]["novelty"] for m in performance]
        )
        if avg_novelty < 0.3:
            report.append("• Consider balancing popular and niche recommendations")

        return "\n".join(report)


def main():
    """Main evaluation function"""
    print("📊 IMDb Recommendation System Evaluation")
    print("=" * 50)

    # Initialize recommendation system
    recommender = IMDbRecommendationSystem()
    evaluator = RecommendationEvaluator(recommender)

    # Test movies for evaluation
    test_movies = [
        "The Dark Knight",
        "Inception",
        "Pulp Fiction",
        "The Matrix",
        "Forrest Gump",
        "The Godfather",
        "Shawshank Redemption",
        "Interstellar",
        "Fight Club",
        "Good Will Hunting",
    ]

    # Generate and display evaluation report
    report = evaluator.generate_evaluation_report(test_movies)
    print(report)

    # Save report to file
    with open("evaluation_report.txt", "w") as f:
        f.write(report)

    print(f"\n📁 Evaluation report saved to 'evaluation_report.txt'")


if __name__ == "__main__":
    main()
