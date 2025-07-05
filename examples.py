#!/usr/bin/env python3
"""
IMDb Dataset Analysis Examples
Demonstrates how to analyze IMDb data directly from TSV files using pandas, polars, and DuckDB
"""

import pandas as pd
import polars as pl
import duckdb
import time
from pathlib import Path
from typing import Optional


def check_files_exist() -> bool:
    """Check if all required TSV files exist"""
    required_files = [
        "title.basics.tsv",
        "name.basics.tsv",
        "title.ratings.tsv",
        "title.crew.tsv",
        "title.episode.tsv",
        "title.akas.tsv",
        "title.principals.tsv",
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False

    print("✅ All TSV files found!")
    return True


def show_file_info():
    """Show information about TSV files"""
    print("\n📁 Dataset File Information")
    print("=" * 60)

    files = [
        ("title.basics.tsv", "969MB", "Movies/TV shows info"),
        ("name.basics.tsv", "851MB", "People info"),
        ("title.ratings.tsv", "27MB", "Ratings & votes"),
        ("title.crew.tsv", "370MB", "Directors & writers"),
        ("title.episode.tsv", "226MB", "TV episodes"),
        ("title.akas.tsv", "2.5GB", "Alternative titles"),
        ("title.principals.tsv", "3.9GB", "Cast & crew details"),
    ]

    for filename, size, description in files:
        status = "✅" if Path(filename).exists() else "❌"
        print(f"{status} {filename:<20} {size:<8} - {description}")


def example_pandas_top_movies(min_votes: int = 10000, limit: int = 10):
    """Find top rated movies using pandas"""
    print(f"\n🏆 Top {limit} Rated Movies (pandas, min {min_votes:,} votes)")
    print("=" * 60)

    start_time = time.time()

    # Load data
    titles = pd.read_csv("title.basics.tsv", sep="\t", low_memory=False)
    ratings = pd.read_csv("title.ratings.tsv", sep="\t")

    # Filter movies and join with ratings
    movies = titles[titles["titleType"] == "movie"]
    result = pd.merge(movies, ratings, on="tconst")

    # Filter by vote threshold and get top rated
    top_movies = result[result["numVotes"] >= min_votes].nlargest(
        limit, "averageRating"
    )[["primaryTitle", "startYear", "averageRating", "numVotes"]]

    load_time = time.time() - start_time

    for i, (_, row) in enumerate(top_movies.iterrows(), 1):
        year = int(row["startYear"]) if pd.notna(row["startYear"]) else "Unknown"
        print(
            f"{i:2d}. {row['primaryTitle']} ({year}) - ⭐ {row['averageRating']}/10 ({row['numVotes']:,} votes)"
        )

    print(f"\n⏱️  Processing time: {load_time:.2f} seconds")


def example_polars_top_movies(min_votes: int = 10000, limit: int = 10):
    """Find top rated movies using polars (faster)"""
    print(f"\n🏆 Top {limit} Rated Movies (polars, min {min_votes:,} votes)")
    print("=" * 60)

    start_time = time.time()

    # Load and process with polars (handle IMDb's \N null values and TSV quoting)
    titles = pl.read_csv(
        "title.basics.tsv",
        separator="\t",
        null_values=["\\N"],
        quote_char=None,
        ignore_errors=True,
    )
    ratings = pl.read_csv(
        "title.ratings.tsv",
        separator="\t",
        null_values=["\\N"],
        quote_char=None,
        ignore_errors=True,
    )

    top_movies = (
        titles.filter(pl.col("titleType") == "movie")
        .join(ratings, on="tconst")
        .filter(pl.col("numVotes") >= min_votes)
        .sort("averageRating", descending=True)
        .head(limit)
        .select(["primaryTitle", "startYear", "averageRating", "numVotes"])
    )

    load_time = time.time() - start_time

    for i, row in enumerate(top_movies.iter_rows(named=True), 1):
        year = row["startYear"] if row["startYear"] is not None else "Unknown"
        print(
            f"{i:2d}. {row['primaryTitle']} ({year}) - ⭐ {row['averageRating']}/10 ({row['numVotes']:,} votes)"
        )

    print(f"\n⏱️  Processing time: {load_time:.2f} seconds")


def example_duckdb_top_movies(min_votes: int = 10000, limit: int = 10):
    """Find top rated movies using DuckDB (SQL on files)"""
    print(f"\n🏆 Top {limit} Rated Movies (DuckDB SQL, min {min_votes:,} votes)")
    print("=" * 60)

    start_time = time.time()

    # Query TSV files directly with SQL
    result = duckdb.sql(
        f"""
        SELECT 
            b.primaryTitle,
            b.startYear,
            r.averageRating,
            r.numVotes
        FROM 'title.basics.tsv' b
        JOIN 'title.ratings.tsv' r ON b.tconst = r.tconst
        WHERE b.titleType = 'movie'
        AND r.numVotes >= {min_votes}
        ORDER BY r.averageRating DESC
        LIMIT {limit}
    """
    ).df()

    load_time = time.time() - start_time

    for i, (_, row) in enumerate(result.iterrows(), 1):
        year = int(row["startYear"]) if pd.notna(row["startYear"]) else "Unknown"
        print(
            f"{i:2d}. {row['primaryTitle']} ({year}) - ⭐ {row['averageRating']}/10 ({row['numVotes']:,} votes)"
        )

    print(f"\n⏱️  Processing time: {load_time:.2f} seconds")


def example_find_actor_movies(actor_name: str, limit: int = 15):
    """Find movies for a specific actor using polars"""
    print(f"\n🎭 Movies featuring '{actor_name}' (polars)")
    print("=" * 60)

    start_time = time.time()

    # Load data (handle IMDb's \N null values and TSV quoting)
    names = pl.read_csv(
        "name.basics.tsv",
        separator="\t",
        null_values=["\\N"],
        quote_char=None,
        ignore_errors=True,
    )
    principals = pl.read_csv(
        "title.principals.tsv",
        separator="\t",
        null_values=["\\N"],
        quote_char=None,
        ignore_errors=True,
    )
    titles = pl.read_csv(
        "title.basics.tsv",
        separator="\t",
        null_values=["\\N"],
        quote_char=None,
        ignore_errors=True,
    )

    # Find actor and their movies
    actor_movies = (
        names.filter(pl.col("primaryName").str.contains(actor_name, literal=False))
        .join(principals, on="nconst")
        .join(titles, on="tconst")
        .filter(pl.col("titleType") == "movie")
        .sort("startYear", descending=True)
        .head(limit)
        .select(["primaryTitle", "startYear", "category", "characters"])
    )

    load_time = time.time() - start_time

    if actor_movies.height == 0:
        print(f"No movies found for '{actor_name}'")
        return

    for row in actor_movies.iter_rows(named=True):
        year = row["startYear"] if row["startYear"] is not None else "Unknown"
        characters = f" as {row['characters']}" if row["characters"] else ""
        print(f"• {row['primaryTitle']} ({year}) - {row['category']}{characters}")

    print(f"\n⏱️  Processing time: {load_time:.2f} seconds")


def example_movies_by_genre(genre: str, limit: int = 10):
    """Find top rated movies by genre using DuckDB"""
    print(f"\n🎬 Top {limit} {genre} Movies (DuckDB)")
    print("=" * 60)

    start_time = time.time()

    result = duckdb.sql(
        f"""
        SELECT 
            b.primaryTitle,
            b.startYear,
            r.averageRating,
            r.numVotes
        FROM 'title.basics.tsv' b
        JOIN 'title.ratings.tsv' r ON b.tconst = r.tconst
        WHERE b.titleType = 'movie'
        AND b.genres LIKE '%{genre}%'
        AND r.numVotes >= 5000
        ORDER BY r.averageRating DESC
        LIMIT {limit}
    """
    ).df()

    load_time = time.time() - start_time

    for _, row in result.iterrows():
        year = int(row["startYear"]) if pd.notna(row["startYear"]) else "Unknown"
        print(
            f"• {row['primaryTitle']} ({year}) - ⭐ {row['averageRating']}/10 ({row['numVotes']:,} votes)"
        )

    print(f"\n⏱️  Processing time: {load_time:.2f} seconds")


def example_tv_show_episodes(show_name: str, limit: int = 20):
    """Find episodes of a TV show using pandas"""
    print(f"\n📺 Episodes of '{show_name}' (pandas)")
    print("=" * 60)

    start_time = time.time()

    # Load data
    titles = pd.read_csv("title.basics.tsv", sep="\t", low_memory=False)
    episodes = pd.read_csv("title.episode.tsv", sep="\t")
    ratings = pd.read_csv("title.ratings.tsv", sep="\t")

    # Find the show
    shows = titles[
        (titles["titleType"].isin(["tvSeries", "tvMiniSeries"]))
        & (titles["primaryTitle"].str.contains(show_name, case=False, na=False))
    ]

    if shows.empty:
        print(f"No TV show found matching '{show_name}'")
        return

    show = shows.iloc[0]
    print(f"Found show: {show['primaryTitle']} ({show['startYear']})")

    # Find episodes
    show_episodes = pd.merge(
        episodes[episodes["parentTconst"] == show["tconst"]],
        titles[["tconst", "primaryTitle"]],
        on="tconst",
    )

    # Add ratings
    show_episodes = pd.merge(show_episodes, ratings, on="tconst", how="left")

    # Sort and limit
    show_episodes = show_episodes.sort_values(["seasonNumber", "episodeNumber"]).head(
        limit
    )

    load_time = time.time() - start_time

    for _, episode in show_episodes.iterrows():
        season = (
            int(episode["seasonNumber"])
            if pd.notna(episode["seasonNumber"])
            else "Unknown"
        )
        ep_num = (
            int(episode["episodeNumber"])
            if pd.notna(episode["episodeNumber"])
            else "Unknown"
        )
        rating = (
            f"⭐ {episode['averageRating']}/10"
            if pd.notna(episode["averageRating"])
            else "No rating"
        )
        print(f"S{season:02d}E{ep_num:02d}: {episode['primaryTitle']} - {rating}")

    print(f"\n⏱️  Processing time: {load_time:.2f} seconds")


def example_dataset_statistics():
    """Show various dataset statistics using DuckDB"""
    print(f"\n📊 Dataset Statistics (DuckDB)")
    print("=" * 60)

    start_time = time.time()

    # Total counts
    total_titles = duckdb.sql("SELECT COUNT(*) FROM 'title.basics.tsv'").fetchone()[0]
    total_movies = duckdb.sql(
        "SELECT COUNT(*) FROM 'title.basics.tsv' WHERE titleType = 'movie'"
    ).fetchone()[0]
    total_people = duckdb.sql("SELECT COUNT(*) FROM 'name.basics.tsv'").fetchone()[0]

    print(f"Total titles: {total_titles:,}")
    print(f"Total movies: {total_movies:,}")
    print(f"Total people: {total_people:,}")

    # Movies by decade
    print(f"\n📅 Movies by Decade:")
    decades = duckdb.sql(
        """
        SELECT 
            (startYear / 10) * 10 as decade,
            COUNT(*) as count
        FROM 'title.basics.tsv'
        WHERE titleType = 'movie'
        AND startYear IS NOT NULL
        AND startYear >= 1900
        GROUP BY (startYear / 10) * 10
        ORDER BY decade
    """
    ).df()

    for _, row in decades.tail(10).iterrows():  # Last 10 decades
        print(f"  {int(row['decade'])}s: {row['count']:,} movies")

    # Top title types
    print(f"\n🎭 Title Types:")
    title_types = duckdb.sql(
        """
        SELECT titleType, COUNT(*) as count
        FROM 'title.basics.tsv'
        WHERE titleType IS NOT NULL
        GROUP BY titleType
        ORDER BY count DESC
        LIMIT 10
    """
    ).df()

    for _, row in title_types.iterrows():
        print(f"  {row['titleType']}: {row['count']:,}")

    load_time = time.time() - start_time
    print(f"\n⏱️  Processing time: {load_time:.2f} seconds")


def example_performance_comparison():
    """Compare performance of pandas vs polars vs DuckDB"""
    print(f"\n⚡ Performance Comparison: Loading title.ratings.tsv")
    print("=" * 60)

    # Pandas
    start_time = time.time()
    df_pandas = pd.read_csv("title.ratings.tsv", sep="\t")
    pandas_time = time.time() - start_time
    print(f"📊 Pandas:  {pandas_time:.2f}s - {len(df_pandas):,} rows")

    # Polars
    start_time = time.time()
    df_polars = pl.read_csv(
        "title.ratings.tsv",
        separator="\t",
        null_values=["\\N"],
        quote_char=None,
        ignore_errors=True,
    )
    polars_time = time.time() - start_time
    print(f"⚡ Polars:  {polars_time:.2f}s - {df_polars.height:,} rows")

    # DuckDB
    start_time = time.time()
    df_duckdb = duckdb.sql("SELECT COUNT(*) FROM 'title.ratings.tsv'").fetchone()[0]
    duckdb_time = time.time() - start_time
    print(f"🦆 DuckDB:  {duckdb_time:.2f}s - {df_duckdb:,} rows")

    print(f"\n🏆 Speed ranking:")
    times = [("Pandas", pandas_time), ("Polars", polars_time), ("DuckDB", duckdb_time)]
    times.sort(key=lambda x: x[1])
    for i, (name, time_taken) in enumerate(times, 1):
        print(f"  {i}. {name}: {time_taken:.2f}s")


def main():
    """Run all examples"""
    print("🎬 IMDb Dataset Analysis Examples")
    print("=" * 60)

    if not check_files_exist():
        print("\n❌ Please make sure all TSV files are in the current directory")
        return

    show_file_info()

    try:
        # Top movies with different tools
        example_pandas_top_movies(min_votes=50000, limit=5)
        example_polars_top_movies(min_votes=50000, limit=5)
        example_duckdb_top_movies(min_votes=50000, limit=5)

        # Actor analysis
        example_find_actor_movies("Tom Hanks", limit=10)

        # Genre analysis
        example_movies_by_genre("Action", limit=8)

        # TV show analysis
        example_tv_show_episodes("Breaking Bad", limit=10)

        # Dataset statistics
        example_dataset_statistics()

        # Performance comparison
        example_performance_comparison()

        print(f"\n✅ All examples completed successfully!")
        print(f"\n💡 Tips:")
        print(f"  • Use polars for fast data processing")
        print(f"  • Use DuckDB for complex SQL queries")
        print(f"  • Use pandas for familiar DataFrame operations")
        print(f"  • Process large files in chunks for memory efficiency")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print(
            f"Make sure all required packages are installed: pip install -r requirements.txt"
        )


if __name__ == "__main__":
    main()
