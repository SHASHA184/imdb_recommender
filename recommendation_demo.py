#!/usr/bin/env python3
"""
IMDb Recommendation System Demo
Interactive examples showcasing all recommendation features
"""

from recommendation_system import IMDbRecommendationSystem
import time


def demo_content_based_recommendations(recommender):
    """Demo content-based recommendations"""
    print("\n" + "=" * 60)
    print("🎯 CONTENT-BASED RECOMMENDATIONS")
    print("=" * 60)

    # Example movies to demonstrate
    demo_movies = [
        "The Dark Knight",
        "Inception",
        "Pulp Fiction",
        "The Matrix",
        "Forrest Gump",
    ]

    for movie in demo_movies:
        print(f"\n🎬 Finding movies similar to '{movie}'...")

        try:
            recommendations = recommender.get_content_based_recommendations(
                movie, n_recommendations=5, min_rating=7.0, min_votes=10000
            )

            if recommendations:
                print(f"✅ Found {len(recommendations)} similar movies:")
                for i, rec in enumerate(recommendations, 1):
                    year = f"({rec['year']})" if rec["year"] else ""
                    print(
                        f"  {i}. {rec['title']} {year} - ⭐ {rec['rating']}/10 "
                        f"({rec['votes']:,} votes) - Similarity: {rec['similarity_score']}"
                    )
            else:
                print("❌ No similar movies found")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 50)


def demo_popular_recommendations(recommender):
    """Demo popular recommendations by genre and year"""
    print("\n" + "=" * 60)
    print("🔥 POPULAR RECOMMENDATIONS")
    print("=" * 60)

    # Popular by genre
    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror"]

    for genre in genres:
        print(f"\n🎭 Top {genre} Movies:")

        try:
            recommendations = recommender.get_recommendations_by_genre(
                genre, n_recommendations=3, min_rating=7.5, min_votes=50000
            )

            for i, rec in enumerate(recommendations, 1):
                year = f"({rec['year']})" if rec["year"] else ""
                print(
                    f"  {i}. {rec['title']} {year} - ⭐ {rec['rating']}/10 "
                    f"({rec['votes']:,} votes)"
                )

        except Exception as e:
            print(f"❌ Error: {e}")

    # Popular by year
    print(f"\n📅 Popular Movies by Year:")
    years = [2020, 2019, 2018, 2010, 2000]

    for year in years:
        print(f"\n🗓️  Best Movies from {year}:")

        try:
            recommendations = recommender.get_recommendations_by_year(
                year, n_recommendations=3, min_rating=7.0, min_votes=25000
            )

            for i, rec in enumerate(recommendations, 1):
                print(
                    f"  {i}. {rec['title']} - ⭐ {rec['rating']}/10 "
                    f"({rec['votes']:,} votes)"
                )

        except Exception as e:
            print(f"❌ Error: {e}")


def demo_hybrid_recommendations(recommender):
    """Demo hybrid recommendations"""
    print("\n" + "=" * 60)
    print("🔀 HYBRID RECOMMENDATIONS")
    print("=" * 60)

    demo_movies = ["The Dark Knight", "Inception", "Avengers"]

    for movie in demo_movies:
        print(f"\n🎬 Hybrid recommendations for '{movie}':")
        print("   (Combining content similarity + popularity)")

        try:
            recommendations = recommender.get_hybrid_recommendations(
                movie,
                n_recommendations=5,
                content_weight=0.6,
                popularity_weight=0.4,
                min_rating=7.0,
                min_votes=10000,
            )

            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    year = f"({rec['year']})" if rec["year"] else ""
                    hybrid_score = rec.get("hybrid_score", 0)
                    print(
                        f"  {i}. {rec['title']} {year} - ⭐ {rec['rating']}/10 "
                        f"- Hybrid Score: {hybrid_score:.3f}"
                    )
            else:
                print("❌ No hybrid recommendations found")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 50)


def demo_search_functionality(recommender):
    """Demo search functionality"""
    print("\n" + "=" * 60)
    print("🔍 SEARCH FUNCTIONALITY")
    print("=" * 60)

    search_queries = [
        "Avengers",
        "Batman",
        "Lord of the Rings",
        "Star Wars",
        "Godfather",
    ]

    for query in search_queries:
        print(f"\n🔍 Search results for '{query}':")

        try:
            results = recommender.search_titles(query, limit=5)

            if results:
                for i, result in enumerate(results, 1):
                    year = f"({result['year']})" if result["year"] else ""
                    rating = (
                        f"⭐ {result['rating']}/10" if result["rating"] else "No rating"
                    )
                    print(
                        f"  {i}. {result['title']} {year} - {rating} "
                        f"({result['votes']:,} votes)"
                    )
            else:
                print("❌ No search results found")

        except Exception as e:
            print(f"❌ Error: {e}")


def demo_tv_shows(recommender):
    """Demo TV show recommendations"""
    print("\n" + "=" * 60)
    print("📺 TV SHOW RECOMMENDATIONS")
    print("=" * 60)

    # Popular TV shows
    print(f"\n📺 Top Rated TV Series:")

    try:
        tv_recommendations = recommender.get_popular_recommendations(
            title_type="tvSeries", n_recommendations=10, min_votes=25000
        )

        for i, rec in enumerate(tv_recommendations, 1):
            year = f"({rec['year']})" if rec["year"] else ""
            print(
                f"  {i}. {rec['title']} {year} - ⭐ {rec['rating']}/10 "
                f"({rec['votes']:,} votes)"
            )

    except Exception as e:
        print(f"❌ Error: {e}")

    # TV shows by genre
    tv_genres = ["Drama", "Comedy", "Crime"]

    for genre in tv_genres:
        print(f"\n📺 Top {genre} TV Series:")

        try:
            recommendations = recommender.get_recommendations_by_genre(
                genre,
                title_type="tvSeries",
                n_recommendations=5,
                min_rating=8.0,
                min_votes=15000,
            )

            for i, rec in enumerate(recommendations, 1):
                year = f"({rec['year']})" if rec["year"] else ""
                print(
                    f"  {i}. {rec['title']} {year} - ⭐ {rec['rating']}/10 "
                    f"({rec['votes']:,} votes)"
                )

        except Exception as e:
            print(f"❌ Error: {e}")


def demo_advanced_filtering(recommender):
    """Demo advanced filtering options"""
    print("\n" + "=" * 60)
    print("🎛️  ADVANCED FILTERING")
    print("=" * 60)

    # High-quality recent movies
    print(f"\n🏆 High-Quality Recent Movies (2020-2024):")

    try:
        recent_movies = recommender.get_popular_recommendations(
            title_type="movie",
            year_range=(2020, 2024),
            n_recommendations=8,
            min_votes=50000,
        )

        for i, rec in enumerate(recent_movies, 1):
            year = f"({rec['year']})" if rec["year"] else ""
            print(
                f"  {i}. {rec['title']} {year} - ⭐ {rec['rating']}/10 "
                f"({rec['votes']:,} votes)"
            )

    except Exception as e:
        print(f"❌ Error: {e}")

    # Cult classics (older movies with high ratings)
    print(f"\n🎭 Cult Classics (1990-2000):")

    try:
        cult_classics = recommender.get_popular_recommendations(
            title_type="movie",
            year_range=(1990, 2000),
            n_recommendations=8,
            min_votes=100000,
        )

        for i, rec in enumerate(cult_classics, 1):
            year = f"({rec['year']})" if rec["year"] else ""
            print(
                f"  {i}. {rec['title']} {year} - ⭐ {rec['rating']}/10 "
                f"({rec['votes']:,} votes)"
            )

    except Exception as e:
        print(f"❌ Error: {e}")


def interactive_demo(recommender):
    """Interactive demo where user can input their own queries"""
    print("\n" + "=" * 60)
    print("🎮 INTERACTIVE DEMO")
    print("=" * 60)

    print("\n🎬 Enter a movie title to get recommendations!")
    print("   Type 'exit' to quit, 'search' to search for movies")

    while True:
        user_input = input("\n➤ Enter movie title (or 'exit'/'search'): ").strip()

        if user_input.lower() == "exit":
            print("👋 Thanks for using the IMDb Recommendation System!")
            break
        elif user_input.lower() == "search":
            search_query = input("➤ Enter search query: ").strip()
            if search_query:
                print(f"\n🔍 Search results for '{search_query}':")
                results = recommender.search_titles(search_query, limit=8)
                if results:
                    for i, result in enumerate(results, 1):
                        year = f"({result['year']})" if result["year"] else ""
                        rating = (
                            f"⭐ {result['rating']}/10"
                            if result["rating"]
                            else "No rating"
                        )
                        print(f"  {i}. {result['title']} {year} - {rating}")
                else:
                    print("❌ No search results found")
        elif user_input:
            print(f"\n🎯 Content-based recommendations for '{user_input}':")
            try:
                recommendations = recommender.get_content_based_recommendations(
                    user_input, n_recommendations=5, min_rating=6.5, min_votes=5000
                )

                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        year = f"({rec['year']})" if rec["year"] else ""
                        print(
                            f"  {i}. {rec['title']} {year} - ⭐ {rec['rating']}/10 "
                            f"(Similarity: {rec['similarity_score']})"
                        )
                else:
                    print(
                        "❌ No recommendations found. Try searching for the exact title first."
                    )
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main demo function"""
    print("🎬 IMDb Recommendation System - Complete Demo")
    print("=" * 60)
    print("🚀 Initializing recommendation system...")

    start_time = time.time()

    try:
        # Initialize the recommendation system
        recommender = IMDbRecommendationSystem()

        init_time = time.time() - start_time
        print(f"✅ System initialized in {init_time:.2f} seconds")

        # Run all demos
        print("\n🎯 Starting comprehensive demo...")

        # 1. Content-based recommendations
        demo_content_based_recommendations(recommender)

        # 2. Popular recommendations
        demo_popular_recommendations(recommender)

        # 3. Hybrid recommendations
        demo_hybrid_recommendations(recommender)

        # 4. Search functionality
        demo_search_functionality(recommender)

        # 5. TV show recommendations
        demo_tv_shows(recommender)

        # 6. Advanced filtering
        demo_advanced_filtering(recommender)

        # 7. Interactive demo
        interactive_demo(recommender)

    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("💡 Make sure all IMDb TSV files are in the current directory")
        print("💡 Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
