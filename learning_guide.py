#!/usr/bin/env python3
"""
Learning Guide: Understanding the IMDb Recommendation System
Step-by-step explanation of how each component works
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class RecommenderLearningGuide:
    """
    A step-by-step guide to understanding recommendation systems
    """

    def __init__(self):
        print("🎓 Learning Guide: Understanding Recommendation Systems")
        print("=" * 60)

    def explain_content_based_filtering(self):
        """
        Explain how content-based filtering works step by step
        """
        print("\n📚 CONTENT-BASED FILTERING EXPLAINED")
        print("-" * 40)

        # Step 1: Sample movie data
        print("\n1️⃣ Sample Movie Data:")
        movies = pd.DataFrame(
            {
                "title": [
                    "The Dark Knight",
                    "Batman Begins",
                    "Iron Man",
                    "Avengers",
                    "Titanic",
                ],
                "genres": [
                    "Action,Crime,Drama",
                    "Action,Adventure",
                    "Action,Adventure,Sci-Fi",
                    "Action,Adventure,Sci-Fi",
                    "Drama,Romance",
                ],
                "rating": [9.0, 8.2, 7.9, 8.0, 7.8],
                "votes": [2500000, 1400000, 950000, 1300000, 1100000],
            }
        )
        print(movies)

        # Step 2: Feature extraction
        print("\n2️⃣ Feature Extraction:")
        print("Converting genres to TF-IDF vectors...")

        # TF-IDF for genres
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(","), lowercase=False)
        genre_features = tfidf.fit_transform(movies["genres"])

        print(f"Genre feature matrix shape: {genre_features.shape}")
        print("Feature names (genres):", tfidf.get_feature_names_out())
        print("Sample TF-IDF matrix:")
        print(genre_features.toarray()[:3])  # Show first 3 movies

        # Step 3: Numerical features
        print("\n3️⃣ Numerical Features:")
        numerical_features = movies[["rating", "votes"]].values

        # Normalize numerical features
        scaler = StandardScaler()
        numerical_features_scaled = scaler.fit_transform(numerical_features)

        print("Original numerical features:")
        print(numerical_features[:3])
        print("Scaled numerical features:")
        print(numerical_features_scaled[:3])

        # Step 4: Combine features
        print("\n4️⃣ Combining Features:")
        from scipy.sparse import hstack

        combined_features = hstack([genre_features, numerical_features_scaled])
        print(f"Combined feature matrix shape: {combined_features.shape}")

        # Step 5: Calculate similarity
        print("\n5️⃣ Calculating Similarity:")
        similarity_matrix = cosine_similarity(combined_features)
        print("Similarity matrix shape:", similarity_matrix.shape)
        print("Similarity between movies:")

        for i, title in enumerate(movies["title"]):
            print(f"\n{title}:")
            similarities = similarity_matrix[i]
            sorted_indices = np.argsort(similarities)[::-1]

            for j in sorted_indices[1:4]:  # Top 3 similar movies (excluding self)
                print(f"  - {movies.iloc[j]['title']}: {similarities[j]:.3f}")

    def explain_tfidf_step_by_step(self):
        """
        Explain TF-IDF calculation step by step
        """
        print("\n📊 TF-IDF CALCULATION EXPLAINED")
        print("-" * 40)

        # Sample documents (movie genres)
        documents = [
            "Action Crime Drama",
            "Action Adventure",
            "Action Adventure Sci-Fi",
            "Drama Romance",
        ]

        print("Sample documents (movie genres):")
        for i, doc in enumerate(documents):
            print(f"  Movie {i+1}: {doc}")

        # Manual TF-IDF calculation
        print("\n1️⃣ Term Frequency (TF) Calculation:")
        print(
            "TF = (Number of times term appears in document) / (Total terms in document)"
        )

        # Calculate TF for each document
        all_words = set()
        for doc in documents:
            all_words.update(doc.split())

        print(f"All unique terms: {sorted(all_words)}")

        # TF calculation example
        doc_0_words = documents[0].split()
        print(f"\nFor '{documents[0]}':")
        for word in sorted(all_words):
            tf = doc_0_words.count(word) / len(doc_0_words)
            print(
                f"  TF('{word}') = {doc_0_words.count(word)}/{len(doc_0_words)} = {tf:.3f}"
            )

        print("\n2️⃣ Inverse Document Frequency (IDF) Calculation:")
        print("IDF = log(Total documents / Documents containing term)")

        for word in sorted(all_words):
            doc_count = sum(1 for doc in documents if word in doc.split())
            idf = np.log(len(documents) / doc_count)
            print(f"  IDF('{word}') = log({len(documents)}/{doc_count}) = {idf:.3f}")

        print("\n3️⃣ TF-IDF = TF × IDF")
        print("This gives higher weight to:")
        print("  - Terms that appear frequently in a document (high TF)")
        print("  - Terms that are rare across all documents (high IDF)")

        # Compare with sklearn
        print("\n4️⃣ Sklearn TF-IDF Result:")
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(documents)

        feature_names = tfidf.get_feature_names_out()
        print("Feature names:", feature_names)
        print("TF-IDF matrix:")
        print(tfidf_matrix.toarray())

    def explain_cosine_similarity(self):
        """
        Explain cosine similarity calculation
        """
        print("\n📐 COSINE SIMILARITY EXPLAINED")
        print("-" * 40)

        # Sample vectors
        vector_a = np.array([1, 2, 3])
        vector_b = np.array([2, 3, 4])
        vector_c = np.array([1, 0, 0])

        print("Sample vectors:")
        print(f"Vector A: {vector_a}")
        print(f"Vector B: {vector_b}")
        print(f"Vector C: {vector_c}")

        def manual_cosine_similarity(v1, v2):
            """Calculate cosine similarity manually"""
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            return dot_product / (norm_v1 * norm_v2)

        print("\n1️⃣ Cosine Similarity Formula:")
        print("cos(θ) = (A · B) / (||A|| × ||B||)")
        print("Where:")
        print("  A · B = dot product of vectors")
        print("  ||A|| = magnitude (norm) of vector A")
        print("  ||B|| = magnitude (norm) of vector B")

        print("\n2️⃣ Manual Calculation:")
        sim_ab = manual_cosine_similarity(vector_a, vector_b)
        sim_ac = manual_cosine_similarity(vector_a, vector_c)
        sim_bc = manual_cosine_similarity(vector_b, vector_c)

        print(f"Similarity(A, B) = {sim_ab:.3f}")
        print(f"Similarity(A, C) = {sim_ac:.3f}")
        print(f"Similarity(B, C) = {sim_bc:.3f}")

        print("\n3️⃣ Interpretation:")
        print("  1.0 = Identical vectors (same direction)")
        print("  0.0 = Orthogonal vectors (perpendicular)")
        print(" -1.0 = Opposite vectors")

        print("\n4️⃣ Why Cosine Similarity for Recommendations?")
        print("  - Measures angle between vectors, not magnitude")
        print("  - Good for high-dimensional sparse data")
        print("  - Focuses on relative importance, not absolute values")
        print("  - Range [0, 1] for non-negative features")

    def explain_hybrid_approach(self):
        """
        Explain hybrid recommendation approach
        """
        print("\n🔀 HYBRID RECOMMENDATION EXPLAINED")
        print("-" * 40)

        # Sample scores from different methods
        movies = ["Movie A", "Movie B", "Movie C", "Movie D"]
        content_scores = [0.8, 0.6, 0.9, 0.3]
        popularity_scores = [0.7, 0.9, 0.4, 0.8]

        print("Sample recommendation scores:")
        print(f"{'Movie':<10} {'Content':<8} {'Popularity':<10}")
        print("-" * 30)
        for i, movie in enumerate(movies):
            print(f"{movie:<10} {content_scores[i]:<8} {popularity_scores[i]:<10}")

        print("\n1️⃣ Weighted Combination:")
        print("Hybrid Score = α × Content Score + β × Popularity Score")
        print("Where α + β = 1")

        # Different weight combinations
        weight_combinations = [
            (0.7, 0.3, "Content-focused"),
            (0.5, 0.5, "Balanced"),
            (0.3, 0.7, "Popularity-focused"),
        ]

        for alpha, beta, description in weight_combinations:
            print(f"\n2️⃣ {description} (α={alpha}, β={beta}):")
            hybrid_scores = []
            for i in range(len(movies)):
                hybrid_score = alpha * content_scores[i] + beta * popularity_scores[i]
                hybrid_scores.append(hybrid_score)
                print(f"  {movies[i]}: {hybrid_score:.3f}")

            # Sort by hybrid score
            sorted_movies = sorted(
                zip(movies, hybrid_scores), key=lambda x: x[1], reverse=True
            )
            print(f"  Ranking: {[movie for movie, _ in sorted_movies]}")

        print("\n3️⃣ Benefits of Hybrid Approach:")
        print("  - Combines strengths of multiple methods")
        print("  - Reduces weaknesses of individual approaches")
        print("  - More robust and diverse recommendations")
        print("  - Can be tuned for different user preferences")

    def explain_evaluation_metrics(self):
        """
        Explain recommendation evaluation metrics
        """
        print("\n📊 EVALUATION METRICS EXPLAINED")
        print("-" * 40)

        print("1️⃣ DIVERSITY:")
        print("  - Measures variety in recommendations")
        print("  - Higher diversity = more different genres/types")
        print("  - Calculation: Unique genres / Total possible genres")

        print("\n2️⃣ NOVELTY:")
        print("  - Measures how 'unexpected' recommendations are")
        print("  - Higher novelty = less obvious recommendations")
        print("  - Calculation: Inverse of item popularity")

        print("\n3️⃣ COVERAGE:")
        print("  - Measures breadth of recommendations")
        print("  - Higher coverage = more different types/years")
        print("  - Important for exploring catalog diversity")

        print("\n4️⃣ QUALITY SCORE:")
        print("  - Combined metric of multiple factors")
        print("  - Balances accuracy, diversity, and novelty")
        print("  - Calculation: Weighted average of all metrics")

        # Example calculation
        print("\n📈 Example Calculation:")
        recommendations = [
            {
                "title": "Action Movie 1",
                "rating": 8.5,
                "votes": 100000,
                "genres": "Action,Adventure",
            },
            {
                "title": "Drama Movie 1",
                "rating": 8.2,
                "votes": 50000,
                "genres": "Drama,Romance",
            },
            {
                "title": "Comedy Movie 1",
                "rating": 7.8,
                "votes": 200000,
                "genres": "Comedy",
            },
        ]

        # Calculate diversity
        all_genres = set()
        for rec in recommendations:
            all_genres.update(rec["genres"].split(","))

        diversity = len(all_genres) / 20  # Assuming 20 max genres
        print(f"Diversity: {len(all_genres)} unique genres / 20 = {diversity:.3f}")

        # Calculate novelty
        novelty_scores = []
        for rec in recommendations:
            novelty = 1 - min(np.log(rec["votes"]) / np.log(2000000), 1.0)
            novelty_scores.append(max(novelty, 0))

        avg_novelty = np.mean(novelty_scores)
        print(f"Novelty: Average novelty score = {avg_novelty:.3f}")

        # Calculate average rating
        avg_rating = np.mean([rec["rating"] for rec in recommendations])
        print(f"Average Rating: {avg_rating:.1f}/10")

        # Combined quality score
        quality_score = (avg_rating / 10 + diversity + avg_novelty) / 3
        print(f"Quality Score: {quality_score:.3f}")

    def practical_exercise(self):
        """
        Provide a practical exercise for learning
        """
        print("\n🎯 PRACTICAL EXERCISE")
        print("-" * 40)

        print("Try implementing a simple recommender yourself:")
        print(
            """
# Exercise: Build a Simple Book Recommender

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample book data
books = pd.DataFrame({
    'title': ['Harry Potter', 'Lord of the Rings', 'Dune', 'Foundation', 'Twilight'],
    'genre': ['Fantasy Young-Adult', 'Fantasy Adventure', 'Science-Fiction', 
              'Science-Fiction', 'Fantasy Romance'],
    'rating': [4.5, 4.8, 4.2, 4.3, 3.8]
})

# TODO: Implement content-based filtering
# 1. Create TF-IDF vectors for genres
# 2. Calculate cosine similarity
# 3. Function to get recommendations for a book
# 4. Test with your favorite book

def get_book_recommendations(book_title, books_df, n_recommendations=3):
    # Your implementation here
    pass

# Test your implementation
recommendations = get_book_recommendations('Harry Potter', books)
print(recommendations)
        """
        )

        print("\n🎓 Learning Steps:")
        print("1. Start with this simple example")
        print("2. Add more features (author, publication year)")
        print("3. Implement evaluation metrics")
        print("4. Try different similarity measures")
        print("5. Add hybrid approach")
        print("6. Compare with our IMDb system")


def main():
    """
    Run the complete learning guide
    """
    guide = RecommenderLearningGuide()

    print("\n🎯 Choose what to learn:")
    print("1. Content-Based Filtering")
    print("2. TF-IDF Step-by-Step")
    print("3. Cosine Similarity")
    print("4. Hybrid Approach")
    print("5. Evaluation Metrics")
    print("6. Practical Exercise")
    print("7. All Topics")

    choice = input("\nEnter your choice (1-7): ").strip()

    if choice == "1":
        guide.explain_content_based_filtering()
    elif choice == "2":
        guide.explain_tfidf_step_by_step()
    elif choice == "3":
        guide.explain_cosine_similarity()
    elif choice == "4":
        guide.explain_hybrid_approach()
    elif choice == "5":
        guide.explain_evaluation_metrics()
    elif choice == "6":
        guide.practical_exercise()
    elif choice == "7":
        guide.explain_content_based_filtering()
        guide.explain_tfidf_step_by_step()
        guide.explain_cosine_similarity()
        guide.explain_hybrid_approach()
        guide.explain_evaluation_metrics()
        guide.practical_exercise()
    else:
        print("Invalid choice. Running all topics...")
        guide.explain_content_based_filtering()
        guide.explain_tfidf_step_by_step()
        guide.explain_cosine_similarity()
        guide.explain_hybrid_approach()
        guide.explain_evaluation_metrics()
        guide.practical_exercise()


if __name__ == "__main__":
    main()
