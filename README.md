# 🎬 IMDb Dataset Analysis & Recommendation System

A comprehensive Python toolkit for analyzing the complete IMDb dataset and providing intelligent movie/TV show recommendations using **pandas**, **polars**, and **DuckDB** - no database required!

## 🚀 New: Recommendation System

### 🎯 Multiple Recommendation Algorithms

-   **Content-Based Filtering**: Recommends based on movie genres, ratings, and metadata
-   **Popularity-Based**: Suggests top-rated movies by genre, year, or overall popularity
-   **Hybrid Approach**: Combines content similarity with popularity metrics for balanced recommendations

### 🔍 Advanced Features

-   **Smart Search**: Find movies by partial title matching
-   **Genre Filtering**: Get recommendations by specific genres
-   **Year-Based Filtering**: Discover movies from specific time periods
-   **Quality Metrics**: Evaluate recommendation diversity, novelty, and quality
-   **Interactive Demo**: Test recommendations with your favorite movies

### 📊 Evaluation Tools

-   **Diversity Scoring**: Measures genre variety in recommendations
-   **Novelty Assessment**: Balances popular vs. niche recommendations
-   **Performance Benchmarking**: Compare different recommendation algorithms
-   **Quality Metrics**: Comprehensive evaluation of recommendation systems

## 🚀 Why TSV-Only Approach?

-   **💾 Storage Efficient**: No database overhead, just the original ~8.5GB
-   **🚀 Fast Setup**: No containers, no imports, just run
-   **🔧 Flexible**: Use pandas, polars, or DuckDB for different needs
-   **📊 Analysis-Ready**: Perfect for data science and exploration
-   **🎯 Direct Access**: Query files directly without preprocessing

## 📁 Project Structure

```
imdb_recommender/
├── 📊 examples.py                    # Analysis examples
├── 🎯 recommendation_system.py       # Main recommendation engine
├── 🎮 recommendation_demo.py         # Interactive demo & examples
├── 📈 recommendation_evaluation.py   # Quality metrics & benchmarking
├── 🐍 requirements.txt               # Python dependencies
├── 📖 README.md                     # This file
├── 📂 title.basics.tsv              # 969MB - Movie/TV show info
├── 📂 name.basics.tsv               # 851MB - Person info
├── 📂 title.ratings.tsv             # 27MB - Ratings
├── 📂 title.crew.tsv                # 370MB - Directors/Writers
├── 📂 title.episode.tsv             # 226MB - Episode info
├── 📂 title.akas.tsv                # 2.5GB - Alternative titles
└── 📂 title.principals.tsv          # 3.9GB - Cast/Crew details
```

## 🗄️ Dataset Overview

### File Contents

-   **title.basics.tsv**: Primary movie/TV show information
-   **name.basics.tsv**: Person names, birth/death years
-   **title.ratings.tsv**: User ratings and vote counts
-   **title.crew.tsv**: Directors and writers
-   **title.episode.tsv**: TV episode details
-   **title.akas.tsv**: Alternative titles (different languages/regions)
-   **title.principals.tsv**: Cast and crew with roles

### Data Relationships

-   `tconst`: Title identifier (movies/shows)
-   `nconst`: Name identifier (people)
-   Files can be joined on these keys

## 🚀 Quick Start

### Prerequisites

-   Python 3.8+
-   ~8.5GB disk space (no additional database storage needed)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic analysis examples
python examples.py

# Try the recommendation system
python recommendation_demo.py

# Run evaluation benchmarks
python recommendation_evaluation.py
```

## 🎯 Recommendation System Usage

### Basic Usage

```python
from recommendation_system import IMDbRecommendationSystem

# Initialize the system
recommender = IMDbRecommendationSystem()

# Get content-based recommendations
recommendations = recommender.get_content_based_recommendations(
    "The Dark Knight",
    n_recommendations=5,
    min_rating=7.0,
    min_votes=10000
)

# Print recommendations
recommender.print_recommendations(recommendations)
```

### Advanced Features

```python
# Get popular movies by genre
action_movies = recommender.get_recommendations_by_genre(
    "Action",
    n_recommendations=10,
    min_rating=7.5
)

# Get hybrid recommendations (content + popularity)
hybrid_recs = recommender.get_hybrid_recommendations(
    "Inception",
    content_weight=0.7,
    popularity_weight=0.3
)

# Search for movies
results = recommender.search_titles("Avengers", limit=5)

# Get recommendations by year
movies_2020 = recommender.get_recommendations_by_year(
    2020,
    n_recommendations=10
)
```

## 📊 Data Analysis Examples

### 1. Using Pandas (Memory-based)

```python
import pandas as pd

# Load and analyze ratings
ratings = pd.read_csv('title.ratings.tsv', sep='\t')
top_rated = ratings.nlargest(10, 'averageRating')

# Join with title info
titles = pd.read_csv('title.basics.tsv', sep='\t')
result = pd.merge(ratings, titles, on='tconst')
```

### 2. Using Polars (Fast & Memory-efficient)

```python
import polars as pl

# Lightning-fast processing
ratings = pl.read_csv('title.ratings.tsv', separator='\t')
top_movies = (
    ratings
    .filter(pl.col('numVotes') > 10000)
    .sort('averageRating', descending=True)
    .head(10)
)
```

### 3. Using DuckDB (SQL on Files)

```python
import duckdb

# Query TSV files directly with SQL
result = duckdb.sql("""
    SELECT
        b.primaryTitle,
        r.averageRating,
        r.numVotes
    FROM 'title.basics.tsv' b
    JOIN 'title.ratings.tsv' r ON b.tconst = r.tconst
    WHERE r.numVotes > 50000
    ORDER BY r.averageRating DESC
    LIMIT 10
""").df()
```

## 🔍 Common Analysis Patterns

### Find Top Movies by Genre

```python
# Filter by genre and get top rated
movies = pl.read_csv('title.basics.tsv', separator='\t')
ratings = pl.read_csv('title.ratings.tsv', separator='\t')

top_action = (
    movies
    .filter(pl.col('genres').str.contains('Action'))
    .join(ratings, on='tconst')
    .filter(pl.col('numVotes') > 1000)
    .sort('averageRating', descending=True)
    .head(20)
)
```

### Actor Filmography

```python
# Find all movies for a specific actor
principals = pl.read_csv('title.principals.tsv', separator='\t')
names = pl.read_csv('name.basics.tsv', separator='\t')

tom_hanks_movies = (
    names
    .filter(pl.col('primaryName').str.contains('Tom Hanks'))
    .join(principals, on='nconst')
    .join(movies, on='tconst')
    .select(['primaryTitle', 'startYear', 'category'])
)
```

### TV Show Analysis

```python
# Analyze TV show episodes
episodes = pl.read_csv('title.episode.tsv', separator='\t')
ratings = pl.read_csv('title.ratings.tsv', separator='\t')

# Find highest rated episodes
top_episodes = (
    episodes
    .join(ratings, on='tconst')
    .join(movies, on='tconst')
    .filter(pl.col('numVotes') > 500)
    .sort('averageRating', descending=True)
    .head(50)
)
```

## ⚡ Performance Tips

### For Large Files (2.5GB+)

```python
# Use chunked reading for memory efficiency
chunks = pd.read_csv('title.akas.tsv', sep='\t', chunksize=100000)
for chunk in chunks:
    # Process chunk by chunk
    process_chunk(chunk)
```

### For Complex Queries

```python
# DuckDB excels at complex SQL operations
conn = duckdb.connect()
conn.execute("""
    CREATE VIEW top_movies AS
    SELECT * FROM 'title.basics.tsv'
    WHERE titleType = 'movie'
    AND startYear > 2000
""")
```

### Memory Optimization

```python
# Use appropriate data types
dtypes = {
    'tconst': 'string',
    'averageRating': 'float32',
    'numVotes': 'int32'
}
```

## 🎬 Interactive Demo

Run the interactive demo to test the recommendation system:

```bash
python recommendation_demo.py
```

Features:

-   **Content-based recommendations** for popular movies
-   **Popular movies by genre** (Action, Comedy, Drama, etc.)
-   **Hybrid recommendations** combining multiple algorithms
-   **Search functionality** with fuzzy matching
-   **TV show recommendations** with genre filtering
-   **Interactive mode** where you can input your own queries

## 📊 Evaluation & Benchmarking

Evaluate the recommendation system performance:

```bash
python recommendation_evaluation.py
```

This will generate a comprehensive report including:

-   **Performance benchmarks** for different algorithms
-   **Quality metrics** (diversity, novelty, coverage)
-   **Recommendation patterns** analysis
-   **Improvement suggestions** based on evaluation results

## 🎯 Use Cases

-   **🎬 Movie Recommendations**: Find similar movies by genre/rating
-   **📈 Trend Analysis**: Analyze movie trends over decades
-   **👥 Network Analysis**: Study actor/director collaborations
-   **📊 Rating Patterns**: Understand voting behavior
-   **🌍 International Cinema**: Explore titles across regions
-   **📺 TV Show Deep Dives**: Episode-by-episode analysis

## 💡 Run Examples

```bash
# See the toolkit in action
python examples.py
```

This demonstrates:

-   Loading and exploring each dataset
-   Cross-file joins and relationships
-   Performance comparisons (pandas vs polars vs DuckDB)
-   Real-world analysis scenarios

## 📚 Dataset Documentation

For detailed information about each TSV file format and columns, see the [official IMDb documentation](https://www.imdb.com/interfaces/).

---

**Total Dataset Size**: ~8.5GB  
**Records**: ~40M+ across all files  
**Last Updated**: Based on your TSV file dates  
**Analysis Ready**: ✅ No preprocessing required
