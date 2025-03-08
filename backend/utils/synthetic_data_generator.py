
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_movie_data(n_movies=1000):
    """Generate synthetic movie data with features"""
    
    # Movie IDs
    movie_ids = list(range(1, n_movies + 1))
    
    # Movie titles - random combinations of adjectives and nouns
    adjectives = ['Dark', 'Bright', 'Silent', 'Loud', 'Lost', 'Hidden', 'Eternal', 'Broken', 
                 'Frozen', 'Golden', 'Savage', 'Divine', 'Ancient', 'Fallen', 'Rising']
    nouns = ['Knight', 'Dawn', 'City', 'Dream', 'Kingdom', 'Legacy', 'Heart', 'Star', 
            'Memory', 'Shadow', 'Empire', 'Ocean', 'Future', 'Hero', 'Destiny']
    
    titles = []
    for _ in range(n_movies):
        title = f"The {random.choice(adjectives)} {random.choice(nouns)}"
        # Ensure unique titles
        while title in titles:
            title = f"The {random.choice(adjectives)} {random.choice(nouns)}"
        titles.append(title)
    
    # Genres - movies can belong to multiple genres
    all_genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror', 
                 'Mystery', 'Romance', 'Sci-Fi', 'Thriller']
    
    # Movie features
    genres = []
    release_years = []
    popularity_scores = []
    durations = []  # in minutes
    
    for _ in range(n_movies):
        # Assign 1-3 genres to each movie
        n_genres = random.randint(1, 3)
        movie_genres = random.sample(all_genres, n_genres)
        genres.append("|".join(movie_genres))
        
        # Release year (between 1980 and 2023)
        release_years.append(random.randint(1980, 2023))
        
        # Popularity score (1-100)
        popularity_scores.append(random.randint(1, 100))
        
        # Duration (70-180 minutes)
        durations.append(random.randint(70, 180))
    
    # Create movie dataframe
    movies_df = pd.DataFrame({
        'movieId': movie_ids,
        'title': titles,
        'genres': genres,
        'releaseYear': release_years,
        'popularityScore': popularity_scores,
        'duration': durations
    })
    
    return movies_df

def generate_user_data(n_users=5000):
    """Generate synthetic user data with features"""
    
    # User IDs
    user_ids = list(range(1, n_users + 1))
    
    # User demographics
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    genders = ['M', 'F', 'Other']
    
    ages = []
    user_genders = []
    activity_levels = []  # How active the user is (1-10)
    join_dates = []
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    
    for _ in range(n_users):
        # Age group
        age_group = random.choice(age_groups)
        ages.append(age_group)
        
        # Gender
        gender = random.choice(genders)
        user_genders.append(gender)
        
        # Activity level
        activity_levels.append(random.randint(1, 10))
        
        # Join date
        random_days = random.randint(0, date_range)
        join_date = start_date + timedelta(days=random_days)
        join_dates.append(join_date.strftime('%Y-%m-%d'))
    
    # Create user dataframe
    users_df = pd.DataFrame({
        'userId': user_ids,
        'ageGroup': ages,
        'gender': user_genders,
        'activityLevel': activity_levels,
        'joinDate': join_dates
    })
    
    return users_df

def generate_ratings(users_df, movies_df, sparsity=0.98):
    """Generate synthetic rating data with realistic patterns"""
    n_users = len(users_df)
    n_movies = len(movies_df)
    
    # Calculate how many ratings we'll generate based on sparsity
    # sparsity = 1 - (n_ratings / (n_users * n_movies))
    # Therefore: n_ratings = (1 - sparsity) * (n_users * n_movies)
    n_ratings = int((1 - sparsity) * (n_users * n_movies))
    
    # Ratings will follow these patterns:
    # 1. Popular movies get more ratings
    # 2. Active users give more ratings
    # 3. Ratings follow a slightly positive skew (most around 3-4)
    
    # Probability distribution for users based on activity level
    user_probs = np.array(users_df['activityLevel']) ** 1.5
    user_probs = user_probs / user_probs.sum()
    
    # Probability distribution for movies based on popularity
    movie_probs = np.array(movies_df['popularityScore']) ** 1.3
    movie_probs = movie_probs / movie_probs.sum()
    
    # Sample user-movie pairs
    user_indices = np.random.choice(n_users, size=n_ratings, p=user_probs, replace=True)
    movie_indices = np.random.choice(n_movies, size=n_ratings, p=movie_probs, replace=True)
    
    # Convert to actual IDs
    user_ids = users_df.iloc[user_indices]['userId'].values
    movie_ids = movies_df.iloc[movie_indices]['movieId'].values
    
    # Create pairs and remove duplicates
    pairs = list(zip(user_ids, movie_ids))
    pairs = list(set(pairs))  # Remove duplicates
    
    # If we lost too many pairs due to duplicates, generate more
    while len(pairs) < n_ratings * 0.9:  # Aim for at least 90% of desired ratings
        additional_users = np.random.choice(n_users, size=n_ratings//5, p=user_probs, replace=True)
        additional_movies = np.random.choice(n_movies, size=n_ratings//5, p=movie_probs, replace=True)
        additional_user_ids = users_df.iloc[additional_users]['userId'].values
        additional_movie_ids = movies_df.iloc[additional_movies]['movieId'].values
        additional_pairs = list(zip(additional_user_ids, additional_movie_ids))
        pairs.extend(additional_pairs)
        pairs = list(set(pairs))  # Remove duplicates again
    
    user_ids, movie_ids = zip(*pairs)
    
    # Generate ratings with a positive skew
    # More 4s and 5s than 1s and 2s
    rating_distribution = [0.05, 0.1, 0.2, 0.4, 0.25]  # Probabilities for ratings 1-5
    ratings = np.random.choice([1, 2, 3, 4, 5], size=len(pairs), p=rating_distribution)
    
    # Generate timestamps
    start_ts = datetime(2020, 1, 1).timestamp()
    end_ts = datetime(2023, 12, 31).timestamp()
    timestamps = np.random.uniform(start_ts, end_ts, size=len(pairs)).astype(int)
    
    # Create ratings dataframe
    ratings_df = pd.DataFrame({
        'userId': user_ids,
        'movieId': movie_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    return ratings_df

def analyze_data(users_df, movies_df, ratings_df):
    """Analyze and visualize the generated data"""
    
    print(f"Generated {len(users_df)} users, {len(movies_df)} movies, and {len(ratings_df)} ratings")
    
    # Rating distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=ratings_df)
    plt.title('Rating Distribution')
    plt.savefig('rating_distribution.png')
    
    # Ratings per user histogram
    ratings_per_user = ratings_df.groupby('userId').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings_per_user, bins=30)
    plt.title('Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.savefig('ratings_per_user.png')
    
    # Ratings per movie histogram
    ratings_per_movie = ratings_df.groupby('movieId').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings_per_movie, bins=30)
    plt.title('Ratings per Movie')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Movies')
    plt.savefig('ratings_per_movie.png')
    
    # Average rating by release year
    movies_with_ratings = ratings_df.merge(movies_df, on='movieId')
    avg_rating_by_year = movies_with_ratings.groupby('releaseYear')['rating'].mean()
    plt.figure(figsize=(12, 6))
    avg_rating_by_year.plot(kind='line')
    plt.title('Average Rating by Release Year')
    plt.xlabel('Release Year')
    plt.ylabel('Average Rating')
    plt.grid(True)
    plt.savefig('avg_rating_by_year.png')
    
    # Matrix sparsity visualization (sample of users and movies)
    sample_users = random.sample(list(users_df['userId']), min(100, len(users_df)))
    sample_movies = random.sample(list(movies_df['movieId']), min(200, len(movies_df)))
    
    sample_ratings = ratings_df[
        ratings_df['userId'].isin(sample_users) & 
        ratings_df['movieId'].isin(sample_movies)
    ]
    
    # Create a pivot table
    matrix = sample_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix.astype(bool), cbar=False, cmap='Blues', yticklabels=False, xticklabels=False)
    plt.title('Rating Matrix Sparsity (Sample)')
    plt.savefig('matrix_sparsity.png')
    
    return {
        'n_users': len(users_df),
        'n_movies': len(movies_df),
        'n_ratings': len(ratings_df),
        'sparsity': 1 - (len(ratings_df) / (len(users_df) * len(movies_df))),
        'avg_rating': ratings_df['rating'].mean(),
        'ratings_per_user_avg': ratings_per_user.mean(),
        'ratings_per_movie_avg': ratings_per_movie.mean()
    }

def create_train_test_split(ratings_df, test_size=0.2):
    """Split the ratings into training and test sets"""
    
    # Sort by timestamp to simulate a temporal split
    ratings_df = ratings_df.sort_values('timestamp')
    
    # Calculate the split index
    split_idx = int(len(ratings_df) * (1 - test_size))
    
    # Split the data
    train_df = ratings_df.iloc[:split_idx]
    test_df = ratings_df.iloc[split_idx:]
    
    # Ensure all users and movies in test set are also in training set (for cold start evaluation)
    test_users = set(test_df['userId'].unique())
    test_movies = set(test_df['movieId'].unique())
    
    train_users = set(train_df['userId'].unique())
    train_movies = set(train_df['movieId'].unique())
    
    # Find users and movies that are in test but not in train
    cold_start_users = test_users - train_users
    cold_start_movies = test_movies - train_movies
    
    # Remove ratings with cold start users or movies from test set
    if cold_start_users or cold_start_movies:
        print(f"Removing {len(cold_start_users)} cold start users and {len(cold_start_movies)} cold start movies from test set")
        test_df = test_df[
            ~test_df['userId'].isin(cold_start_users) & 
            ~test_df['movieId'].isin(cold_start_movies)
        ]
    
    return train_df, test_df

def save_data(users_df, movies_df, ratings_df, train_df, test_df):
    """Save all dataframes to CSV files"""
    
    users_df.to_csv('users.csv', index=False)
    movies_df.to_csv('movies.csv', index=False)
    ratings_df.to_csv('ratings.csv', index=False)
    train_df.to_csv('ratings_train.csv', index=False)
    test_df.to_csv('ratings_test.csv', index=False)
    
    print("Data saved to CSV files")

def main():
    # Generate data
    print("Generating movie data...")
    movies_df = generate_movie_data(n_movies=1000)
    
    print("Generating user data...")
    users_df = generate_user_data(n_users=5000)
    
    print("Generating ratings data...")
    ratings_df = generate_ratings(users_df, movies_df, sparsity=0.98)
    
    # Analyze data
    print("Analyzing data...")
    stats = analyze_data(users_df, movies_df, ratings_df)
    print("Data statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create train/test split
    print("Creating train/test split...")
    train_df, test_df = create_train_test_split(ratings_df)
    
    # Save data
    print("Saving data...")
    save_data(users_df, movies_df, ratings_df, train_df, test_df)
    
    print("Done!")

if __name__ == "__main__":
    main()
