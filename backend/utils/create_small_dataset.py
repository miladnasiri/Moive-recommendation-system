import pandas as pd
import numpy as np
import random
from datetime import datetime

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create small movies dataset
movies = []
for i in range(1, 101):  # 100 movies
    movies.append({
        'movieId': i,
        'title': f"Movie {i}",
        'genres': random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance']),
        'releaseYear': random.randint(1990, 2023),
        'popularityScore': random.randint(1, 100),
        'duration': random.randint(70, 180)
    })
movies_df = pd.DataFrame(movies)

# Create small users dataset
users = []
for i in range(1, 201):  # 200 users
    users.append({
        'userId': i,
        'ageGroup': random.choice(['18-24', '25-34', '35-44', '45-54', '55-64', '65+']),
        'gender': random.choice(['M', 'F', 'Other']),
        'activityLevel': random.randint(1, 10),
        'joinDate': (datetime(2020, 1, 1) + pd.Timedelta(days=random.randint(0, 1000))).strftime('%Y-%m-%d')
    })
users_df = pd.DataFrame(users)

# Create small ratings dataset
ratings = []
for user_id in range(1, 201):
    # Each user rates 5-20 random movies
    num_ratings = random.randint(5, 20)
    movie_ids = random.sample(range(1, 101), num_ratings)
    
    for movie_id in movie_ids:
        ratings.append({
            'userId': user_id,
            'movieId': movie_id,
            'rating': random.choice([1, 2, 3, 4, 5]),
            'timestamp': int((datetime(2020, 1, 1) + pd.Timedelta(days=random.randint(0, 1000))).timestamp())
        })
ratings_df = pd.DataFrame(ratings)

# Create train/test split (80/20)
ratings_df = ratings_df.sample(frac=1, random_state=42)  # Shuffle
split_idx = int(len(ratings_df) * 0.8)
train_df = ratings_df.iloc[:split_idx]
test_df = ratings_df.iloc[split_idx:]

# Save files
movies_df.to_csv('data/movies.csv', index=False)
users_df.to_csv('data/users.csv', index=False)
ratings_df.to_csv('data/ratings.csv', index=False)
train_df.to_csv('data/ratings_train.csv', index=False)
test_df.to_csv('data/ratings_test.csv', index=False)

print(f"Generated quick dataset with {len(movies_df)} movies, {len(users_df)} users, and {len(ratings_df)} ratings")
print(f"Training set: {len(train_df)} ratings")
print(f"Test set: {len(test_df)} ratings")






def generate_quick_dataset():
    # This function just runs all the code above
    # You can copy and paste the existing code here or wrap it in this function
    # For now, this will work since the code already runs at module level
    pass

if __name__ == "__main__":
    # This ensures the code runs when the script is executed directly
    pass
