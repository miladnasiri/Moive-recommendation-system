import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time


from models.recommendation_models import *



from models.recommendation_models import load_data, compare_models
# And
from models.recommendation_models import (
    PopularityRecommender, 
    UserBasedCF, 
    ItemBasedCF, 
    MatrixFactorization, 
    NeuralCF, 
    HybridRecommender
)
# And
from demo.recommendation_demo import interactive_recommendations

def data_generation_workflow():
    """Generate synthetic data for the recommendation system"""
    # Import the data generator
    from synthetic_data_generator import generate_movie_data, generate_user_data, generate_ratings, analyze_data, create_train_test_split, save_data
    
    print("Generating synthetic movie data...")
    movies_df = generate_movie_data(n_movies=1000)
    
    print("Generating synthetic user data...")
    users_df = generate_user_data(n_users=5000)
    
    print("Generating synthetic ratings data...")
    ratings_df = generate_ratings(users_df, movies_df, sparsity=0.98)
    
    print("Analyzing data...")
    stats = analyze_data(users_df, movies_df, ratings_df)
    
    print("Creating train/test split...")
    train_df, test_df = create_train_test_split(ratings_df)
    
    print("Saving data...")
    save_data(users_df, movies_df, ratings_df, train_df, test_df)
    
    print("Data generation complete!")
    return users_df, movies_df, train_df, test_df

def interactive_recommendations():
    """Interactive demo where you can get recommendations for a user"""
    # Load data
    users_df, movies_df, train_df, test_df = load_data()
    
    # Train a hybrid model
    print("\nTraining recommendation models...")
    
    # Train component models
    user_cf = UserBasedCF(k=20, metric='cosine')
    item_cf = ItemBasedCF(k=20, metric='cosine')
    mf = MatrixFactorization(n_factors=50, n_epochs=5)
    
    print("Training User-CF model...")
    user_cf.fit(train_df)
    
    print("Training Item-CF model...")
    item_cf.fit(train_df)
    
    print("Training Matrix Factorization model...")
    mf.fit(train_df)
    
    # Create and train hybrid model
    hybrid = HybridRecommender(
        models=[user_cf, item_cf, mf],
        weights=[0.3, 0.3, 0.4]
    )
    
    # Use the hybrid model for recommendations
    model = hybrid
    
    # Start interactive loop
    while True:
        # Display available users
        print("\nAvailable users:")
        sample_users = random.sample(list(train_df['userId'].unique()), min(5, len(train_df['userId'].unique())))
        for i, user_id in enumerate(sample_users):
            print(f"{i+1}. User {user_id}")
        
        # Get user selection
        try:
            choice = input("\nSelect a user (1-5) or enter a user ID, or 'q' to quit: ")
            if choice.lower() == 'q':
                break
            
            if choice.isdigit() and 1 <= int(choice) <= len(sample_users):
                user_id = sample_users[int(choice) - 1]
            else:
                user_id = int(choice)
                if user_id not in train_df['userId'].unique():
                    print(f"User {user_id} not found in the dataset. Please try again.")
                    continue
        except ValueError:
            print("Invalid input. Please try again.")
            continue
        
        # Show user profile
        user_info = users_df[users_df['userId'] == user_id].iloc[0]
        print(f"\nUser {user_id} Profile:")
        print(f"Age Group: {user_info['ageGroup']}")
        print(f"Gender: {user_info['gender']}")
        print(f"Activity Level: {user_info['activityLevel']}/10")
        print(f"Join Date: {user_info['joinDate']}")
        
        # Show user's ratings
        user_ratings = train_df[train_df['userId'] == user_id]
        print(f"\nUser {user_id} has rated {len(user_ratings)} movies")
        
        print("\nTop-rated movies by this user:")
        top_rated = user_ratings.sort_values('rating', ascending=False).head(5)
        for _, row in top_rated.iterrows():
            movie_info = movies_df[movies_df['movieId'] == row['movieId']].iloc[0]
            print(f"  - {movie_info['title']} ({movie_info['releaseYear']}) - {movie_info['genres']}")
            print(f"    Rating: {row['rating']}/5")
        
        # Get recommendations
        print(f"\nGetting recommendations for User {user_id}...")
        start_time = time.time()
        recommendations = model.recommend(user_id, n=10)
        elapsed = time.time() - start_time
        
        print(f"\nTop 10 recommendations (generated in {elapsed:.2f} seconds):")
        for i, (movie_id, score) in enumerate(recommendations):
            movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            print(f"{i+1}. {movie_info['title']} ({movie_info['releaseYear']})")
            print(f"   Genres: {movie_info['genres']}")
            print(f"   Popularity: {movie_info['popularityScore']}/100")
            print(f"   Recommendation Score: {score:.2f}/5")
        
        # Ask if user wants to explore a specific recommendation
        explore = input("\nExplain recommendation for which movie? (1-10, or press Enter to skip): ")
        if explore.isdigit() and 1 <= int(explore) <= len(recommendations):
            idx = int(explore) - 1
            movie_id = recommendations[idx][0]
            movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            
            print(f"\nWhy we recommended '{movie_info['title']}':")
            
            # Get similar users who liked this movie
            if isinstance(model, HybridRecommender):
                user_model = model.models[0]  # User-CF model
            else:
                user_model = model
                
            if isinstance(user_model, UserBasedCF):
                user_idx = user_model.user_id_map[user_id]
                movie_idx = user_model.movie_id_map[movie_id]
                
                # Get similar users
                user_vector = user_model.user_item_matrix[user_idx].reshape(1, -1)
                distances, indices = user_model.model.kneighbors(user_vector)
                
                # Get similar users who rated this movie highly
                neighbor_indices = indices.flatten()
                neighbor_ratings = user_model.user_item_matrix[neighbor_indices, movie_idx].toarray().flatten()
                
                # Count positive ratings
                positive_count = sum(1 for r in neighbor_ratings if r >= 4)
                
                print(f"- {positive_count} similar users rated this movie 4 stars or higher")
            
            # Show movies with similar genres that the user liked
            movie_genres = set(movie_info['genres'].split('|'))
            
            # Find user-rated movies with similar genres
            user_liked_movies = user_ratings[user_ratings['rating'] >= 4]
            genre_matches = []
            
            for _, row in user_liked_movies.iterrows():
                rated_movie = movies_df[movies_df['movieId'] == row['movieId']].iloc[0]
                rated_genres = set(rated_movie['genres'].split('|'))
                common_genres = movie_genres.intersection(rated_genres)
                
                if common_genres:
                    genre_matches.append((rated_movie['title'], common_genres))
            
            if genre_matches:
                print("- Genre similarity to movies you liked:")
                for title, common in genre_matches[:3]:
                    print(f"  * '{title}' - Common genres: {', '.join(common)}")
            
            # Show popularity information
            print(f"- This movie has a popularity score of {movie_info['popularityScore']}/100")
            
            # Show release year information
            user_avg_year = user_ratings.merge(movies_df, on='movieId')['releaseYear'].mean()
            print(f"- Released in {movie_info['releaseYear']} (Your average watched year: {user_avg_year:.0f})")

if __name__ == "__main__":
    print("Movie Recommendation System Demo")
    print("================================")
    
    # Ask if user wants to generate new data or use existing
    choice = input("Generate new synthetic data? (y/n): ")
    if choice.lower() == 'y':
        data_generation_workflow()
    
    # Run interactive recommendation demo
    interactive_recommendations()
