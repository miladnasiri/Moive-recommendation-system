
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dot, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import random
import os

# And

# And


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class BaseRecommender:
    """Base class for all recommender models"""
    
    def __init__(self, name):
        self.name = name
        self.train_time = 0
        self.predict_time = 0
    
    def fit(self, train_data):
        """Train the model on training data"""
        start_time = time.time()
        self._fit(train_data)
        self.train_time = time.time() - start_time
        return self
    
    def predict(self, user_id, movie_ids):
        """Predict ratings for given user-movie pairs"""
        start_time = time.time()
        predictions = self._predict(user_id, movie_ids)
        self.predict_time = time.time() - start_time
        return predictions
    
    def recommend(self, user_id, n=10, exclude_rated=True):
        """Recommend top N movies for a user"""
        start_time = time.time()
        recommendations = self._recommend(user_id, n, exclude_rated)
        self.predict_time += time.time() - start_time
        return recommendations
    
    def _fit(self, train_data):
        """To be implemented by derived classes"""
        raise NotImplementedError
    
    def _predict(self, user_id, movie_ids):
        """To be implemented by derived classes"""
        raise NotImplementedError
    
    def _recommend(self, user_id, n, exclude_rated):
        """Default implementation - can be overridden by derived classes"""
        if not hasattr(self, 'all_movie_ids'):
            raise ValueError("Model doesn't have all_movie_ids attribute")
        
        # Get all movie ids to predict
        movie_ids_to_predict = self.all_movie_ids
        
        # Exclude already rated movies if requested
        if exclude_rated and hasattr(self, 'user_item_matrix'):
            rated_movie_indices = self.user_item_matrix[self.user_id_map[user_id]].nonzero()[1]
            rated_movie_ids = [self.movie_id_list[idx] for idx in rated_movie_indices]
            movie_ids_to_predict = [mid for mid in movie_ids_to_predict if mid not in rated_movie_ids]
        
        # Predict ratings for all movies
        predictions = self.predict(user_id, movie_ids_to_predict)
        
        # Sort by predicted rating
        movie_predictions = list(zip(movie_ids_to_predict, predictions))
        movie_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return movie_predictions[:n]

class PopularityRecommender(BaseRecommender):
    """Recommend based on movie popularity (most rated or highest average rating)"""
    
    def __init__(self, method='count'):
        super().__init__(f"Popularity ({method})")
        self.method = method  # 'count' or 'average'
        self.movie_scores = None
        self.all_movie_ids = None
    
    def _fit(self, train_data):
        # Calculate popularity scores
        if self.method == 'count':
            # Popularity by number of ratings
            self.movie_scores = train_data.groupby('movieId').size().reset_index(name='score')
        else:
            # Popularity by average rating
            self.movie_scores = train_data.groupby('movieId')['rating'].mean().reset_index(name='score')
        
        # Sort by score in descending order
        self.movie_scores = self.movie_scores.sort_values('score', ascending=False)
        
        # Store all movie IDs for recommendation
        self.all_movie_ids = self.movie_scores['movieId'].values
        
        # Create a mapping from movie ID to score for faster lookup
        self.movie_score_map = dict(zip(self.movie_scores['movieId'], self.movie_scores['score']))
    
    def _predict(self, user_id, movie_ids):
        # For any movie, return its popularity score
        predictions = [self.movie_score_map.get(movie_id, 0) for movie_id in movie_ids]
        
        # Normalize to 1-5 range if using count method
        if self.method == 'count':
            max_score = self.movie_scores['score'].max()
            min_score = self.movie_scores['score'].min()
            predictions = [1 + 4 * (p - min_score) / (max_score - min_score) if max_score > min_score else 3 for p in predictions]
        
        return predictions
    
    def _recommend(self, user_id, n=10, exclude_rated=True):
        # For popularity model, we just return the top N most popular movies
        # We ignore user_id since recommendations are the same for all users
        top_movies = self.movie_scores.head(n)
        return list(zip(top_movies['movieId'], top_movies['score']))

class UserBasedCF(BaseRecommender):
    """User-Based Collaborative Filtering using k-nearest neighbors"""
    
    def __init__(self, k=20, metric='cosine'):
        super().__init__(f"UserCF (k={k}, {metric})")
        self.k = k
        self.metric = metric
        self.user_item_matrix = None
        self.model = None
        self.user_id_map = None
        self.movie_id_map = None
        self.user_id_list = None
        self.movie_id_list = None
        self.all_movie_ids = None
    
    def _fit(self, train_data):
        # Create user-item matrix
        user_ids = train_data['userId'].unique()
        movie_ids = train_data['movieId'].unique()
        
        # Create mappings between original IDs and matrix indices
        self.user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
        self.movie_id_map = {movie_id: i for i, movie_id in enumerate(movie_ids)}
        
        # Create reverse mappings
        self.user_id_list = user_ids
        self.movie_id_list = movie_ids
        
        # Store all movie IDs for recommendation
        self.all_movie_ids = movie_ids
        
        # Create sparse matrix
        n_users = len(user_ids)
        n_movies = len(movie_ids)
        
        # Map user and movie IDs to matrix indices
        user_indices = [self.user_id_map[user_id] for user_id in train_data['userId']]
        movie_indices = [self.movie_id_map[movie_id] for movie_id in train_data['movieId']]
        
        # Create sparse matrix with ratings
        ratings = train_data['rating'].values
        self.user_item_matrix = csr_matrix((ratings, (user_indices, movie_indices)), shape=(n_users, n_movies))
        
        # Train KNN model for finding similar users
        self.model = NearestNeighbors(n_neighbors=self.k, metric=self.metric, algorithm='brute')
        self.model.fit(self.user_item_matrix)
    
    def _predict(self, user_id, movie_ids):
        # Check if user exists in training data
        if user_id not in self.user_id_map:
            # Return average rating for each movie
            return [3.0] * len(movie_ids)
        
        # Get user index
        user_idx = self.user_id_map[user_id]
        
        # Convert movie IDs to matrix indices
        movie_indices = [self.movie_id_map.get(movie_id, -1) for movie_id in movie_ids]
        
        # Find K nearest neighbors
        user_vector = self.user_item_matrix[user_idx].reshape(1, -1)
        distances, indices = self.model.kneighbors(user_vector)
        
        # Get similarity scores (convert distances to similarities)
        similarities = 1 - distances.flatten()
        
        # Get neighbor indices
        neighbor_indices = indices.flatten()
        
        # Predict ratings for each movie
        predictions = []
        for movie_idx in movie_indices:
            if movie_idx == -1:
                # Movie not in training data
                predictions.append(3.0)
                continue
            
            # Get ratings from neighbors for this movie
            neighbor_ratings = self.user_item_matrix[neighbor_indices, movie_idx].toarray().flatten()
            
            # Filter out users who haven't rated the movie
            mask = neighbor_ratings > 0
            if not np.any(mask):
                # No neighbor rated this movie
                predictions.append(3.0)
                continue
            
            # Get ratings and similarities for users who rated the movie
            valid_ratings = neighbor_ratings[mask]
            valid_similarities = similarities[mask]
            
            # Weighted average
            prediction = np.sum(valid_ratings * valid_similarities) / np.sum(valid_similarities)
            predictions.append(prediction)
        
        return predictions

class ItemBasedCF(BaseRecommender):
    """Item-Based Collaborative Filtering using k-nearest neighbors"""
    
    def __init__(self, k=20, metric='cosine'):
        super().__init__(f"ItemCF (k={k}, {metric})")
        self.k = k
        self.metric = metric
        self.user_item_matrix = None
        self.item_item_matrix = None
        self.model = None
        self.user_id_map = None
        self.movie_id_map = None
        self.user_id_list = None
        self.movie_id_list = None
        self.all_movie_ids = None
    
    def _fit(self, train_data):
        # Create user-item matrix (same as in UserBasedCF)
        user_ids = train_data['userId'].unique()
        movie_ids = train_data['movieId'].unique()
        
        self.user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
        self.movie_id_map = {movie_id: i for i, movie_id in enumerate(movie_ids)}
        
        self.user_id_list = user_ids
        self.movie_id_list = movie_ids
        
        self.all_movie_ids = movie_ids
        
        n_users = len(user_ids)
        n_movies = len(movie_ids)
        
        user_indices = [self.user_id_map[user_id] for user_id in train_data['userId']]
        movie_indices = [self.movie_id_map[movie_id] for movie_id in train_data['movieId']]
        
        ratings = train_data['rating'].values
        self.user_item_matrix = csr_matrix((ratings, (user_indices, movie_indices)), shape=(n_users, n_movies))
        
        # Train KNN model for finding similar items
        # Note: We transpose the matrix to get item features
        self.model = NearestNeighbors(n_neighbors=self.k, metric=self.metric, algorithm='brute')
        self.model.fit(self.user_item_matrix.T.tocsr())
    
    def _predict(self, user_id, movie_ids):
        # Check if user exists in training data
        if user_id not in self.user_id_map:
            # Return average rating for each movie
            return [3.0] * len(movie_ids)
        
        # Get user index
        user_idx = self.user_id_map[user_id]
        
        # Get user's ratings
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Get indices of movies rated by the user
        rated_movie_indices = user_ratings.nonzero()[0]
        
        # If user hasn't rated any movies, return average
        if len(rated_movie_indices) == 0:
            return [3.0] * len(movie_ids)
        
        # Predict ratings for each movie
        predictions = []
        for movie_id in movie_ids:
            if movie_id not in self.movie_id_map:
                # Movie not in training data
                predictions.append(3.0)
                continue
            
            movie_idx = self.movie_id_map[movie_id]
            
            # Skip if user already rated this movie
            if user_ratings[movie_idx] > 0:
                predictions.append(user_ratings[movie_idx])
                continue
            
            # Get similar items
            movie_vector = self.user_item_matrix.T[movie_idx].reshape(1, -1)
            distances, indices = self.model.kneighbors(movie_vector)
            
            # Convert distances to similarities
            similarities = 1 - distances.flatten()
            
            # Get ratings for similar items
            similar_item_indices = indices.flatten()
            similar_item_ratings = user_ratings[similar_item_indices]
            
            # Filter out items that the user hasn't rated
            mask = similar_item_ratings > 0
            if not np.any(mask):
                # User hasn't rated any similar items
                predictions.append(3.0)
                continue
            
            # Get ratings and similarities for items the user has rated
            valid_ratings = similar_item_ratings[mask]
            valid_similarities = similarities[mask]
            
            # Weighted average
            prediction = np.sum(valid_ratings * valid_similarities) / np.sum(valid_similarities)
            predictions.append(prediction)
        
        return predictions

class MatrixFactorization(BaseRecommender):
    """Matrix Factorization using SVD"""
    
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=0.1):
        super().__init__(f"SVD (factors={n_factors})")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.user_id_map = None
        self.movie_id_map = None
        self.user_id_list = None
        self.movie_id_list = None
        self.all_movie_ids = None
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
    
    def _fit(self, train_data):
        # Create user and item mappings
        user_ids = train_data['userId'].unique()
        movie_ids = train_data['movieId'].unique()
        
        self.user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
        self.movie_id_map = {movie_id: i for i, movie_id in enumerate(movie_ids)}
        
        self.user_id_list = user_ids
        self.movie_id_list = movie_ids
        
        self.all_movie_ids = movie_ids
        
        n_users = len(user_ids)
        n_movies = len(movie_ids)
        
        # Create user-item matrix
        user_indices = [self.user_id_map[user_id] for user_id in train_data['userId']]
        movie_indices = [self.movie_id_map[movie_id] for movie_id in train_data['movieId']]
        
        ratings = train_data['rating'].values
        user_item_matrix = csr_matrix((ratings, (user_indices, movie_indices)), shape=(n_users, n_movies))
        
        # Calculate global mean
        self.global_mean = np.mean(ratings)
        
        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_movies, self.n_factors))
        
        # Initialize biases to zeros
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_movies)
        
        # Stochastic gradient descent
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.arange(len(train_data))
            np.random.shuffle(indices)
            
            # Iterate over all ratings
            for idx in indices:
                u = self.user_id_map[train_data.iloc[idx]['userId']]
                i = self.movie_id_map[train_data.iloc[idx]['movieId']]
                r = train_data.iloc[idx]['rating']
                
                # Predict current rating
                pred = self.global_mean + self.user_biases[u] + self.item_biases[i] + np.dot(self.user_factors[u], self.item_factors[i])
                
                # Calculate error
                err = r - pred
                
                # Update biases
                self.user_biases[u] += self.lr * (err - self.reg * self.user_biases[u])
                self.item_biases[i] += self.lr * (err - self.reg * self.item_biases[i])
                
                # Update factors
                temp_user_factors = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (err * temp_user_factors - self.reg * self.item_factors[i])
            
            # Optionally print rmse for this epoch
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs} completed")
    
    def _predict(self, user_id, movie_ids):
        # Check if user exists in training data
        if user_id not in self.user_id_map:
            # Return global mean for each movie
            return [self.global_mean] * len(movie_ids)
        
        # Get user index
        user_idx = self.user_id_map[user_id]
        
        # Predict ratings for each movie
        predictions = []
        for movie_id in movie_ids:
            if movie_id not in self.movie_id_map:
                # Movie not in training data
                predictions.append(self.global_mean)
                continue
            
            movie_idx = self.movie_id_map[movie_id]
            
            # Calculate prediction
            pred = (self.global_mean + 
                   self.user_biases[user_idx] + 
                   self.item_biases[movie_idx] + 
                   np.dot(self.user_factors[user_idx], self.item_factors[movie_idx]))
            
            # Clip prediction to rating range
            pred = max(1.0, min(5.0, pred))
            
            predictions.append(pred)
        
        return predictions

class NeuralCF(BaseRecommender):
    """Neural Collaborative Filtering using a neural network"""
    
    def __init__(self, n_factors=16, layers=[64, 32, 16], dropout=0.2, n_epochs=20, batch_size=256):
        super().__init__(f"NeuralCF (factors={n_factors}, layers={layers})")
        self.n_factors = n_factors
        self.layers = layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = None
        self.user_id_map = None
        self.movie_id_map = None
        self.user_id_list = None
        self.movie_id_list = None
        self.all_movie_ids = None
        self.n_users = None
        self.n_movies = None
    
    def _create_model(self):
        # Input layers
        user_input = Input(shape=(1,), name='user_input')
        item_input = Input(shape=(1,), name='item_input')
        
        # Embedding layers
        user_embedding = Embedding(input_dim=self.n_users, 
                                   output_dim=self.n_factors, 
                                   embeddings_regularizer=l2(0.01),
                                   name='user_embedding')(user_input)
        item_embedding = Embedding(input_dim=self.n_movies,
                                   output_dim=self.n_factors,
                                   embeddings_regularizer=l2(0.01),
                                   name='item_embedding')(item_input)
        
        # Flatten embeddings
        user_flatten = Flatten()(user_embedding)
        item_flatten = Flatten()(item_embedding)
        
        # Concatenate embeddings
        concat = Concatenate()([user_flatten, item_flatten])
        
        # Multi-layer perceptron
        x = concat
        for i, layer_size in enumerate(self.layers):
            x = Dense(layer_size, activation='relu', name=f'layer_{i}')(x)
            x = Dropout(self.dropout)(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = Model(inputs=[user_input, item_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.002),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _fit(self, train_data):
        # Create user and item mappings
        user_ids = train_data['userId'].unique()
        movie_ids = train_data['movieId'].unique()
        
        self.user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
        self.movie_id_map = {movie_id: i for i, movie_id in enumerate(movie_ids)}
        
        self.user_id_list = user_ids
        self.movie_id_list = movie_ids
        
        self.all_movie_ids = movie_ids
        
        self.n_users = len(user_ids)
        self.n_movies = len(movie_ids)
        
        # Create model
        self.model = self._create_model()
        
        # Prepare data for training
        user_indices = np.array([self.user_id_map[user_id] for user_id in train_data['userId']])
        movie_indices = np.array([self.movie_id_map[movie_id] for movie_id in train_data['movieId']])
        ratings = np.array(train_data['rating']).astype('float32')
        
        # Normalize ratings to [0, 1]
        ratings = (ratings - 1) / 4
        
        # Train model
        history = self.model.fit(
            [user_indices, movie_indices],
            ratings,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            validation_split=0.1,
            verbose=1
        )
        
        return history
    
    def _predict(self, user_id, movie_ids):
        # Check if user exists in training data
        if user_id not in self.user_id_map:
            # Return middle rating for each movie
            return [3.0] * len(movie_ids)
        
        # Get user index
        user_idx = self.user_id_map[user_id]
        
        # Prepare input for prediction
        user_indices = np.array([user_idx] * len(movie_ids))
        movie_indices = np.array([self.movie_id_map.get(movie_id, 0) for movie_id in movie_ids])
        
        # Filter out movies not in training data
        valid_mask = np.array([movie_id in self.movie_id_map for movie_id in movie_ids])
        
        # If all movies are invalid, return default ratings
        if not np.any(valid_mask):
            return [3.0] * len(movie_ids)
        
        # Predict only for valid movies
        valid_user_indices = user_indices[valid_mask]
        valid_movie_indices = movie_indices[valid_mask]
        
        # Make predictions
        predictions = self.model.predict([valid_user_indices, valid_movie_indices], verbose=0).flatten()
        
        # Convert back to original rating scale
        predictions = predictions * 4 + 1
        
        # Prepare final predictions, including default values for invalid movies
        final_predictions = []
        valid_idx = 0
        for is_valid in valid_mask:
            if is_valid:
                final_predictions.append(predictions[valid_idx])
                valid_idx += 1
            else:
                final_predictions.append(3.0)
        
        return final_predictions

class HybridRecommender(BaseRecommender):
    """Hybrid recommender that combines multiple models"""
    
    def __init__(self, models, weights=None):
        model_names = [model.name for model in models]
        super().__init__(f"Hybrid ({' + '.join(model_names)})")
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
        self.all_movie_ids = None
    
    def _fit(self, train_data):
        # Fit all individual models
        for model in self.models:
            print(f"Training {model.name}...")
            model.fit(train_data)
        
        # Set all_movie_ids from the first model
        if hasattr(self.models[0], 'all_movie_ids'):
            self.all_movie_ids = self.models[0].all_movie_ids
    
    def _predict(self, user_id, movie_ids):
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            all_predictions.append(model.predict(user_id, movie_ids))
        
        # Combine predictions using weights
        combined_predictions = np.zeros(len(movie_ids))
        for i, predictions in enumerate(all_predictions):
            combined_predictions += np.array(predictions) * self.weights[i]
        
        return combined_predictions.tolist()

def evaluate_model(model, test_data, top_n=10, verbose=True):
    """Evaluate a model on test data"""
    
    start_time = time.time()
    
    # Get all unique users in test data
    test_users = test_data['userId'].unique()
    
    # Metrics
    precision_sum = 0
    recall_sum = 0
    ndcg_sum = 0
    hit_ratio_sum = 0
    
    # For each user
    for user_id in tqdm(test_users, desc=f"Evaluating {model.name}", disable=not verbose):
        # Get items the user has rated in test set
        user_test_data = test_data[test_data['userId'] == user_id]
        test_items = set(user_test_data['movieId'])
        
        # Get ratings for these items
        test_ratings = dict(zip(user_test_data['movieId'], user_test_data['rating']))
        
        # Get positive items (rating >= 4)
        positive_items = set(item for item, rating in test_ratings.items() if rating >= 4)
        
        # Skip users with no positive items
        if not positive_items:
            continue
        
        # Get top-N recommendations
        recommendations = model.recommend(user_id, n=top_n)
        recommended_items = [item for item, _ in recommendations]
        
        # Calculate metrics
        # Precision@N
        precision = len(set(recommended_items) & positive_items) / top_n
        precision_sum += precision
        
        # Recall@N
        recall = len(set(recommended_items) & positive_items) / max(1, len(positive_items))
        recall_sum += recall
        
        # Hit Ratio@N
        hit = 1 if len(set(recommended_items) & positive_items) > 0 else 0
        hit_ratio_sum += hit
        
        # NDCG@N
        dcg = 0
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(positive_items), top_n)))
        for i, item in enumerate(recommended_items):
            if item in positive_items:
                dcg += 1 / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_sum += ndcg
    
    # Calculate average metrics
    n_users = len(test_users)
    precision = precision_sum / n_users
    recall = recall_sum / n_users
    hit_ratio = hit_ratio_sum / n_users
    ndcg = ndcg_sum / n_users
    
    # Calculate evaluation time
    eval_time = time.time() - start_time
    
    results = {
        'precision': precision,
        'recall': recall,
        'hit_ratio': hit_ratio,
        'ndcg': ndcg,
        'eval_time': eval_time
    }
    
    if verbose:
        print(f"Results for {model.name}:")
        print(f"  Precision@{top_n}: {precision:.4f}")
        print(f"  Recall@{top_n}: {recall:.4f}")
        print(f"  Hit Ratio@{top_n}: {hit_ratio:.4f}")
        print(f"  NDCG@{top_n}: {ndcg:.4f}")
        print(f"  Training time: {model.train_time:.2f}s")
        print(f"  Evaluation time: {eval_time:.2f}s")
    
    return results

def compare_models(models, train_data, test_data, top_n=10):
    """Train and evaluate multiple models and compare their performance"""
    
    results = {}
    
    for model in models:
        print(f"\nTraining and evaluating {model.name}...")
        model.fit(train_data)
        result = evaluate_model(model, test_data, top_n=top_n)
        results[model.name] = result
    
    # Create comparison table
    comparison = pd.DataFrame(results).T
    comparison['train_time'] = [model.train_time for model in models]
    
    # Reorder columns
    columns = ['precision', 'recall', 'hit_ratio', 'ndcg', 'train_time', 'eval_time']
    comparison = comparison[columns]
    
    return comparison

def load_data():
    """Load data from CSV files"""
    
    # Load user data
    users_df = pd.read_csv('users.csv')
    
    # Load movie data
    movies_df = pd.read_csv('movies.csv')
    
    # Load ratings data
    train_df = pd.read_csv('ratings_train.csv')
    test_df = pd.read_csv('ratings_test.csv')
    
    print(f"Loaded {len(users_df)} users, {len(movies_df)} movies")
    print(f"Training set: {len(train_df)} ratings")
    print(f"Test set: {len(test_df)} ratings")
    
    return users_df, movies_df, train_df, test_df

def main():
    """Main function to run the recommendation system"""
    
    # Load data
    print("Loading data...")
    users_df, movies_df, train_df, test_df = load_data()
    
    # Create models to compare
    models = [
        PopularityRecommender(method='count'),
        PopularityRecommender(method='average'),
        UserBasedCF(k=20, metric='cosine'),
        ItemBasedCF(k=20, metric='cosine'),
        MatrixFactorization(n_factors=50, n_epochs=10),
        # NeuralCF requires more time to train, uncomment if needed
        # NeuralCF(n_factors=16, layers=[64, 32], n_epochs=5)
    ]
    
    # Create a hybrid model
    hybrid_model = HybridRecommender(
        models=[models[2], models[3], models[4]],  # User-CF, Item-CF, and MF
        weights=[0.3, 0.3, 0.4]  # Giving slightly more weight to MF
    )
    models.append(hybrid_model)
    
    # Compare models
    print("\nComparing models...")
    comparison = compare_models(models, train_df, test_df, top_n=10)
    
    # Print comparison table
    print("\nModel Comparison:")
    print(comparison)
    
    # Save comparison to CSV
    comparison.to_csv('model_comparison.csv')
    print("Comparison saved to model_comparison.csv")
    
    # Select best model for demonstration
    best_model = models[comparison['ndcg'].idxmax()]
    print(f"\nBest model: {best_model.name}")
    
    # Demonstrate recommendations for a few users
    print("\nSample Recommendations:")
    sample_users = train_df['userId'].unique()[:5]  # First 5 users
    
    for user_id in sample_users:
        print(f"\nRecommendations for User {user_id}:")
        
        # Get user's top-rated movies
        user_ratings = train_df[train_df['userId'] == user_id]
        top_rated = user_ratings.sort_values('rating', ascending=False).head(3)
        
        print("User's top-rated movies:")
        for _, row in top_rated.iterrows():
            movie_info = movies_df[movies_df['movieId'] == row['movieId']].iloc[0]
            print(f"  - {movie_info['title']} ({movie_info['releaseYear']}): {row['rating']}/5")
        
        # Get recommendations
        recommendations = best_model.recommend(user_id, n=5)
        
        print("Recommended movies:")
        for movie_id, score in recommendations:
            movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            print(f"  - {movie_info['title']} ({movie_info['releaseYear']}): Score = {score:.2f}")
    
    print("\nDone!")

def visualize_results(comparison):
    """Visualize model comparison results"""
    
    # Metrics to plot
    metrics = ['precision', 'recall', 'ndcg', 'hit_ratio']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        # Sort by metric value
        sorted_df = comparison.sort_values(metric, ascending=False)
        
        # Create bar chart
        ax = axes[i]
        bars = ax.bar(sorted_df.index, sorted_df[metric])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                   f'{height:.4f}', ha='center', va='bottom')
        
        # Add labels and title
        ax.set_title(f'{metric.upper()}@10')
        ax.set_ylim(0, sorted_df[metric].max() * 1.15)
        ax.tick_params(axis='x', rotation=45)
    
    # Add training time plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sorted_df = comparison.sort_values('train_time')
    bars = ax2.bar(sorted_df.index, sorted_df['train_time'])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{height:.2f}s', ha='center', va='bottom')
    
    ax2.set_title('Training Time (seconds)')
    ax2.set_ylabel('Time (s)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout and save
    fig.tight_layout()
    fig2.tight_layout()
    
    fig.savefig('model_metrics_comparison.png')
    fig2.savefig('model_training_time.png')
    
    print("Visualizations saved to 'model_metrics_comparison.png' and 'model_training_time.png'")

# Example function to demonstrate more advanced analysis
def analyze_user_segments(users_df, movies_df, train_df, test_df, best_model):
    """Analyze recommendation performance for different user segments"""
    
    print("Analyzing recommendation performance by user segment...")
    
    # Define user segments (by activity level)
    users_df['activity_segment'] = pd.qcut(users_df['activityLevel'], 3, 
                                          labels=['Low', 'Medium', 'High'])
    
    # Create dict to store results
    segment_results = {}
    
    # Evaluate for each segment
    for segment in ['Low', 'Medium', 'High']:
        # Get users in this segment
        segment_users = users_df[users_df['activity_segment'] == segment]['userId'].values
        
        # Filter test data to only include these users
        segment_test = test_df[test_df['userId'].isin(segment_users)]
        
        # Skip if no data
        if len(segment_test) == 0:
            continue
        
        # Evaluate
        print(f"Evaluating for {segment} activity users...")
        results = evaluate_model(best_model, segment_test, verbose=False)
        segment_results[segment] = results
    
    # Create comparison dataframe
    segment_comparison = pd.DataFrame(segment_results).T
    
    print("\nRecommendation performance by user activity level:")
    print(segment_comparison[['precision', 'recall', 'ndcg', 'hit_ratio']])
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    segment_comparison[['precision', 'recall', 'ndcg', 'hit_ratio']].plot(kind='bar', ax=ax)
    ax.set_title('Recommendation Performance by User Activity Level')
    ax.set_ylabel('Metric Value')
    ax.legend(title='Metric')
    
    fig.tight_layout()
    fig.savefig('segment_performance.png')
    
    print("Segment analysis visualization saved to 'segment_performance.png'")

# Run the program if executed directly
if __name__ == "__main__":
    main()
