import os
import sys
import argparse
import pandas as pd
import numpy as np

def train_and_evaluate():
    """Train models and evaluate performance"""
    # Load data directly
    print("Loading data...")
    users_df = pd.read_csv('data/users.csv')
    movies_df = pd.read_csv('data/movies.csv')
    train_df = pd.read_csv('data/ratings_train.csv')
    test_df = pd.read_csv('data/ratings_test.csv')
    
    print(f"Loaded {len(users_df)} users, {len(movies_df)} movies")
    print(f"Training set: {len(train_df)} ratings")
    print(f"Test set: {len(test_df)} ratings")
    
    # Import recommendation models - only do this after data is loaded
    sys.path.insert(0, os.path.abspath('models'))
    from recommendation_models import (
        PopularityRecommender, 
        UserBasedCF,
        ItemBasedCF, 
        MatrixFactorization, 
        HybridRecommender,
        evaluate_model,
        compare_models
    )
    
    # Create models
    print("Creating models...")
    models = [
        PopularityRecommender(method='count'),
        PopularityRecommender(method='average'),
        UserBasedCF(k=20, metric='cosine'),
        ItemBasedCF(k=20, metric='cosine'),
        MatrixFactorization(n_factors=20, n_epochs=5)  # Reduced for faster processing
    ]
    
    # Create hybrid model
    print("Creating hybrid model...")
    hybrid = HybridRecommender(
        models=[models[2], models[3], models[4]],
        weights=[0.3, 0.3, 0.4]
    )
    models.append(hybrid)
    
    # Train and evaluate
    print("Training and evaluating models...")
    results = compare_models(models, train_df, test_df)
    
    print("\nModel Comparison Results:")
    print(results)
    
    # Save results
    results.to_csv('data/model_comparison_results.csv')
    print("Results saved to data/model_comparison_results.csv")
    
    return results

def run_demo():
    """Run interactive recommendation demo"""
    print("Starting recommendation demo...")
    sys.path.insert(0, os.path.abspath('demo'))
    from recommendation_demo import interactive_recommendations
    interactive_recommendations()

def main():
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--train-models', action='store_true', help='Train recommendation models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    args = parser.parse_args()
    
    if args.train_models or args.evaluate:
        train_and_evaluate()
        
    if args.demo:
        run_demo()
        
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()
