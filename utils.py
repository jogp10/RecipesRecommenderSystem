import networkx as nx
from networkx.algorithms import community
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNBasic, NormalPredictor, SVD
from surprise.model_selection import train_test_split
from sklearn.model_selection import train_test_split as tts
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import defaultdict


def community_detection(g):
    # Louvain method
    louvain_communities = community.greedy_modularity_communities(g)
    
    # Girvan-Newman algorithm
    girvan_newman_communities = tuple(community.girvan_newman(g))
    
    # Label Propagation
    label_propagation_communities = list(community.label_propagation_communities(g))
    
    return {
        "Louvain": louvain_communities,
        "Girvan-Newman": girvan_newman_communities,
        "Label Propagation": label_propagation_communities
    }

def compute_centralities(g, top):
    degree_centrality = nx.degree_centrality(g)
    
    closseness_centrality = nx.closeness_centrality(g)

    betweenness_centrality = nx.betweenness_centrality(g, weight='weight')

    eigenvector_centrality = nx.eigenvector_centrality(g, weight='weight')

    pagerank = nx.pagerank(g, weight='weight')
    
    if top == None:
        top = len(degree_centrality)
    
    degree_centrality_top10 = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top]
    betweenness_centrality_top10 = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:top]
    eigenvector_centrality_top10 = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:top]
    pagerank_top10 = sorted(pagerank, key=lambda x: x[1], reverse=True)[:top]
    closseness_centrality_top_10 = sorted(closseness_centrality.items(), key=lambda x: x[1], reverse=True)[:top]

    return degree_centrality_top10, betweenness_centrality_top10, eigenvector_centrality_top10, pagerank_top10, closseness_centrality_top_10


def plot_centrality_rating(centrality_list, df_member):
    centrality_df = pd.DataFrame(centrality_list, columns=['member_id', 'centrality'])
    centrality_df['member_id'] = centrality_df['member_id'].astype('int64')


    # Merge centrality_df with member_df to combine centrality with average rating
    merged_df = pd.merge(df_member, centrality_df, on='member_id')
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df['centrality'], merged_df['member_avg_rating'], alpha=0.5)
    plt.title('Centrality vs. Average Rating')
    plt.xlabel('Centrality')
    plt.ylabel('Average Rating')
    plt.grid(True)
    plt.show()

def plot_centrality_over_time_joined(centrality_list, df_member):
    # Convert centrality_list to a DataFrame
    centrality_df = pd.DataFrame(centrality_list, columns=['member_id', 'centrality'])
    centrality_df['member_id'] = centrality_df['member_id'].astype('int64')

    # Merge centrality_df with df_member to combine centrality with member joined year
    merged_df = pd.merge(df_member, centrality_df, on='member_id')

    # Convert member_joined to datetime if it's not already
    merged_df['member_joined'] = pd.to_datetime(merged_df['member_joined'])

    # Extract the year from the member_joined column
    merged_df['joined_year'] = merged_df['member_joined'].dt.year

    # Group by joined_year and calculate the average centrality for each group
    centrality_by_year = merged_df.groupby('joined_year')['centrality'].mean()

    # Plotting
    plt.figure(figsize=(18, 6))
    plt.plot(centrality_by_year.index, centrality_by_year.values, marker='o')
    plt.title('Average Centrality Over Time Joined')
    plt.xlabel('Year Joined')
    plt.ylabel('Average Centrality')
    plt.grid(True)
    plt.xticks(centrality_by_year.index)  # Set x-ticks to years
    plt.show() 

def collaborative_filtering(df_reviews, communities, test_fraction = 0.20, user_based = True, model_type = 'KNN'):
    rmse_scores = []
    mae_scores = []
    precision_scores = []
    recall_scores = []
    sizes = []
    
    for i, community in enumerate(communities):
        community = [int(c) for c in community]
        community_reviews = df_reviews[df_reviews['member_id'].isin(community)]
    
        if len(community_reviews) > 0:
        
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(community_reviews[['member_id', 'recipe_id', 'rating']], reader)
            trainset, testset = train_test_split(data, test_size = test_fraction)

            if model_type == 'KNN':
                model = KNNBasic(sim_options={'user_based': user_based}, verbose= False)
            elif model_type == 'SVD':
                model = SVD(verbose = False)
            elif model_type == "Random":
                model = NormalPredictor()


            model.fit(trainset)

            predictions = model.test(testset, verbose= False)
            
            precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)

            precision = sum(prec for prec in precisions.values()) / len(precisions)
            recall = sum(rec for rec in recalls.values()) / len(recalls)

            rmse = accuracy.rmse(predictions, verbose= False)
            mae = accuracy.mae(predictions, verbose =False)
        
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            precision_scores.append(precision)
            recall_scores.append(recall)
            sizes.append(len(community))
            
            print(f"\033[1mCommunity {i + 1}\033[0m -> Size: {len(community)}")
            print(f"\033[1mRMSE ->\033[0m", rmse)
            print(f"\033[1mMAE ->\033[0m", mae)
            print(f"\033[1mPrecision@10 ->\033[0m", precision)
            print(f"\033[1mRecall@10 ->\033[0m", recall)
            print()
            
        else:
            print(f"Community {i + 1} has insufficient data for recommendations.")

    
    avg_rmse = sum(rmse_scores) / len(rmse_scores)
    avg_mae = sum(mae_scores) / len(mae_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    
    return avg_rmse, avg_mae, avg_precision, avg_recall

def collaborative_filtering_with_values(df_reviews, communities, test_fraction = 0.20, user_based = True, model_type = 'KNN'):
    rmse_scores = []
    mae_scores = []
    precision_scores = []
    recall_scores = []
    sizes = []
    
    for i, community in enumerate(communities):
        community = [int(c) for c in community]
        community_reviews = df_reviews[df_reviews['member_id'].isin(community)]
    
        if len(community_reviews) > 0:
        
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(community_reviews[['member_id', 'recipe_id', 'rating']], reader)
            trainset, testset = train_test_split(data, test_size = test_fraction)

            if model_type == 'KNN':
                model = KNNBasic(sim_options={'user_based': user_based}, verbose= False)
            elif model_type == 'SVD':
                model = SVD(verbose = False)
            elif model_type == "Random":
                model = NormalPredictor()


            model.fit(trainset)

            predictions = model.test(testset, verbose= False)
            
            precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)

            precision = sum(prec for prec in precisions.values()) / len(precisions)
            recall = sum(rec for rec in recalls.values()) / len(recalls)

            rmse = accuracy.rmse(predictions, verbose= False)
            mae = accuracy.mae(predictions, verbose =False)
        
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            precision_scores.append(precision)
            recall_scores.append(recall)
            sizes.append(len(community))
            
            print(f"\033[1mCommunity {i + 1}\033[0m -> Size: {len(community)}")
            print(f"\033[1mRMSE ->\033[0m", rmse)
            print(f"\033[1mMAE ->\033[0m", mae)
            print(f"\033[1mPrecision@10 ->\033[0m", precision)
            print(f"\033[1mRecall@10 ->\033[0m", recall)
            print()
            
        else:
            print(f"Community {i + 1} has insufficient data for recommendations.")

    
    avg_rmse = sum(rmse_scores) / len(rmse_scores)
    avg_mae = sum(mae_scores) / len(mae_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    
    return avg_rmse, avg_mae, avg_precision, avg_recall, rmse_scores, mae_scores, sizes

def find_similars(df_reviews, df_recipes):
    all_recommendations = {}
    
    # Merge reviews with recipe characteristics
    combined_data = pd.merge(df_reviews, df_recipes, on='recipe_id', how='left')
    
    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_data['ingredient_food_kg_names'])
    
    # Calculate cosine similarity
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Get recipe titles and ids
    recipe_ids = combined_data['recipe_id'].tolist()
    recipe_titles = combined_data['title'].tolist()
    
    # Iterate through each recipe
    for recipe_id in combined_data['recipe_id'].unique():
        recipe_index = combined_data[combined_data['recipe_id'] == recipe_id].index[0]
        similar_indices = cosine_similarities[recipe_index].argsort()[:-6:-1]  # Top 5 similar recipes
        
        # Keep track of unique similar recipes
        unique_similar_recipe_ids = set()
        
        # Iterate through similar recipes and their indices
        for idx in similar_indices:
            similar_recipe_id = recipe_ids[idx]
            
            # Skip if similar recipe is the original recipe or already encountered
            if similar_recipe_id == recipe_id or similar_recipe_id in unique_similar_recipe_ids:
                continue
            
            # Store unique similar recipe ID
            unique_similar_recipe_ids.add(similar_recipe_id)
            
            # Store similar recipe data
            similar_recipe_title = recipe_titles[idx]
            similar_score = cosine_similarities[recipe_index][idx]
            
            # Add similar recipe to recommendations
            if recipe_id not in all_recommendations:
                all_recommendations[recipe_id] = {
                    'original_title': recipe_titles[recipe_index],
                    'similar_recipes': []
                }
            
            all_recommendations[recipe_id]['similar_recipes'].append({
                'title': similar_recipe_title,
                'id': similar_recipe_id,
                'score': similar_score
            })
    
    return all_recommendations


def create_similar_recipes_dataframe(all_recommendations):
    similar_recipes_data = []

    for recipe_id, recipe_data in all_recommendations.items():
        original_title = recipe_data['original_title']
        similar_recipes_count = len(recipe_data['similar_recipes'])

        similar_scores = [similar_recipe['score'] for similar_recipe in recipe_data['similar_recipes']]

        for similar_recipe in recipe_data['similar_recipes']:
            similar_title = similar_recipe['title']
            similar_id = similar_recipe['id']
            similar_score = similar_recipe['score']

            similar_recipes_data.append({
                'recipe_id': recipe_id,
                'recipe_title': original_title,
                'similar_recipe_id': similar_id,
                'similar_recipe_title': similar_title,
                'score': similar_score
            })

    df_similar_recipes = pd.DataFrame(similar_recipes_data)

    # Ensure uniqueness of the pair recipe_id and similar_recipe_id
    df_similar_recipes.drop_duplicates(subset=['recipe_id', 'similar_recipe_id'], inplace=True)

    return df_similar_recipes


def calculate_average_similarity(df_similar_recipes):
    total_similar_recipes = df_similar_recipes['recipe_id'].nunique()
    total_similar_score = df_similar_recipes['score'].sum()
    total_pairs = len(df_similar_recipes)
    
    avg_similar_recipes_per_recipe = total_pairs / total_similar_recipes
    avg_similar_score = total_similar_score / total_pairs
    
    return avg_similar_recipes_per_recipe, avg_similar_score

def content_based_filtering(df_reviews, df_similar_recipes, filtered_communities):

    total_precision = 0
    total_recall = 0
    total_communities = 0
    
    for i, community in enumerate(filtered_communities):
        community = [int(c) for c in community]
        community_reviews = df_reviews[df_reviews['member_id'].isin(community)]

        # Merge reviews dataframe with recipe similarity data
        community_reviews = pd.merge(community_reviews, df_similar_recipes, on='recipe_id', how='left')

        community_reviews = community_reviews[community_reviews['member_id'].isin(community)]

        # Initialize evaluation metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Iterate through similar recipes and their ratings
        for user in community:
            user_reviews = community_reviews[community_reviews['member_id'] == user]
            for index, row in user_reviews.iterrows():
                # Check if the similar recipe has been reviewed
                if row['similar_recipe_id'] in user_reviews['recipe_id'].values:
                    # Compare ratings between original and similar recipe
                    if row['rating'] == user_reviews[user_reviews['recipe_id'] == row['similar_recipe_id']]['rating'].values[0]:
                        true_positives += 1
                    else:
                        false_positives += 1
                else:
                    false_negatives += 1

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

        # Update total evaluation metrics
        total_precision += precision
        total_recall += recall
        total_communities += 1

    # Calculate average precision, recall
    avg_precision = total_precision / total_communities if total_communities > 0 else 0
    avg_recall = total_recall / total_communities if total_communities > 0 else 0

    return avg_precision, avg_recall


def overall_content_based_filtering(df_reviews, df_similar_recipes, filtered_communities):

    total_precision = 0
    total_recall = 0
    total_users = 0
    
    # Concatenate all communities into one list of member ids
    all_members = [int(c) for community in filtered_communities for c in community]
    all_member_reviews = df_reviews[df_reviews['member_id'].isin(all_members)]

    # Merge reviews dataframe with recipe similarity data
    all_member_reviews = pd.merge(all_member_reviews, df_similar_recipes, on='recipe_id', how='left')

    # Initialize evaluation metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate through all users in the concatenated list
    for user_id in all_members:
        user_reviews = all_member_reviews[all_member_reviews['member_id'] == user_id]

        # Iterate through similar recipes and their ratings
        for index, row in user_reviews.iterrows():
            # Check if the similar recipe has been reviewed
            if row['similar_recipe_id'] in user_reviews['recipe_id'].values:
                # Compare ratings between original and similar recipe
                if row['rating'] == user_reviews[user_reviews['recipe_id'] == row['similar_recipe_id']]['rating'].values[0]:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                false_negatives += 1

    # Calculate precision and recall
    avg_precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    avg_recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    return avg_precision, avg_recall

def user_profiles(filtered_communities, df_reviews, df_recipes, df_members):
    # Create a DataFrame to store the user profiles
    user_profiles = pd.DataFrame()

    # Iterate through each community
    for i, community in enumerate(filtered_communities):
        # Create a DataFrame for the current community
        community_df = pd.DataFrame(community, columns=['member_id'])

        community_df['member_id'] = community_df['member_id'].astype('int64')

        # Merge with the members DataFrame to get the user names
        community_df = pd.merge(community_df, df_members[['member_id', 'member_name']], on='member_id')

        # Add a column for the community number
        community_df['community'] = i + 1

        # Add the community DataFrame to the user profiles DataFrame
        user_profiles = pd.concat([user_profiles, community_df])

    # Merge with the reviews DataFrame to get the reviews for each user
    user_profiles = pd.merge(user_profiles, df_reviews[['member_id', 'recipe_id', 'rating']], on='member_id')

    # Merge with the recipes DataFrame to get the recipe titles
    user_profiles = pd.merge(user_profiles, df_recipes[['recipe_id', 'title', 'ingredient_food_kg_names']], on='recipe_id')

    # Group by user and aggregate ingredient vectors
    user_profiles_grouped = user_profiles.groupby('member_id').agg({
        'ingredient_food_kg_names': ' '.join
    }).reset_index()

    # Vectorize the ingredient_food_kg_names column
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(user_profiles_grouped['ingredient_food_kg_names'])

    # Convert the TF-IDF matrix to a DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Merge the TF-IDF DataFrame with the user profiles DataFrame
    user_profiles_final = pd.concat([user_profiles_grouped, tfidf_df], axis=1)

    return user_profiles_final

def get_top_favorite_ingredients(user_profiles, user_id, top_n=10):
    # Filter user profile by user_id
    user_profile = user_profiles[user_profiles['member_id'] == user_id].iloc[0]

    # Get TF-IDF values for ingredients
    tfidf_values = user_profile.drop(['member_id', 'ingredient_food_kg_names'])

    # Sort ingredients by TF-IDF values and get top n
    top_ingredients = tfidf_values.sort_values(ascending=False).head(top_n)

    return top_ingredients

def get_top_favorite_ingredients_per_community(user_profiles, filtered_communities, top_n=10):
    top_favorite_per_community = {}

    # Iterate over each community
    for i, community in enumerate(filtered_communities):

        # Create a DataFrame for the current community
        community_df = pd.DataFrame(community, columns=['member_id'])

        community_df['member_id'] = community_df['member_id'].astype('int64')

        community_df = user_profiles[user_profiles['member_id'].isin(community_df['member_id'])]
        
        # Drop unnecessary columns
        community_df = community_df.drop(['member_id', 'ingredient_food_kg_names'], axis=1)

        # Sum TF-IDF values for each ingredient
        ingredient_sum = community_df.sum()

        # Sort ingredients by sum TF-IDF values and get top n
        top_ingredients = ingredient_sum.sort_values(ascending=False).head(top_n)

        # Store top favorite ingredients for the community
        top_favorite_per_community[i + 1] = top_ingredients

    return top_favorite_per_community

def community_recipe_recommendations(df_recipes, top_favorite_per_community):
    community_recommendations = {}

    # Iterate over each community
    for community, top_ingredients in top_favorite_per_community.items():
        # Filter recipes containing at least one of the community's favorite ingredients
        community_recipes = df_recipes[df_recipes['ingredient_food_kg_names'].apply(lambda x: any(ingredient in x for ingredient in top_ingredients.index))]
        
        # Remove duplicate recipes
        community_recipes = community_recipes.drop_duplicates(subset=['recipe_id'])
        
        # Calculate score for each recipe based on the TF-IDF values of favorite ingredients present
        def calculate_score(ingredients):
            score = 0
            for ingredient, tfidf_value in top_ingredients.items():
                if ingredient in ingredients:
                    score += tfidf_value  # Use TF-IDF value directly as coefficient
            return score
        
        community_recipes['score'] = community_recipes['ingredient_food_kg_names'].apply(calculate_score)
        
        # Rank recommendations based on score
        ranked_recommendations = community_recipes.sort_values(by='score', ascending=False)
        
        # Store the ranked recommendations for the community
        community_recommendations[community] = ranked_recommendations[['recipe_id', 'title', 'score']]

    return community_recommendations

def evaluate_recommendations(community_recommendations, df_reviews, filtered_communities):
    precision_at_10 = {}
    recall_at_10 = {}
    total_precision = 0
    total_recall = 0

    # Iterate over each community's recommendations
    for community_id, recommendations_df in community_recommendations.items():
        community = filtered_communities[community_id-1]

        # Create a DataFrame for the current community
        community_df = pd.DataFrame(community, columns=['member_id'])

        community_df['member_id'] = community_df['member_id'].astype('int64')
        
        # Get the list of top 10 recommended recipes
        recommended_recipe_ids = recommendations_df['recipe_id'].head(10).tolist()

        # Get the actual recipes interacted with by users from the community
        actual_interactions = df_reviews[df_reviews['member_id'].isin(community_df['member_id'])]
        actual_recipe_ids = actual_interactions['recipe_id'].unique()

        # Calculate precision@10
        true_positives = len(set(recommended_recipe_ids).intersection(actual_recipe_ids))
        precision_at_10[community_id] = true_positives / min(len(recommended_recipe_ids), len(actual_recipe_ids))

        # Calculate recall@10
        recall_at_10[community_id] = true_positives / len(actual_recipe_ids)

        # Print accuracy and recall@10 for each community
        print(f"Community {community_id}:")
        print(f"Accuracy@10: {precision_at_10[community_id]}")
        print(f"Recall@10: {recall_at_10[community_id]}")
        print()

        # Update total precision and recall
        total_precision += precision_at_10[community_id]
        total_recall += recall_at_10[community_id]

    # Calculate average precision and recall@10
    avg_precision_at_10 = total_precision / len(community_recommendations)
    avg_recall_at_10 = total_recall / len(community_recommendations)

    return avg_precision_at_10, avg_recall_at_10

def initial_obs(df):
    display(df.head(10))
    print(f"\n\033[1mAttributes:\033[0m {list(df.columns)}")
    print(f"\033[1mEntries:\033[0m {df.shape[0]}")
    print(f"\033[1mAttribute Count:\033[0m {df.shape[1]}")
    
    print(f"\n\033[1m----Null Count----\033[0m")
    print(df.isna().sum())


def plot_reviews_rating(df):
    rating_counts = df['rating'].value_counts().sort_index()

    # Plot the number of reviews for each rating
    plt.bar(rating_counts.index, rating_counts.values)

    # Add labels and title
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.title('Number of Reviews for Each Rating')

    # Show the plot
    plt.show()

def plot_num_users_num_reviews(df):
    reviews_count = df.groupby('member_id')['review_id'].count()

    # Plot the distribution of ratings
    plt.figure(figsize=(15, 6))
    plt.hist(reviews_count, bins=range(1, 16), align='left', color='skyblue', edgecolor='black')
    plt.title('Distribution of Number of Reviews per User')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Number of Members')
    plt.xticks(range(1, 16))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # Plot the distribution of number of reviews per user for the rest of the numbers
    plt.figure(figsize=(15, 6))
    plt.hist(reviews_count, bins=range(16, reviews_count.max() + 1), align='left', color='skyblue', edgecolor='black')
    plt.title('Distribution of Number of Reviews per User (16 and above)')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Number of Users')
    plt.xticks(range(16, reviews_count.max() + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def evaluate_model(model, trainset, testset):
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)

    return rmse, mae, predictions, precision, recall

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def precision_recall_at_k(predictions, k=10, threshold=3):
    """Return precision and recall at k metrics for each user"""

    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls