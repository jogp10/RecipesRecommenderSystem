import networkx as nx
from networkx.algorithms import community
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNBasic, NormalPredictor, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

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
    

def collaborative_filtering(df_reviews, communities, test_fraction, user_based = True, model_type = 'KNN'):
    rmse_scores = []
    mae_scores = []
    
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

            model.fit(trainset)

            predictions = model.test(testset, verbose= False)
            rmse = accuracy.rmse(predictions, verbose= False)
            mae = accuracy.mae(predictions, verbose =False)
        
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            
            print(f"\033[1mCommunity {i + 1}\033[0m -> Size: {len(community)}")
            print(f"\033[1mRMSE ->\033[0m", rmse)
            print(f"\033[1mMAE ->\033[0m", mae)
            print()
            
        else:
            print(f"Community {i + 1} has insufficient data for recommendations.")

    
    avg_rmse = sum(rmse_scores) / len(rmse_scores)
    avg_mae = sum(mae_scores) / len(mae_scores)
    
    return avg_rmse, avg_mae

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
    ratings_count = df.groupby('member_id')['rating'].count().clip(upper=50)

    # Plot the distribution of ratings
    plt.figure(figsize=(10, 6))
    plt.hist(ratings_count, bins=range(1, 15), color='skyblue', edgecolor='black')
    plt.title('Distribution of Reviews')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Number of Members')
    plt.xticks(range(1, 20))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()