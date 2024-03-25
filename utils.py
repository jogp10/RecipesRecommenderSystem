import networkx as nx
from networkx.algorithms import community
import pandas as pd
import matplotlib.pyplot as plt


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


def initial_obs(df):
    display(df.head(10))
    print(f"\n\033[1mAttributes:\033[0m {list(df.columns)}")
    print(f"\033[1mEntries:\033[0m {df.shape[0]}")
    print(f"\033[1mAttribute Count:\033[0m {df.shape[1]}")
    
    print(f"\n\033[1m----Null Count----\033[0m")
    print(df.isna().sum())

