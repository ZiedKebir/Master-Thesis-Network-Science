# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:32:57 2024

@author: ZiedKEBIR
"""
import pandas as pd 
import os 
import networkx  as nx
import matplotlib.pyplot as plt 
import math
import seaborn as sns 
import copy
from datetime import datetime
from pyvis.network import Network


os. chdir('C:/Users/ZiedKEBIR/OneDrive - Actinvision/Bureau/Master Thesis/ms-identity-python-daemon-master/ms-identity-python-daemon-master/1-Call-MsGraph-WithSecret')



############################### Count relationship by month #######################
messages = pd.read_csv('final_dataset_for_analysis_final_one_1.csv')
#df['createdDateTime'] = pd.to_datetime(df['createdDateTime'], utc=True)

messages['createdDateTime'] = [pd.to_datetime(i).date() for i in list(messages['createdDateTime'])]



actin_chat_logs = pd.read_excel('Data_without_missing_values_corrected.xlsx')

actin_chat_logs = actin_chat_logs.drop_duplicates(subset='chat_id')
actin_chat_logs = actin_chat_logs.reset_index(drop=True)



cutoff_date = pd.Timestamp('2023-08-01').date()
messages_2023 = messages[messages['createdDateTime'] >= cutoff_date]


messages_prior_2023  =  messages[messages['createdDateTime'] < cutoff_date]

messages_prior_2023  = messages_prior_2023 [messages_prior_2023 ['chat_id'].isin(list(actin_chat_logs['chat_id']))]

messages_prior_2023 ['createdDateTime'] = pd.to_datetime(messages_prior_2023 ['createdDateTime'])


messages_prior_2023 ['year'] = messages_prior_2023 ['createdDateTime'].dt.year
messages_prior_2023 ['month'] = messages_prior_2023 ['createdDateTime'].dt.month
messages_prior_2023 ['day'] = messages_prior_2023 ['createdDateTime'].dt.day

count_per_month_2023 = messages_prior_2023 .groupby(['day','year', 'month', 'chat_id']).size().reset_index(name='count')
count_per_month_2023 = count_per_month_2023.groupby(['chat_id']).mean().reset_index()
#count_per_month_2023 = messages_2023.groupby(['day','year', 'month', 'chat_id']).size().reset_index(name='count')


count_per_month_2023['count'].describe()




# chat messages for september 2023 and forward
chat_ids_messages = list(set(messages_prior_2023 ['chat_id']))
# logs of the chats that are only for september 2023 and forward

logs_filtered_2023 = actin_chat_logs[actin_chat_logs['chat_id'].isin(chat_ids_messages)]

logs_filtered_2023.to_excel('logs_filtered_2023.xlsx')



#plot the distribution of the chat counts - determine when a relation should be considered
fig, ax = plt.subplots()
sns.displot(count_per_month_2023['count'])
#ax.set_ylim(0, 200)
#ax.set_xlim(1,2000)
plt.show()


#convert all the id users into an ID which is incremented by one. so that I can recegnize what user each node represents
all_user_ids_2023 = list()
all_user_ids_2023.extend(list(logs_filtered_2023['user_1_id']))
all_user_ids_2023.extend(list(logs_filtered_2023['user_2_id']))
all_user_ids_2023 = list(set(all_user_ids_2023))

# Extract the full name and gender of users who sent messages in September 2023 and forward
nom_prénom_genre_2023 = pd.read_excel('nom_prénom_genre_corrected.xlsx')
nom_prénom_genre_2023 = nom_prénom_genre_2023[nom_prénom_genre_2023['id'].isin(all_user_ids_2023)]

id_parties  = [{'node_id':i+1,'user_id':all_user_ids_2023[i]}for i in range(0,len(all_user_ids_2023))]
df = pd.DataFrame(id_parties)
df.to_excel('id_parties_2023.xlsx')


############### Building the network #############
users_information = pd.read_excel('users_information_completed.xlsx')
#Exclude non squad user in order to solely evaluate the inter teams communication
#users_information = users_information[(users_information['Actin']=='Oui') & (users_information['squad_clean_reduced']!='Other')]
users_information = users_information[(users_information['Actin']=='Oui')]


#Teleport
#logs_filtered_2023_2 = logs_filtered_2023[(logs_filtered_2023['user_1_id'].isin(list(users_information['user_id']))) & (logs_filtered_2023['user_2_id'].isin(list(users_information['user_id'])))]

def get_color(genre):
    squad_colors = {
        'f': 'red',
        'm': 'blue'
    }
    return squad_colors.get(genre, 'grey')  # Default color if squad not found




### Build the network ####
G_gender = nx.Graph()

#Create an empty graph with just nodes and no connections
for node_id in list(users_information['node_id']):
    G_gender.add_node(node_id)



# add links if the count is higher than the global average of the number of links 
chat_id_pb  = list()

## Add the links and only consider a relationship if the number of messages sent between two individuals is higher than the average number of messages sent per month
for chat_id in list(set(count_per_month_2023['chat_id'])):
    try:
        print(chat_id)
        party_1 = logs_filtered_2023[logs_filtered_2023['chat_id'] == chat_id]['user_1_id'].values[0]
        print(party_1)
        party_1_id = users_information[users_information['user_id'] == party_1]['node_id'].values[0]
        party_2 = logs_filtered_2023[logs_filtered_2023['chat_id'] == chat_id]['user_2_id'].values[0]
        party_2_id = users_information[users_information['user_id'] == party_2]['node_id'].values[0]
        count = count_per_month_2023[count_per_month_2023['chat_id'] == chat_id]['count'].values[0]
        mean_count = count_per_month_2023['count'].mean() 
        if count > mean_count:
            G_gender.add_edge(party_1_id, party_2_id)
    except Exception as e:
        chat_id_pb.append(chat_id)
        print(f"Error processing chat_id {chat_id}: {e}")




import networkx as nx
from pyvis.network import Network
import matplotlib.cm as cm

# Assuming G is your NetworkX graph

# Relabel nodes to string type if necessary
G_gender = nx.relabel_nodes(G_gender, lambda x: str(x))

## Too many nonconnected nodes 
non_connected_nodes = [node for node, degree in G_gender.degree() if degree == 0]
#drop non connected nodes 
G_gender.remove_nodes_from(non_connected_nodes)

# Detect communities
#partition = community_louvain.best_partition(G)
#print(f"Communities: {partition}")

# Initialize pyvis Network object
net = Network(notebook = True, cdn_resources = "remote",
                bgcolor = "#222222",
                font_color = "white",
                height = "750px",
                width = "100%",
                select_menu = True,
                filter_menu = True,
)
# Set the physics layout of the network
net.barnes_hut()

# Convert NetworkX graph to pyvis network
net.from_nx(G_gender)
net.show_buttons(filter_=["physics"])

# Customize nodes based on their degrees
degree_dict = dict(G_gender.degree())
for node in net.nodes:
    node_id = node['id']
    node['size'] = degree_dict[node_id] * 3  # Adjust the multiplier for better visualization
    #node['color'] = '#f39c12'  # Custom color for nodes
    genre = users_information[users_information['node_id'] == int(node_id)]['genre'].values[0]
    print(genre)
    if pd.isna(genre):
        genre = 'nan'
    node['color'] = get_color(genre)  # Assign color based on squad
    
    # Assign community to group nodes with same color
    #node['group'] = partition[int(node_id)]
    

    
    # Assign community to group nodes with same color
    #node['group'] = partition[int(node_id)]
    
# Customize edges (optional)
for edge in net.edges:
    edge['color'] = '#ecf0f1'
    edge['width'] = 2  # Adjust edge width for better visibility

# Customize the layout to make the network static
net.repulsion(node_distance=2000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)

# Add labels to nodes for better context
for node in net.nodes:
    node['label'] = node['id']

# Save and show the network
net.show("network_2023_prior_sep_genre.html")



## Network analysis: 2023 september forward

############## Node-Level Measures #######"

# 1. Degree Centrality: Measures the number of connections (edges) a node has.

degree_centrality = nx.degree_centrality(G_gender)


# 2. Betweenness Centrality: Measures the extent to which a node lies on paths between other nodes.

betweenness_centrality = nx.betweenness_centrality(G_gender)


# 3. Closeness Centrality: Measures the average length of the shortest path from a node to all other nodes.

closeness_centrality = nx.closeness_centrality(G_gender)


# 4. Eigenvector Centrality: Measures a node's influence based on the number of links it has to other nodes and their importance.

eigenvector_centrality = nx.eigenvector_centrality(G_gender)


# 5. Pagerank: Measures the importance of nodes using the PageRank algorithm.

pagerank = nx.pagerank(G_gender)


# 6. Katz Centrality: Measures the relative influence of a node within a network by considering the immediate neighbors and all other nodes.

#katz_centrality = nx.katz_centrality(G)

########## Edge-Level Measures ##########

# 1. Edge Betweenness Centrality: Measures the extent to which an edge lies on paths between other nodes.
edge_betweenness = nx.edge_betweenness_centrality(G_gender)


########## Graph-Level Measures ##########

# 1. Density: Measures the proportion of potential connections in a network that are actual connections.

density = nx.density(G_gender)

# 2. Diameter: Measures the longest shortest path between any two nodes in the network.

diameter = nx.diameter(G_gender)

# 3. Average Clustering Coefficient: Measures the degree to which nodes in a graph tend to cluster together.

avg_clustering = nx.average_clustering(G_gender)


# 4. Average Shortest Path Length: Measures the average length of the shortest path between all pairs of nodes.

avg_shortest_path_length = nx.average_shortest_path_length(G_gender)


# 5. Transitivity: Measures the overall probability for the network to have adjacent nodes interconnected.

transitivity = nx.transitivity(G_gender)

# 6. Assortativity: Measures the similarity of connections in the graph with respect to the node degree.

assortativity = nx.degree_assortativity_coefficient(G_gender)

#################### Community Detection Measures ###############
import networkx.algorithms.community as nx_comm
communities = nx_comm.greedy_modularity_communities(G_gender)
modularity = nx_comm.modularity(G_gender, communities)

################### Shortest Path #########################

print("Degree Centrality:", degree_centrality)
print("Betweenness Centrality:", betweenness_centrality)
print("Closeness Centrality:", closeness_centrality)
print("Eigenvector Centrality:", eigenvector_centrality)
print("PageRank:", pagerank)
print("Density:", density)
print("Diameter:", diameter)
print("Average Clustering Coefficient:", avg_clustering)
print("Average Shortest Path Length:", avg_shortest_path_length)
print("Transitivity:", transitivity)
print("Assortativity:", assortativity)
print("Modularity:", modularity)

users_information['node_id'] = users_information['node_id'].astype(str)

users_information['degree_centrality_before_sep_2023'] = users_information['node_id'].map(degree_centrality)
users_information['Betweenness_centrality_before_sep_2023'] = users_information['node_id'].map(betweenness_centrality)
users_information['closeness_centrality_before_sep_2023'] = users_information['node_id'].map(closeness_centrality)
users_information['eigenvector_centrality_before_sep_2023'] = users_information['node_id'].map(eigenvector_centrality)
users_information['pagerank_before_sep_2023'] = users_information['node_id'].map(pagerank)
users_information['density_before_sep_2023'] = [density for i in range(0,len(users_information))]
users_information['diameter_before_sep_2023'] = [diameter for i in range(0,len(users_information))]
users_information['transitivity_before_sep_2023'] = [transitivity for i in range(0,len(users_information))] 
users_information['assortativity_before_sep_2023'] = [assortativity for i in range(0,len(users_information))]


users_information.to_excel('users_information_completed.xlsx')

