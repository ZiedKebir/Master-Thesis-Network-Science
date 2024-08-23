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


os. chdir('')



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

messages_2023 = messages_2023[messages_2023['chat_id'].isin(list(actin_chat_logs['chat_id']))]

messages_2023['createdDateTime'] = pd.to_datetime(messages_2023['createdDateTime'])


messages_2023['year'] = messages_2023['createdDateTime'].dt.year
messages_2023['month'] = messages_2023['createdDateTime'].dt.month
messages_2023['day'] = messages_2023['createdDateTime'].dt.day

count_per_month_2023 = messages_2023.groupby(['day','year', 'month', 'chat_id']).size().reset_index(name='count')
count_per_month_2023 = count_per_month_2023.groupby(['chat_id']).mean().reset_index()
#count_per_month_2023 = messages_2023.groupby(['day','year', 'month', 'chat_id']).size().reset_index(name='count')


# Plotting the distribution
plt.figure(figsize=(10, 6))
sns.histplot(count_per_month_2023['count'], bins=10, kde=True, color='skyblue', edgecolor='black')


count_per_month_2023['count'].describe()




# chat messages for september 2023 and forward
chat_ids_messages = list(set(messages_2023['chat_id']))
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
users_information = users_information[(users_information['Actin']=='Yes')]
users_information['node_id'] = users_information['node_id'].astype(str)


#Teleport
#logs_filtered_2023_2 = logs_filtered_2023[(logs_filtered_2023['user_1_id'].isin(list(users_information['user_id']))) & (logs_filtered_2023['user_2_id'].isin(list(users_information['user_id'])))]



squads = list(set(users_information['squad_clean_reduced']))

def get_color(squad):
    squad_colors = {
        'Squad Power Up!': 'orange',
        #'IT': '#33a02c',
        #'ADV': '#e31a1c',
        'Squad Data-Witchers': 'blue',
        #'CEO': '#6a3d9a',
        #'Communication & Marketing': '#b15928',
        #'Squad GAC': '#a6cee3',
        'Squad Ada': 'green',
        'Squad Les Arcs': 'yellow',
        #'Externe': '#fdbf6f',
        'Other':'red',
        # Default color for nodes with NaN squad value
        #'nan': '#f39c12' 
    }
    return squad_colors.get(squad, 'red')  # Default color if squad not found


### Build the network ####
G1 = nx.Graph()


#Create an empty graph with just nodes and no connections
for node_id in list(users_information['node_id']):
    G1.add_node(node_id)


G1.nodes

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
        #if count > 30:
        if count > mean_count:
            G1.add_edge(party_1_id, party_2_id)
    except Exception as e:
        chat_id_pb.append(chat_id)
        print(f"Error processing chat_id {chat_id}: {e}")




import networkx as nx
from pyvis.network import Network
import matplotlib.cm as cm

# Assuming G is your NetworkX graph

# Relabel nodes to string type if necessary
G1 = nx.relabel_nodes(G1, lambda x: str(x))

## Too many nonconnected nodes 
non_connected_nodes = [node for node, degree in G1.degree() if degree == 0]
#drop non connected nodes 
G1.remove_nodes_from(non_connected_nodes)

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
net.from_nx(G1)
net.show_buttons(filter_=["physics"])

# Customize nodes based on their degrees
degree_dict = dict(G1.degree())
for node in net.nodes:
    node_id = node['id']
    node['size'] = degree_dict[node_id] * 3  # Adjust the multiplier for better visualization
    #node['color'] = '#f39c12'  # Custom color for nodes
    squad = users_information[users_information['node_id'] == node_id]['squad_clean_reduced'].values[0]
    print(squad)
    if pd.isna(squad):
        squad = 'nan'
    node['color'] = get_color(squad)  # Assign color based on squad
    
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
net.show("network_2023_sep_forward.html")



# Power Law Distribution
plt.figure(figsize=(10, 6))
sns.histplot(G1.degree(), bins=6, kde=True, color='skyblue', edgecolor='black',legend = False)
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Distribution of Nodes Degrees')




## Network analysis: 2023 september forward

############## Node-Level Measures #######"

# 1. Degree Centrality: Measures the number of connections (edges) a node has.

degree_centrality_2023 = nx.degree_centrality(G1)


# 2. Betweenness Centrality: Measures the extent to which a node lies on paths between other nodes.

betweenness_centrality_2023 = nx.betweenness_centrality(G1)


# 3. Closeness Centrality: Measures the average length of the shortest path from a node to all other nodes.

closeness_centrality_2023 = nx.closeness_centrality(G1)


# 4. Eigenvector Centrality: Measures a node's influence based on the number of links it has to other nodes and their importance.

eigenvector_centrality_2023 = nx.eigenvector_centrality(G1)


# 5. Pagerank: Measures the importance of nodes using the PageRank algorithm.

pagerank_2023 = nx.pagerank(G1)


# 6. Katz Centrality: Measures the relative influence of a node within a network by considering the immediate neighbors and all other nodes.

#katz_centrality_2023 = nx.katz_centrality(G1)

########## Edge-Level Measures ##########

# 1. Edge Betweenness Centrality: Measures the extent to which an edge lies on paths between other nodes.
edge_betweenness_2023 = nx.edge_betweenness_centrality(G1)


########## Graph-Level Measures ##########

# 1. Density: Measures the proportion of potential connections in a network that are actual connections.

density_2023 = nx.density(G1)

# 2. Diameter: Measures the longest shortest path between any two nodes in the network.

diameter_2023 = nx.diameter(G1)

# 3. Average Clustering Coefficient: Measures the degree to which nodes in a graph tend to cluster together.

avg_clustering_2023 = nx.average_clustering(G1)


# 4. Average Shortest Path Length: Measures the average length of the shortest path between all pairs of nodes.

avg_shortest_path_length_2023 = nx.average_shortest_path_length(G1)


# 5. Transitivity: Measures the overall probability for the network to have adjacent nodes interconnected.

transitivity_2023 = nx.transitivity(G1)

# 6. Assortativity: Measures the similarity of connections in the graph with respect to the node degree.

assortativity_2023 = nx.degree_assortativity_coefficient(G1)

#################### Community Detection Measures ###############


print("Degree Centrality:", degree_centrality_2023)
print("Betweenness Centrality:", betweenness_centrality_2023)
print("Closeness Centrality:", closeness_centrality_2023)
print("Eigenvector Centrality:", eigenvector_centrality_2023)
print("PageRank:", pagerank_2023)
print("Density:", density_2023)
print("Diameter:", diameter_2023)
print("Average Shortest Path Length:", avg_shortest_path_length_2023)
print("Transitivity:", transitivity_2023)
print("Assortativity:", assortativity_2023)


users_information['degree_centrality_after_sep_2023'] = users_information['node_id'].map(degree_centrality_2023)
users_information['Betweenness_centrality_after_sep_2023'] = users_information['node_id'].map(betweenness_centrality_2023)
users_information['closeness_centrality_after_sep_2023'] = users_information['node_id'].map(closeness_centrality_2023)
users_information['eigenvector_centrality_after_sep_2023'] = users_information['node_id'].map(eigenvector_centrality_2023)
users_information['pagerank_after_sep_2023'] = users_information['node_id'].map(pagerank_2023)
users_information['density_after_sep_2023'] = [density_2023 for i in range(0,len(users_information))]
users_information['diameter_after_sep_2023'] = [diameter_2023 for i in range(0,len(users_information))]
users_information['transitivity_after_sep_2023'] = [transitivity_2023 for i in range(0,len(users_information))] 
users_information['assortativity_after_sep_2023'] = [assortativity_2023 for i in range(0,len(users_information))]
users_information['avg_shortest_path_after_sep_2023'] = [avg_shortest_path_length_2023 for i in range(0,len(users_information))]



users_information.to_excel('users_information_completed.xlsx')




### Degree Distribution: 
import numpy as np
degrees = list(dict(G1.degree).values())
log_base = 2 
min_degree = min(degrees)
max_degree = max(degrees)
bin_edges = np.logspace(np.log10(min_degree),np.log10(max_degree),num=int(np.log2(max_degree-min_degree+1))+1,base=log_base)
bin_counts,_ = np.histogram(degrees,bins=bin_edges)
bin_widths = np.diff(bin_edges)
normalized_counts = bin_counts/bin_widths

plt.figure()
plt.loglog(bin_edges[:-1],normalized_counts,marker='o',linestyle='none')
plt.show()    



plt.hist(dict(G1.degree).values(), bins=60)

#plot the distribution of the chat counts - determine when a relation should be considered
fig, ax = plt.subplots()
sns.displot(dict(G1.degree).values(),bins=30)
#ax.set_ylim(0, 200)
#ax.set_xlim(1,2000)
plt.show()

