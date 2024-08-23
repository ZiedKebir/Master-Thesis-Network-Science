# -*- coding: utf-8 -*-
"""
Created on Thu May 30 23:00:02 2024

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


os. chdir('C:/Users/ZiedKEBIR/OneDrive - Actinvision/Bureau/Master Thesis/ms-identity-python-daemon-master/ms-identity-python-daemon-master/1-Call-MsGraph-WithSecret')


## Import the dataframe ####



count_chats = pd.read_excel('count_chats_final_one_1.xlsx')
chat_parties_actin_only = pd.read_excel('logs_actin_only.xlsx')
count_chats_actin_only = count_chats[count_chats['chat_id'].isin(chat_parties_actin_only['chat_id'])]



#convert all the id users into an ID which is incremented by one. so that I can recegnize what user each node represents
all_user_ids = list()
all_user_ids.extend(list(chat_parties_actin_only['user_1_id']))
all_user_ids.extend(list(chat_parties_actin_only['user_2_id']))
all_user_ids = list(set(all_user_ids))

id_parties  = [{'node_id':i+1,'user_id':all_user_ids[i]}for i in range(0,len(all_user_ids))]
df = pd.DataFrame(id_parties)
df.to_excel('id_parties.xlsx')


#plot the distribution of the chat counts - determine when a relation should be considered
fig, ax = plt.subplots()
sns.displot(count_chats_actin_only, x="count", bins = range(0,48096,3000))
#ax.set_ylim(0, 200)
#ax.set_xlim(1,50000)
plt.show()

count_chats_actin_only.describe()

### Simple case consider a link to exist only if in absolute terms (in all T) the number of chats sent between individuals in at least bigger than the average number of messages sent for all the company 
G = nx.Graph()

#Create an empty graph with just nodes and no connections
for node_id in df['node_id']:
    G.add_node(node_id)
    


# add links if the count is higher than the global average of the number of links 
chat_id_pb  = list()


for chat_id in count_chats_actin_only['chat_id']:
    try:
        party_1 = chat_parties_actin_only[chat_parties_actin_only['chat_id'] == chat_id]['user_1_id'].values[0]
        party_1_id = df[df['user_id'] == party_1]['node_id'].values[0]
        party_2 = chat_parties_actin_only[chat_parties_actin_only['chat_id'] == chat_id]['user_2_id'].values[0]
        party_2_id = df[df['user_id'] == party_2]['node_id'].values[0]
        
        count = count_chats[count_chats['chat_id'] == chat_id]['count'].values[0]
        mean_count = count_chats['count'].mean()
        
        if count > mean_count:
            G.add_edge(party_1_id, party_2_id)
    except Exception as e:
        chat_id_pb.append(chat_id)
        print(f"Error processing chat_id {chat_id}: {e}")


chat_parties_actin_only_pb = chat_parties_actin_only[chat_parties_actin_only['chat_id'].isin(chat_id_pb)]

# Try to look for the ids in the missing chats 
chat_messages = pd.read_csv('final_dataset_for_analysis_final_one_1.csv')
for i in chat_id_pb:
    id_1 = chat_messages[chat_messages['chat_id']==i]['id_user_1']
    id_2 = chat_messages[chat_messages['chat_id']==i]['id_user_2']


G.remove_node(42)
G.remove_node(83)


nodes_connected = list()

# extract the nodes that are not connected and delete them from the graph 
for i in G.edges():
    nodes_connected.append(i[0])
    nodes_connected.append(i[1])
nodes_connected = list(set(nodes_connected))
for node_id in df['node_id']:
    if node_id not in nodes_connected:
        G.remove_node(node_id)


degree_dict = dict(G.degree())
node_size = [v * 100 for v in degree_dict.values()]


# Set the size of the figure
plt.figure(figsize=(50, 25),dpi=60)  # You can adjust the width and height as needed
pos = nx.random_layout(G)
# Draw the graph
#nx.draw(G, pos, with_labels=True, node_size=node_size, node_color='lightblue', font_size=10, font_color='darkblue')

# Show the plot
plt.show()

from pyvis.network import Network
net = Network(notebook=True)

net.from_nx(G)

net.show('example.html')



nx.draw(G)
G.has_edge(party_1_id,party_2_id)



############################# FILL MISSING DATA #######################################
## I'm going to fill the missing users with user names and user ids based on data from the large data set that contains chat message logs
f = pd.read_csv('final_dataset_for_analysis_final_one_1.csv') ##Final dataset containing messages


a = pd.read_excel('logs_actin_only.xlsx') ## logs of the chats

log_actin_only_missing = a[pd.isna(a['user_1_id']) | pd.isna(a['user_2_id'])] ##the rows that are missing at least a user


f_filtered = f[f['chat_id']=='19:3a418c20-9000-41ec-8deb-e2dfaf82419d_358f0194-6b0e-4dd3-af35-c24fe8a9ec87@unq.gbl.spaces']


i = 0
chat_id = log_actin_only_missing.iloc[i]['chat_id']
filtered_chat_messages = f[f['chat_id']==chat_id]
message_sender_name = [i for i in list(set(filtered_chat_messages['message_sender_name'])) if pd.isna(i)==False]
message_sender_ids = [{i:filtered_chat_messages[filtered_chat_messages['message_sender_name']==i]['message_sender_id'].values[0]} for i in message_sender_name ]
print(message_sender_ids)



def remove_duplicate_dicts(list_of_dicts):
    """
    Removes duplicate dictionaries from a list of dictionaries.

    Args:
    list_of_dicts (list): A list containing dictionaries.

    Returns:
    list: A list with unique dictionaries.
    """
    # Remove duplicates by converting dictionaries to tuples (hashable) and back to dictionaries
    unique_list_of_dicts = [dict(t) for t in {tuple(d.items()) for d in list_of_dicts}]
    return unique_list_of_dicts



list_missing_values = list()
for i in range(0,len(log_actin_only_missing)): 
    message_sender_ids = list()
    chat_id = log_actin_only_missing.iloc[i]['chat_id']
    filtered_chat_messages = f[f['chat_id']==chat_id]
    ## extract users from message_sender_name and message_sender_id
    message_sender_name = [i for i in list(set(filtered_chat_messages['message_sender_name'])) if pd.isna(i)==False] #list containing the unique message sender names in the chat 
    if len(message_sender_name) > 0:
        message_sender_ids.extend([{i:filtered_chat_messages[filtered_chat_messages['message_sender_name']==i]['message_sender_id'].values[0]} for i in message_sender_name ])
    else:
        pass
    ## extract users from name_user_1 and id_user_1 
    message_sender_name = [i for i in list(set(filtered_chat_messages['name_user_1'])) if pd.isna(i)==False] #list containing the unique name_user_1 in the chat 
    if len(message_sender_name) > 0:
        message_sender_ids.extend([{i:filtered_chat_messages[filtered_chat_messages['name_user_1']==i]['id_user_1'].values[0]} for i in message_sender_name ])
    else:
        pass
    
    ## extract users from name_user_2 and id_user_2
    message_sender_name = [i for i in list(set(filtered_chat_messages['name_user_2'])) if pd.isna(i)==False] #list containing the unique name_user_1 in the chat 
    if len(message_sender_name) > 0:
        message_sender_ids.extend([{i:filtered_chat_messages[filtered_chat_messages['name_user_2']==i]['id_user_2'].values[0]} for i in message_sender_name ])
    else:
        pass
    
    list_missing_values.append({chat_id:remove_duplicate_dicts(message_sender_ids)})
    

len(list(list_missing_values[3].values())[0])


#Extract the ids that even after correction still contain missing values
list_missing_values_not_found = list()
for i in list_missing_values: 
    print(list(i.values()))
    print(list(i.values())[0])


## in this list I have the values that can't at all be filled because I d'ont have the data 
list_missing_values_not_found = [i for i in list_missing_values if len(list(i.values())[0])!=2]

list_missing_values_to_add = [i for i in list_missing_values if len(list(i.values())[0])==2]



[list(i.values())[0] for i in list_missing_values_to_add[0]['19:09c7dc9e-5871-4ba6-b595-4020749b1925_f168f289-6808-4ba8-b18a-f6554e6d8f2b@unq.gbl.spaces']]




x = a2[a2['chat_id']=='19:09c7dc9e-5871-4ba6-b595-4020749b1925_f168f289-6808-4ba8-b18a-f6554e6d8f2b@unq.gbl.spaces']



y = a[a['chat_id']=='19:09c7dc9e-5871-4ba6-b595-4020749b1925_f168f289-6808-4ba8-b18a-f6554e6d8f2b@unq.gbl.spaces']


## Fill the missing values of the chat logs dataframe with the missing user names and user ids
for i in range(0,len(list_missing_values_to_add)):
    chat_id = list(list_missing_values_to_add[i].keys())[0]
    names = [list(j.keys())[0] for j in list_missing_values_to_add[i][chat_id]]
    ids =   [list(j.values())[0] for j in list_missing_values_to_add[i][chat_id]]
    a.loc[a['chat_id']==chat_id,'user_1_Name'] = names[0]
    a.loc[a['chat_id']==chat_id,'user_2_Name'] = names[1]
    a.loc[a['chat_id']==chat_id,'user_1_id'] = ids[0]
    a.loc[a['chat_id']==chat_id,'user_2_id'] = ids[1]

a = a[a['chat_id'].isin(list(log_actin_only_missing['chat_id']))!=True]

a.to_excel('Data_without_missing_values.xlsx')



################################# Extract the gender of the network memebers ################"""

prenoms_genre = pd.read_excel("Prenoms.xlsx")
id_parties = pd.read_excel('id_parties.xlsx')
actin_chat_logs = pd.read_excel('Data_without_missing_values_corrected.xlsx')

actin_chat_logs = actin_chat_logs.drop_duplicates(subset='chat_id')
actin_chat_logs = actin_chat_logs.reset_index(drop=True)




prénom1 = [{'id': actin_chat_logs.iloc[i]['user_1_id'] , 'prénom':actin_chat_logs.iloc[i]['user_1_Name'].split(' ')[0], 'Nom': actin_chat_logs.iloc[i]['user_1_Name'].split(' ')[1]}
    
           if not pd.isna(actin_chat_logs.iloc[i]['user_1_Name'])
           else {'id' : actin_chat_logs.iloc[i]['user_1_id'],'prénom':actin_chat_logs.iloc[i]['user_1_Name']}
           for i in range(0,len(actin_chat_logs))]


prénom2 = [{'id': actin_chat_logs.iloc[i]['user_2_id'] , 'prénom':actin_chat_logs.iloc[i]['user_2_Name'].split(' ')[0], 'Nom': actin_chat_logs.iloc[i]['user_2_Name'].split(' ')[1]}
           if not pd.isna(actin_chat_logs.iloc[i]['user_2_Name'])
           else {'id' : actin_chat_logs.iloc[i]['user_2_id'],'prénom':actin_chat_logs.iloc[i]['user_2_Name']}
           for i in range(0,len(actin_chat_logs))]


prénom = prénom1 + prénom2

def remove_dup_list_dict (list_dict):
    non_dup_list = list()
    non_dup_list.append(list_dict[0])
    for i in list_dict:
        keys = [i['id'] for i in non_dup_list]
        if i['id'] not in keys:
            non_dup_list.append(i)
        else:
            pass
    return non_dup_list
        
non_dup_prénom = remove_dup_list_dict(prénom)




for i in non_dup_prénom: 
    if 'Nom' not in i.keys():
        i['Nom']= ''
    try: 
        genre = prenoms_genre[prenoms_genre['01_prenom'].str.upper()==i['prénom'].upper()]['02_genre'].values

        i['genre'] = genre[0]
     
    except:
        i['genre'] = ''



nom_prenom_genre = pd.DataFrame(non_dup_prénom)
nom_prenom_genre.to_excel('Nom_prénom_genre.xlsx')

### Vérifier les utilistareurs qui ont plus qu'un ID.

##We have 5 users with multiple ids
x = [{i['prénom'],i['Nom']} for i in non_dup_prénom]
count_occ = list()
nom_prenom_passé = list()
for i in x: 
    if i not in nom_prenom_passé:
        nom_prenom_passé.append(i)
        count_occ.append([i,x.count(i)])
multi_id_users = [i for i in count_occ if i[1]>1]
## We have 5 users that have multiple ids on microsoft teams. I parsed through the Data_without_missing_values_corrected.xlsx to maunually unify the ids of the five users 

# Extract two datasets one prior to 2024 and one after 2024 

messages = pd.read_csv('final_dataset_for_analysis_final_one_1.csv')
#df['createdDateTime'] = pd.to_datetime(df['createdDateTime'], utc=True)


messages['createdDateTime'] = [pd.to_datetime(i).date() for i in list(messages['createdDateTime'])]



cutoff_date = pd.Timestamp('2023-09-01').date()

messages_2023 = messages[messages['createdDateTime'] >= cutoff_date]

messages_prior_2023  =  messages[messages['createdDateTime'] < cutoff_date]

############# Analysis of network of September 2023 forward #############################
messages_2023 = messages_2023[messages_2023['chat_id'].isin(list(actin_chat_logs['chat_id']))]
# chat messages for september 2023 and forward
chat_ids_messages = list(set(messages_2023['chat_id']))
# logs of the chats that are only for september 2023 and forward
logs_filtered_2023 = actin_chat_logs[actin_chat_logs['chat_id'].isin(chat_ids_messages)]
count_2023 = [{'chat_id':i,'count': list(messages_2023['chat_id']).count(i)} for i in chat_ids_messages]
df_count_2023 = pd.DataFrame(count_2023)

#convert all the id users into an ID which is incremented by one. so that I can recegnize what user each node represents
all_user_ids_2023 = list()
all_user_ids_2023.extend(list(logs_filtered_2023['user_1_id']))
all_user_ids_2023.extend(list(logs_filtered_2023['user_2_id']))
all_user_ids_2023 = list(set(all_user_ids_2023))

# Extract the full name and gender of users who sent messages in September 2023 and forward
nom_prénom_genre_2023 = pd.read_excel('nom_prénom_genre.xlsx')
nom_prénom_genre_2023 = nom_prénom_genre_2023[nom_prénom_genre_2023['id'].isin(all_user_ids_2023)]

id_parties  = [{'node_id':i+1,'user_id':all_user_ids_2023[i]}for i in range(0,len(all_user_ids_2023))]
df = pd.DataFrame(id_parties)
df.to_excel('id_parties_2023.xlsx')


#plot the distribution of the chat counts - determine when a relation should be considered
fig, ax = plt.subplots()
sns.displot(df_count_2023['count'])
#ax.set_ylim(0, 200)
#ax.set_xlim(1,50000)
plt.show()
df_count_2023.describe()




G = nx.Graph()

#Create an empty graph with just nodes and no connections
for node_id in list(df['node_id']):
    G.add_node(node_id)
    

len(df['node_id'])



# add links if the count is higher than the global average of the number of links 
chat_id_pb  = list()

chat_id = chat_ids_messages[1]



party_1 = logs_filtered_2023[logs_filtered_2023['chat_id'] == chat_id]['user_1_id'].values[0]
party_1_id = df[df['user_id'] == party_1]['node_id'].values[0]


for chat_id in df_count_2023['chat_id']:
    try:
        print(chat_id)
        party_1 = logs_filtered_2023[logs_filtered_2023['chat_id'] == chat_id]['user_1_id'].values[0]
        party_1_id = df[df['user_id'] == party_1]['node_id'].values[0]
        party_2 = logs_filtered_2023[logs_filtered_2023['chat_id'] == chat_id]['user_2_id'].values[0]
        party_2_id = df[df['user_id'] == party_2]['node_id'].values[0]
            
        count = df_count_2023[df_count_2023['chat_id'] == chat_id]['count'].values[0]
        mean_count = df_count_2023['count'].mean()
            
        if count > mean_count:
            G.add_edge(party_1_id, party_2_id)
    except Exception as e:
        chat_id_pb.append(chat_id)
        print(f"Error processing chat_id {chat_id}: {e}")



# Create a NetworkX graph (replace G with your actual graph)
#G = nx.karate_club_graph()  # Example graph; replace with your graph

# Relabel nodes to string type if necessary
G = nx.relabel_nodes(G, lambda x: str(x))

# Initialize pyvis Network object
net = Network(notebook=True, height='75%', width='75%', bgcolor='#222222', font_color='white')

# Set the physics layout of the network
net.barnes_hut()
net.show_buttons(filter_=["physics"])

# Convert NetworkX graph to pyvis network
net.from_nx(G)

# Customize nodes based on their degrees
degree_dict = dict(G.degree())
for node in net.nodes:
    node_id = node['id']
    node['size'] = degree_dict[node_id] * 1.5  # Adjust the multiplier for better visualization

# Customize the layout to make the network static
#net.repulsion(node_distance=1000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)


# Save and show the network
net.show("network2.html")



nx.draw(G)
G.has_edge(party_1_id,party_2_id)



logs_filtered_2023['user_']


import networkx as nx
from pyvis.network import Network

# Initialize the NetworkX graph
G = nx.Graph()

# Add nodes and edges to the graph
G.add_node(1, label='Node 1', color='red')
G.add_node(2, label='Node 2', color='blue')
G.add_edge(1, 2)

# Create a Network object
net = Network()

# Convert NetworkX graph to PyVis
net.from_nx(G)

# Add legend nodes and edges
legend_x = 0
legend_y = -300
net.add_node('Legend Red Nodes', label='Red Nodes', color='red', x=legend_x, y=legend_y, fixed=True)
net.add_node('Legend Blue Nodes', label='Blue Nodes', color='blue', x=legend_x, y=legend_y - 50, fixed=True)

# Optionally, add edges to the legend for clarity (transparent edges)
net.add_edge('Legend Red Nodes', 'Legend Blue Nodes', color='transparent')

# Generate and show the network
net.plot()