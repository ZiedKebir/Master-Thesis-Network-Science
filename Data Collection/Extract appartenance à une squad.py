# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:26:43 2024

@author: ZiedKEBIR
"""

import sys  # For simplicity, we'll read config file from 1st CLI param sys.argv[1]
import json
import logging

import requests
import msal

import os
import pandas as pd
import copy
import time

os. chdir('C:/Users/ZiedKEBIR/OneDrive - Actinvision/Bureau/Master Thesis/ms-identity-python-daemon-master/ms-identity-python-daemon-master/1-Call-MsGraph-WithSecret')

# Optional logging
# logging.basicConfig(level=logging.DEBUG)
f = open('parameters.json')


config = json.load(f)

config["client_id"]


config.get("authority")

#config = json.load(open(sys.argv[1]))

# Create a preferably long-lived app instance which maintains a token cache.
app = msal.ConfidentialClientApplication(
    client_id=config["client_id"], authority="...",
    client_credential=config["secret"],
    )

# The pattern to acquire a token looks like this.
result = None

# Firstly, looks up a token from cache
# Since we are looking for token for the current app, NOT for an end user,
# notice we give account parameter as None.
result = app.acquire_token_silent(config["scope"], account=None)



if not result:
    logging.info("No suitable token exists in cache. Let's get a new one from AAD.")
    result = app.acquire_token_for_client(scopes=config["scope"])
    print(result)
    

def create_token():
    global result

    config.get("authority")

    #config = json.load(open(sys.argv[1]))

    # Create a preferably long-lived app instance which maintains a token cache.
    app = msal.ConfidentialClientApplication(
        client_id=config["client_id"], authority="...",
        client_credential=config["secret"],
        )

    # The pattern to acquire a token looks like this.
    result = None

    # Firstly, looks up a token from cache
    # Since we are looking for token for the current app, NOT for an end user,
    # notice we give account parameter as None.
    result = app.acquire_token_silent(scopes=config["scope"], account=None)

    if not result:
        logging.info("No suitable token exists in cache. Let's get a new one from AAD.")
        result = app.acquire_token_for_client(scopes=config["scope"])
        print(result)
        print("***********************************************************")
    

# Extract user information: squad, country, job title ...
actin_chat_logs = pd.read_excel('Data_without_missing_values_corrected.xlsx')
actin_chat_logs = actin_chat_logs.drop_duplicates(subset='chat_id')
actin_chat_logs = actin_chat_logs.reset_index(drop=True)




all_user_ids = list()
all_user_ids.extend(list(actin_chat_logs['user_1_id']))
all_user_ids.extend(list(actin_chat_logs['user_2_id']))
all_user_ids = list(set(all_user_ids))

id_parties  = [{'node_id':i+1,'user_id':all_user_ids[i]}for i in range(0,len(all_user_ids))]

count = 0
for i in range(0,len(id_parties)):
    id_user = id_parties[i]['user_id'] 
    headers = {'Authorization': 'Bearer ' + result['access_token']}
    
    try:
        url = "https://graph.microsoft.com/beta/users/"+id_user
        response = requests.get(url, headers=headers)
        f2 = response.json()
        id_parties[i]['city']=f2['city']
        id_parties[i]['squad']=f2['department']
        id_parties[i]['hiredate']=f2['employeeHireDate']
        id_parties[i]['Nom']=f2['displayName']
        id_parties[i]['jobTitle']=f2['jobTitle']
        id_parties[i]['email']=f2['mail']
        id_parties[i]['country']=f2['country']
        count+=1
        print('Inf of user '+id_user+' added successfully')
        print(count)
    except:
        id_parties[i]['city']=''
        id_parties[i]['squad']=''
        id_parties[i]['country']=''
        id_parties[i]['hiredate']=''
        id_parties[i]['Nom']=''
        id_parties[i]['jobTitle']=''
        id_parties[i]['email']=''
        id_parties[i]['country']=''

user_information = pd.DataFrame(id_parties)

user_information.to_excel('users_information.xlsx')


## Can't extract data of the users who already left the company. Thus I need to parse through the logs dataset find the user id with missing information and extract the missing data
empty_user_information = user_information[user_information['Nom']==''] 
empty_user_information_ids = [i for i in list(empty_user_information['user_id']) if not pd.isna(i)]

df.loc[df['A'] > 1, 'B'] = 100


for i in empty_user_information_ids:
    try:
        relevant_logs = actin_chat_logs[actin_chat_logs['user_1_id']==i]
        user_name = list(relevant_logs['user_1_Name'])[0]
    except:
        relevant_logs = actin_chat_logs[actin_chat_logs['user_2_id']==i]
        user_name = list(relevant_logs['user_2_Name'])[0]
    
    user_information.loc[user_information['user_id']==i,'Nom'] = user_name

## Add gender
nom_prénom_genre= pd.read_excel('Nom_prénom_genre_corrected.xlsx')
all_user_ids = [i for i in list(user_information['user_id']) if not pd.isna(i)]


# I had to complete som of the genders manually
for i in all_user_ids:
    print(i)
    genre = nom_prénom_genre[nom_prénom_genre['id']==i]['genre'].values[0]
    user_information.loc[user_information['user_id']==i,'genre'] = genre




user_information.to_excel('users_information_completed.xlsx')

## I have to build the network by only considering nodes with the actinvision property Oui. For after 2023 only consider 
