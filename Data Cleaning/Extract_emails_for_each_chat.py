# -*- coding: utf-8 -*-
"""
This file allowed to extract the logs_all_with_email.json 
in order to determine the group chats that include individuals who are part of Actinvision 


"""


"""
Created on Fri May 17 13:17:49 2024

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
#Extraction ended the 10th of May 2024

os. chdir('C:/Users/ZiedKEBIR/OneDrive - Actinvision/Bureau/Master Thesis/ms-identity-python-daemon-master/ms-identity-python-daemon-master/1-Call-MsGraph-WithSecret')

# Optional logging
# logging.basicConfig(level=logging.DEBUG)
f = open('parameters.json')


config = json.load(f)

config


config.get("authority")

#config = json.load(open(sys.argv[1]))

# Create a preferably long-lived app instance which maintains a token cache.
app = msal.ConfidentialClientApplication(
    client_id=config["client_id"], authority="https://login.microsoftonline.com/172aacaf-35a5-46b3-9cac-698c57c9439b",
    client_credential=config["secret"],
    )


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
    result = app.acquire_token_silent(config["scope"], account=None)

    if not result:
        logging.info("No suitable token exists in cache. Let's get a new one from AAD.")
        result = app.acquire_token_for_client(scopes=config["scope"])
        print("***********************************************************")
   
create_token()


# all logs 
logs = pd.read_csv("logs_all.csv")

chat_ids = list(set(logs['chat_id']))


headers = {'Authorization': 'Bearer ' + result['access_token']}
url = "https://graph.microsoft.com/v1.0/chats/"+to_relaunch[0]+"?$expand=members"
response = requests.get(url, headers=headers)
response.json()


#chat_related_emails = dict()
#chat_related_emails_all = list()
#chat_one_member = list()
id2 = 876
#for i in range(1083,len(to_relaunch)):
while id2 < len(chat_ids):
    id2+=1
    print("Completion rate is:", id2/len(chat_ids))
    try:
        headers = {'Authorization': 'Bearer ' + result['access_token']}
        url = "https://graph.microsoft.com/v1.0/chats/"+chat_ids[id2]+"?$expand=members"
        response = requests.get(url, headers=headers)
        response.json()
        
        chat_related_emails['chat_id'] = chat_ids[id2]
        chat_related_emails['user_1_id'] = response.json()['members'][0]['userId']
        chat_related_emails['user_1_Name'] =response.json()['members'][0]['displayName']
        chat_related_emails['user_1_email'] = response.json()['members'][0]['email']
        chat_related_emails['user_2_id'] = response.json()['members'][1]['userId']
        chat_related_emails['user_2_Name'] =response.json()['members'][1]['displayName']
        chat_related_emails['user_2_email'] = response.json()['members'][1]['email']
        temp_dict = copy.copy(chat_related_emails)
        
    except IndexError:
        chat_one_member.append(chat_ids[id2])
        chat_related_emails['chat_id'] = chat_ids[id2]
        chat_related_emails['user_1_id'] = response.json()['members'][0]['userId']
        chat_related_emails['user_1_Name'] =response.json()['members'][0]['displayName']
        chat_related_emails['user_1_email'] = response.json()['members'][0]['email']
        chat_related_emails['user_2_id'] = ''
        chat_related_emails['user_2_Name'] =''
        chat_related_emails['user_2_email'] = ''
        temp_dict = copy.copy(chat_related_emails)

        
    except:
        print("stopped at id", chat_ids[id2])
        pass
    
    chat_related_emails_all.append(temp_dict)
    print(len(chat_related_emails_all))
    


def remove_duplicate_dicts(dicts):
    seen = set()
    unique_dicts = []
    for d in dicts:
        # Convert the dictionary to a tuple of items and check if it's in the seen set
        t = tuple(sorted(d.items()))
        if t not in seen:
            seen.add(t)
            unique_dicts.append(d)
    return unique_dicts

# Removing duplicates
chat_related_emails_all = remove_duplicate_dicts(chat_related_emails_all)

to_relaunch = [i for i in chat_ids  if i not in [j['chat_id']for j in chat_related_emails_all]]

len(chat_ids) - len([j['chat_id']for j in chat_related_emails_all])


import json
with open('logs_all_with_emails.json', 'w') as f:
    json.dump(chat_related_emails_all, f)


