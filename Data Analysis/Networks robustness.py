# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:51:53 2024

@author: ZiedKEBIR
"""
import random 

G1_extended

G1

G

### Random Attack 
    # Before September 2023
    
    
G_attack = copy.deepcopy(G1) 
nodes = list(G1.nodes())

count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(removed_node)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)
    nodes = list(G_attack.nodes())

number_of_removed_nodes.pop()


number_of_removed_nodes_old = copy.deepcopy(number_of_removed_nodes)
size_biggest_component_old = copy.deepcopy(size_biggest_component)



    # After September 2023
G_attack = copy.deepcopy(G) 
nodes = list(G.nodes())

count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(removed_node)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)
    nodes = list(G_attack.nodes())

number_of_removed_nodes.pop()


number_of_removed_nodes_2023.pop()

number_of_removed_nodes_2023 = copy.deepcopy(number_of_removed_nodes)
size_biggest_component_2023 = copy.deepcopy(size_biggest_component)






    # 2027
G_attack = copy.deepcopy(G1_extended) 
nodes = list(G1_extended.nodes())

count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(removed_node)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)
    nodes = list(G_attack.nodes())

number_of_removed_nodes.pop()
#size_biggest_component.pop()

number_of_removed_nodes_2027 = copy.deepcopy(number_of_removed_nodes)
size_biggest_component_2027 = copy.deepcopy(size_biggest_component)


number_of_removed_nodes_old.extend([i for i in range(len(number_of_removed_nodes_old)+1,217+1)])
size_biggest_component_old.extend([0 for i in range(0,217-len(size_biggest_component_old))])

number_of_removed_nodes_2023.extend([i for i in range(len(number_of_removed_nodes_2023)+1,217+1)])
size_biggest_component_2023.extend([0 for i in range(0,217-len(size_biggest_component_2023))])




len(number_of_removed_nodes_old)
len(size_biggest_component_old)


len(number_of_removed_nodes_2023)
len(size_biggest_component_2023)

len(number_of_removed_nodes_2027)
len(size_biggest_component_2027)


total_number_nodes_old = size_biggest_component_old[0]
plt.plot(number_of_removed_nodes_old,[i/total_number_nodes_old for i in size_biggest_component_old],label='Before 2023',color='grey')
total_number_nodes_2023 = size_biggest_component_2023[0]
plt.plot(number_of_removed_nodes_2023,[i/total_number_nodes_2023 for i in size_biggest_component_2023], label='2023',color='red')
total_number_nodes_2027 = size_biggest_component_2027[0]
plt.plot(number_of_removed_nodes_2027,[i/total_number_nodes_2027 for i in size_biggest_component_2027], label = '2027',color='yellow')
plt.title('Networks Robustness: Random Attack')
plt.xlabel('Number of nodes removed ')
plt.ylabel('Size of the biggest component')
plt.legend()
plt.show()


### Targeted Attack: Degree Centality 
    # Before September 2023 
G_attack = copy.deepcopy(G) 
nodes_ordered = list(dict(sorted(dict(G.degree).items(), key=lambda x:x[1], reverse=True)).keys())


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)



number_of_removed_nodes_old_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_old_target.pop()
size_biggest_component_old_target = copy.deepcopy(size_biggest_component)

    #After September 2023 

G_attack = copy.deepcopy(G1) 
nodes_ordered = list(dict(sorted(dict(G1.degree).items(), key=lambda x:x[1], reverse=True)).keys())


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)



number_of_removed_nodes_2023_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_2023_target.pop()
size_biggest_component_2023_target = copy.deepcopy(size_biggest_component)




    #2027

G_attack = copy.deepcopy(G1_extended) 
nodes_ordered = list(dict(sorted(dict(G1_extended.degree).items(), key=lambda x:x[1], reverse=True)).keys())


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)
    

    


number_of_removed_nodes_2027_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_2027_target.pop()
size_biggest_component_2027_target = copy.deepcopy(size_biggest_component)



total_number_nodes_old_target = size_biggest_component_old_target[0]
plt.plot(number_of_removed_nodes_old_target,[i/total_number_nodes_old_target for i in size_biggest_component_old_target],label='Before 2023',color='grey')
total_number_nodes_2023_target = size_biggest_component_2023_target[0]
plt.plot(number_of_removed_nodes_2023_target,[i/total_number_nodes_2023_target for i in size_biggest_component_2023_target], label='2023',color='red')
total_number_nodes_2027_target = size_biggest_component_2027_target[0]
plt.plot(number_of_removed_nodes_2027_target[0:90],[i/total_number_nodes_2027_target for i in size_biggest_component_2027_target][0:90], label = '2027',color='yellow')
plt.title('Networks Robustness: Targeted Attack Degree Centrality')
plt.xlabel('Number of nodes removed ')
plt.ylabel('Size of the biggest component')
plt.legend()
plt.show()





### Targeted Attack: betweenness_centrality_2023
    # Before September 2023 
G_attack = copy.deepcopy(G) 
nodes_ordered = list(dict(sorted(dict(betweenness_centrality).items(), key=lambda x:x[1], reverse=True)).keys())


G_attack.nodes()


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)



number_of_removed_nodes_old_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_old_target.pop()
size_biggest_component_old_target = copy.deepcopy(size_biggest_component)

    #After September 2023 

G_attack = copy.deepcopy(G1) 
nodes_ordered = list(dict(sorted(dict(betweenness_centrality_2023).items(), key=lambda x:x[1], reverse=True)).keys())


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)




number_of_removed_nodes_2023_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_2023_target.pop()
size_biggest_component_2023_target = copy.deepcopy(size_biggest_component)




    #2027
    
    

G_attack = copy.deepcopy(G1_extended) 
nodes_ordered = list(dict(sorted(dict(betweenness_centrality_2027).items(), key=lambda x:x[1], reverse=True)).keys())


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)
    

    


number_of_removed_nodes_2027_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_2027_target.pop()
size_biggest_component_2027_target = copy.deepcopy(size_biggest_component)




total_number_nodes_old_target = size_biggest_component_old_target[0]
plt.plot(number_of_removed_nodes_old_target,[i/total_number_nodes_old_target for i in size_biggest_component_old_target],label='Before 2023',color='grey')
total_number_nodes_2023_target = size_biggest_component_2023_target[0]
plt.plot(number_of_removed_nodes_2023_target,[i/total_number_nodes_2023_target for i in size_biggest_component_2023_target], label='2023',color='red')
total_number_nodes_2027_target = size_biggest_component_2027_target[0]
plt.plot(number_of_removed_nodes_2027_target[0:90],[i/total_number_nodes_2027_target for i in size_biggest_component_2027_target][0:90], label = '2027',color='yellow')
plt.title('Networks Robustness: Targeted Attack - Betweenness Centrality')
plt.xlabel('Number of nodes removed ')
plt.ylabel('Size of the biggest component')
plt.legend()
plt.show()


### Targeted Attack: Closeness_centrality_2023
    # Before September 2023 
G_attack = copy.deepcopy(G) 

nodes_ordered = list(dict(sorted(dict(closeness_centrality).items(), key=lambda x:x[1], reverse=True)).keys())


G_attack.nodes()


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)



number_of_removed_nodes_old_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_old_target.pop()
size_biggest_component_old_target = copy.deepcopy(size_biggest_component)

    #After September 2023 

G_attack = copy.deepcopy(G1) 
nodes_ordered = list(dict(sorted(dict(closeness_centrality_2023).items(), key=lambda x:x[1], reverse=True)).keys())


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)




number_of_removed_nodes_2023_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_2023_target.pop()
size_biggest_component_2023_target = copy.deepcopy(size_biggest_component)




    #2027
    
G_attack = copy.deepcopy(G1_extended) 
nodes_ordered = list(dict(sorted(dict(closeness_centrality_2027).items(), key=lambda x:x[1], reverse=True)).keys())


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)
    

    


number_of_removed_nodes_2027_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_2027_target.pop()
size_biggest_component_2027_target = copy.deepcopy(size_biggest_component)




total_number_nodes_old_target = size_biggest_component_old_target[0]
plt.plot(number_of_removed_nodes_old_target,[i/total_number_nodes_old_target for i in size_biggest_component_old_target],label='Before 2023',color='grey')
total_number_nodes_2023_target = size_biggest_component_2023_target[0]
plt.plot(number_of_removed_nodes_2023_target,[i/total_number_nodes_2023_target for i in size_biggest_component_2023_target], label='2023',color='red')
total_number_nodes_2027_target = size_biggest_component_2027_target[0]
plt.plot(number_of_removed_nodes_2027_target[0:90],[i/total_number_nodes_2027_target for i in size_biggest_component_2027_target][0:90], label = '2027',color='yellow')
plt.title('Networks Robustness: Targeted Attack - Closeness Centrality')
plt.xlabel('Number of nodes removed ')
plt.ylabel('Size of the biggest component')
plt.legend()
plt.show()



### Targeted Attack: eigenvector_centrality
    # Before September 2023 
G_attack = copy.deepcopy(G) 

nodes_ordered = list(dict(sorted(dict(eigenvector_centrality).items(), key=lambda x:x[1], reverse=True)).keys())


G_attack.nodes()


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)



number_of_removed_nodes_old_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_old_target.pop()
size_biggest_component_old_target = copy.deepcopy(size_biggest_component)

    #After September 2023 

G_attack = copy.deepcopy(G1) 
nodes_ordered = list(dict(sorted(dict(eigenvector_centrality_2023).items(), key=lambda x:x[1], reverse=True)).keys())


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)




number_of_removed_nodes_2023_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_2023_target.pop()
size_biggest_component_2023_target = copy.deepcopy(size_biggest_component)




    #2027
    
G_attack = copy.deepcopy(G1_extended) 
nodes_ordered = list(dict(sorted(dict(eigenvector_centrality_2027).items(), key=lambda x:x[1], reverse=True)).keys())


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)
    

    


number_of_removed_nodes_2027_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_2027_target.pop()
size_biggest_component_2027_target = copy.deepcopy(size_biggest_component)




total_number_nodes_old_target = size_biggest_component_old_target[0]
plt.plot(number_of_removed_nodes_old_target,[i/total_number_nodes_old_target for i in size_biggest_component_old_target],label='Before 2023',color='grey')
total_number_nodes_2023_target = size_biggest_component_2023_target[0]
plt.plot(number_of_removed_nodes_2023_target,[i/total_number_nodes_2023_target for i in size_biggest_component_2023_target], label='2023',color='red')
total_number_nodes_2027_target = size_biggest_component_2027_target[0]
plt.plot(number_of_removed_nodes_2027_target[0:90],[i/total_number_nodes_2027_target for i in size_biggest_component_2027_target][0:90], label = '2027',color='yellow')
plt.title('Networks Robustness: Targeted Attack - Eigenvector Centrality')
plt.xlabel('Number of nodes removed ')
plt.ylabel('Size of the biggest component')
plt.legend()
plt.show()




### Targeted Attack: Pagerank
    # Before September 2023 
G_attack = copy.deepcopy(G) 

nodes_ordered = list(dict(sorted(dict(pagerank).items(), key=lambda x:x[1], reverse=True)).keys())


G_attack.nodes()


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)



number_of_removed_nodes_old_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_old_target.pop()
size_biggest_component_old_target = copy.deepcopy(size_biggest_component)

    #After September 2023 

G_attack = copy.deepcopy(G1) 
nodes_ordered = list(dict(sorted(dict(pagerank_2023).items(), key=lambda x:x[1], reverse=True)).keys())


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)




number_of_removed_nodes_2023_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_2023_target.pop()
size_biggest_component_2023_target = copy.deepcopy(size_biggest_component)




    #2027
    
G_attack = copy.deepcopy(G1_extended) 
nodes_ordered = list(dict(sorted(dict(pagerank_2027).items(), key=lambda x:x[1], reverse=True)).keys())


count = 0
number_of_removed_nodes = list()
size_biggest_component = list()
for i in nodes_ordered:
    print(i)
    count+=1
    number_of_removed_nodes.append(count)
    #removed_node = random.choice(list(G_attack.nodes()))
    G_attack.remove_node(i)
    bc = [len(c) for c in sorted(nx.connected_components(G_attack), key=len, reverse=True)][0]
    size_biggest_component.append(bc)
    

    


number_of_removed_nodes_2027_target = copy.deepcopy(number_of_removed_nodes)
number_of_removed_nodes_2027_target.pop()
size_biggest_component_2027_target = copy.deepcopy(size_biggest_component)




total_number_nodes_old_target = size_biggest_component_old_target[0]
plt.plot(number_of_removed_nodes_old_target,[i/total_number_nodes_old_target for i in size_biggest_component_old_target],label='Before 2023',color='grey')
total_number_nodes_2023_target = size_biggest_component_2023_target[0]
plt.plot(number_of_removed_nodes_2023_target,[i/total_number_nodes_2023_target for i in size_biggest_component_2023_target], label='2023',color='red')
total_number_nodes_2027_target = size_biggest_component_2027_target[0]
plt.plot(number_of_removed_nodes_2027_target[0:90],[i/total_number_nodes_2027_target for i in size_biggest_component_2027_target][0:90], label = '2027',color='yellow')
plt.title('Networks Robustness: Targeted Attack - Pagerank Centrality')
plt.xlabel('Number of nodes removed ')
plt.ylabel('Size of the biggest component')
plt.legend()
plt.show()



