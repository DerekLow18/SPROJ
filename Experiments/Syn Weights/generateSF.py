import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#create a graph with degrees following a power law distribution
s = nx.utils.powerlaw_sequence(50, 2.5) #100 nodes, power-law exponent 2.5
G = nx.expected_degree_graph(s, selfloops=False)

print(G.nodes())
print(G.edges())
matrix = nx.to_numpy_matrix(G)
np.savetxt("sfNetworkPop50.csv",matrix,delimiter =',')
#draw and show graph
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos)
plt.show()