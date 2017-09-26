import nest
import pylab
import nest.topology as topp
import networkx as nx
import matplotlib.pyplot as plt

#generate all nodes
#neurons
pop1 = nest.Create("iaf_psc_alpha", 50)
nest.SetStatus(pop1, {"I_e": 376.0})
#multimeter to detect membrance potential
multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})

Ex = 20
d = 1.0

conn_dict = {"rule": "fixed_indegree", "indegree": Ex} #connection dictionary
syn_dict = {"delay": d}

nest.Connect(pop1, pop1, conn_dict)
nest.Connect(multimeter, [1])

#show me the connections
#print(nest.GetConnections())
#pop1_layer = topp.CreateLayer(pop1)
#nest.PrintNetwork()s

#print(nest.GetConnections())

nest.Simulate(1000.0)
mmStatus = nest.GetStatus(multimeter)[0]
Vms = mmStatus["events"]["V_m"]
ts = mmStatus["events"]["times"]

pylab.figure(1)
pylab.plot(ts, Vms)

pylab.figure(2)
G = nx.Graph()
for i in pop1:
	G.add_node(i)
nx.draw(G, with_labels=True)
plt.show()