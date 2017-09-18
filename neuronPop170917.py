import nest
import pylab

#Creating a population of nodes
#ndict will be the dictionary containing parameters used to create the neurons in population

ndict = {"I_e":200.0, "tau_m": 20.0}
#now we create a list of ids of all neurons (100 neurons) with ndict params:
neuronpop = nest.Create("iaf_psc_alpha",100, params = ndict)

#instad of using the (params =) argument, we can set the default using:
nest.SetDefaults("iaf_psc_alpha", ndict)
#now, all neurons with iaf_psc_alpha designation will also include the new defaults in ndict
neuronpop1 = nest.Create("iaf_psc_alpha",100)
neuronpop2 = nest.Create("iaf_psc_alpha",100)
neuronpop3 = nest.Create("iaf_psc_alpha",100)
#the above three neuron groups have the same settings as the original neuronpop group
for neuron in neuronpop1:
	print(neuron)
#it is also possible to create a neuron model with its own default params
#idict = {"I_e":300.0}
#nest.CopyModel("iaf_psc_alpha", "exc_iaf_neuron", params = idict)
#with the above code, all exc_iaf_neurons will have the same params as the iaf_psc_alpha, but
#with a different I_e value

#epop1 = nest.Create("exc_iaf_neurons", 100) #population of 100 exc_iaf_neurons with new params

