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

edict = {"I_e":200.0, "tau_m": 20.0}
nest.CopyModel("iaf_psc_alpha", "exc_iaf_neuron")
nest.SetDefaults("exc_iaf_neuron", edict)

#it is also possible to create a neuron model with its own default params
idict = {"I_e":300.0}
nest.CopyModel("iaf_psc_alpha", "inh_iaf_neuron", params = idict)
#now there is a neuron called exc_iaf_neuron, which is not an included model, that behaves like
#iaf_psc_alpha, but with a background current of 300.0

#with the above code, all exc_iaf_neurons will have the same params as the iaf_psc_alpha, but
#with a different I_e value
epop1 = nest.Create("exc_iaf_neuron", 100) #population of 100 exc_iaf_neurons with new params
epop2 = nest.Create("exc_iaf_neuron", 100)
ipop1 = nest.Create("inh_iaf_neuron", 30)
ipop2 = nest.Create("inh_iaf_neuron", 30)

#create a population with an inhomogenous set of params
parameter_list=[{"I_e":200.0, "tau_m":20.0},{"I_e":150.0,"tau_m":30.0}]
epop3 = nest.Create("exc_iaf_neuron", 2, parameter_list)


#SETTING PARAMTERS FOR POPULATIONS OF NEURONS
Vth = -55 #set the threshold voltage
Vrest = -70 #set the resting voltage

#loop over the voltage params of epop1 and set the status for each one individually
for neuron in epop1:
	nest.SetStatus([neuron],{"V_m":Vrest+(Vth-Vrest)*numpy.random.rand()}) #random distribution for parameter

#or give setstatus a list of dictionaries to be set to a corresponding number of nodes
dVms = [{"V_m": Vrest+(Vth-Vrest)*numpy.random.rand()} for x in epop1]
nest.SetStatus(epop1, dVms)

# if only one parameter must be randomised, this is an easy way to do it
Vms = Vrest+(Vth-Vrest)*numpy.random.rand(len(epop1))
nest.SetStatus(epop1, "V_m", Vms)
