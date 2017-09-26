import pylab
import nest

#change the default parameters of a spike-timing-dependant plasticity, tau is time constant
nest.SetDefaults("stdp_synapse",{"tau_plus": 15.0})
#create another synapse type, modeled after STDP, with Wmax: 90.0
nest.CopyModel("stdp_synapse","layer1_stdp_synapse",{"Wmax": 90.0})
nest.Create("iaf_psc_alpha",params={"tau_minus": 30.0})

epop1 = nest.Create("iaf_psc_alpha", 10) #create a pop of 10 iaf_psc_alpha neurons
nest.SetStatus(epop1, {"I_e": 376.0}) #set their background current to 376.0
epop2 = nest.Create("iaf_psc_alpha", 10) #create another pop of 10 iaf_psc_alpha neurons
multimeter = nest.Create("multimeter", 10) #create 10 multimeters
nest.SetStatus(multimeter,{"withtime":True, "record_from":["V_m"]})#time, record from membrance voltage

#no connectivity pattern specified, default to "all_to_all" - each neuron in pop 1 is connected to
# all neurons in pop2, resulting in n^2 connections
#nest.Connect(pop1, pop2, syn_spec={"weight":20.0})

#however, the neurons can be connect with "one_to_one", meaning the first neuron in pop1 is connected
#to the first neuron in pop2, resulting in n connections

#nest.Connect(pop1, pop2, "one_to_one", syn_spec={"weight":20.0, "delay":1.0})

d = 1.0 #delay
Je = 2.0 #weight
Ke = 20 # number of incoming random connections from epop1
Ji = -4.0 #weight
Ki = 12 #number of incoming random connections from ipop1

#define connection routine 
conn_dict = {"rule": "fixed_indegree", "indegree":Ke}
syn_dict = {"model": "stdp_synapse", "alpha": 1.0}
nest.Connect(epop1, epop2, conn_dict, syn_dict)