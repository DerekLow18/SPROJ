import pylab
import nest
import nest.raster_plot as raster

epop1 = nest.Create("iaf_psc_alpha", 10) #create a pop of 10 iaf_psc_alpha neurons
#nest.SetStatus(epop1, {"I_e": 376.0}) #set their background current to 376.0
ipop1 = nest.Create("iaf_psc_alpha", 10) #create another pop of 10 iaf_psc_alpha neurons
spikes = nest.Create("spike_detector",1)
noise = nest.Create("poisson_generator",1,{'rate':poisson_rate})
#multimeter = nest.Create("multimeter", 10) #create 10 multimeters
#nest.SetStatus(multimeter,{"withtime":True, "record_from":["V_m"]})#time, record from membrance voltage


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

#indegree allows us to create n random connections to post from randomly selected neurons to source
#population pre.
conn_dict_ex = {"rule": "fixed_indegree", "indegree":Ke}
conn_dict_in = {"rule": "fixed_indegree", "indegree":Ki}
syn_dict_ex = {"delay": d, "weight": Je}
syn_dict_in = {"delay": d, "weight": Ji}
nest.Connect(epop1, ipop1, conn_dict_ex, syn_dict_ex) #connects 20 epop1 to ipop1
nest.Connect(ipop1, epop1, conn_dict_in, syn_dict_in) #connects 12 ipop1 to epop1
nest.Connect(epop1, spikes)
nest.Connect(ipop1, spikes)

#finally, connect multimeters with pop2
#nest.Connect(multimeter, ipop1)
nest.Simulate(1000.0)

'''
pylab.figure(1)
Vms1 = dmm["events"]["V_m"]
Ts1 = dmm ["events"]["times"]
pylab.plot(Ts1, Vms1)
'''

#print (nest.GetConnections())
#nest.PrintNetwork()
plot = nest.raster_plot.from_device(spikes, hist=True)
pylab.show()