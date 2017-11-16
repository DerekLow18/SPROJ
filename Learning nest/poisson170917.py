import nest
import pylab

#create a neuron called "neuron"
#type integrate-and-fire alpha-shaped post synaptic currents
neuron = nest.Create("iaf_psc_alpha")
neuron2 = nest.Create("iaf_psc_alpha")
nest.GetStatus(neuron) #get all properties of the created neuron
nest.GetStatus(neuron,"I_e") #get the constant background current of neuron

#add poisson process
#ex = excitatory / in = inhibitory
noise_ex = nest.Create("poisson_generator")
noise_in = nest.Create("poisson_generator")
nest.SetStatus(noise_ex, {"rate":80000.0}) #rate in Hz
nest.SetStatus(noise_in,{"rate":15000.0})

#set the background current to a level that will cause a spike periodically
nest.SetStatus(neuron, {"I_e": 0.0})

#set synaptic weight in pA amplitude, connect to neuron
syn_dict_ex = {"weight":1.2}
syn_dict_in = {"weight": -2.0}
nest.Connect(noise_ex, neuron, syn_spec = syn_dict_ex)
#nest.Connect(noise_in, neuron, syn_spec = syn_dict_in)


#create the device used to record the membrane voltage of a neuron over time
multimeter = nest.Create("multimeter")

#withtime to record points in time, V_m to specify membrane voltage
nest.SetStatus(multimeter,{"withtime":True, "record_from":["V_m"]})

#device to record the spiking events produced by a neuron.
#you can use "params" as an alternative to "SetStatus" done in previous steps.
#wtihgid indicates whether or not to record the source id of the event, or the id of the neuron
spikedetector = nest.Create("spike_detector", params={"withgid": True, "withtime":True})

#connect all three nodes. flow of events; neuron spike sends event to spike detector
#multimeter message transmits to neuron to request membrane potential info
nest.Connect(multimeter, neuron)
nest.Connect(neuron,spikedetector)

#create a second neuron with a different background current
nest.SetStatus(neuron2, {"I_e": 370.0})
nest.Connect(multimeter, neuron2)
nest.Connect(neuron2, spikedetector)
syn_neuron_ex = {"weight":100}
nest.Connect(neuron, neuron2, syn_spec = syn_neuron_ex)

#simulate for 1000ms
nest.Simulate(1000.0)

#gets the status dictionary of node multimeter and indexes it (hence the 0)
dmm = nest.GetStatus(multimeter)[0]
#print(dmm)

#dmm's dictionary contains entry events, which in turn has a dictionary that contains
# V_m and times
#Vms = dmm["events"]["V_m"]
#ts = dmm["events"]["times"]

#create a plot with label 1
pylab.figure(1)

#pylab.plot(ts, Vms)
dSD = nest.GetStatus(spikedetector, keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]
pylab.plot(ts,evs,".")

#create a second neuron with a different background current
pylab.figure(2)
Vms1 = dmm["events"]["V_m"][::2]#start at index 0: till the end: every second entry
ts1 = dmm["events"]["times"][::2]
pylab.plot(ts1,Vms1)

Vms2= dmm["events"]["V_m"][1::2]#start at index 1:till the end: every second entry
ts2 = dmm["events"]["times"][1::2]
pylab.plot(ts2,Vms2)

pylab.show()
