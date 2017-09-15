import nest
import pylab

#create a neuron called "neuron"
#type integrate-and-fire alpha-shaped post synaptic currents
neuron = nest.Create("iaf_psc_alpha")
nest.GetStatus(neuron) #get all properties of the created neuron
nest.GetStatus(neuron,"I_e") #get the constant background current of neuron

#set the background current to a level that will cause a spike periodically
nest.SetStatus(neuron, {"I_e": 376.0})

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

#simulate for 1000ms
nest.Simulate(1000.0)

#gets the status dictionary of node multimeter and indexes it (hence the 0)
dmm = nest.GetStatus(multimeter)[0]
#print(dmm)

#dmm's dictionary contains entry events, which in turn has a dictionary that contains
# V_m and times
Vms = dmm["events"]["V_m"]
#print(Vms)

ts = dmm["events"]["times"]
#print(ts)

#create a plot with label 1
pylab.figure(1)

pylab.plot(ts, Vms)

dSD = nest.GetStatus(spikedetector, keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]
pylab.figure(2)
pylab.plot(ts,evs,".")
pylab.show()