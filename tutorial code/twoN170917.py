import nest
import pylab

#create a neuron called "neuron"
#type integrate-and-fire alpha-shaped post synaptic currents
neuron1 = nest.Create("iaf_psc_alpha")
#set the background current to a level that will spike periodically
nest.SetStatus(neuron1, {"I_e": 376.0})
#create a second neuron. We will not set any background current, we will connect to neuron1
neuron2=nest.Create("iaf_psc_alpha")
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
nest.Connect(multimeter, neuron2)
#connect neuon1 and neuron2, set the synaptic weight to 20 amplitude
nest.Connect(neuron1, neuron2, syn_spec = {"weight":20.0})

#simulate for 1000ms
nest.Simulate(1000.0)

#gets the status dictionary of node multimeter and indexes it (hence the 0)
dmm = nest.GetStatus(multimeter)[0]
#create a plot with label 1
pylab.figure(1)

#pylab.plot(ts, Vms)
dSD = nest.GetStatus(spikedetector, keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]
pylab.plot(ts,evs,".")

pylab.figure(2)
Vms1 = dmm["events"]["V_m"][::2]#start at index 0: till the end: every second entry
ts1 = dmm["events"]["times"][::2]
pylab.plot(ts1,Vms1)

pylab.show()
