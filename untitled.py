import nest
import pylab

pop1 = nest.Create("iaf_psc_alpha", 50)
nest.SetStatus(pop1, {"I_e":376.0})