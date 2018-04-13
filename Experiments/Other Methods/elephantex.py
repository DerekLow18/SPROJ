import elephant
import matplotlib.pyplot as plt
import quantities as pq
binned_st1 = elephant.conversion.BinnedSpikeTrain(
    elephant.spike_train_generation.homogeneous_poisson_process(
        10. * pq.Hz, t_start=0 * pq.ms, t_stop=5000 * pq.ms),
    binsize=5. * pq.ms)
binned_st2 = elephant.conversion.BinnedSpikeTrain(
    elephant.spike_train_generation.homogeneous_poisson_process(
        10. * pq.Hz, t_start=0 * pq.ms, t_stop=5000 * pq.ms),
    binsize=5. * pq.ms)
cc_hist = elephant.spike_train_correlation.cross_correlation_histogram(
    binned_st1, binned_st2, window=[-30,30],
    border_correction=False,
    binary=False, kernel=None, method='memory')
print(cc_hist[0].times.magnitude)
print(cc_hist[0][:, 0].magnitude)
x1 = binned_st1.to_array()
x2 = binned_st2.to_array()
x11 = x1[0]
x21 = x2[0]
y1 = [timestep for timestep in range(len(x11))]
y2 = [timestep for timestep in range(len(x21))]
fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(y1,x11)
ax2.plot(y2,x21)
fig.savefig("../../Main Writing/Figures/HPProcess.svg",format = 'svg')
plt.show()

plt.bar(
    left=cc_hist[0].times.magnitude,
    height=cc_hist[0][:, 0].magnitude,
    width=cc_hist[0].sampling_period.magnitude)
plt.xlabel('time (' + str(cc_hist[0].times.units) + ')')
plt.ylabel('cross-correlation histogram')
plt.axis('tight')
plt.savefig("../../Main Writing/Figures/HPxc.svg",format = 'svg')
plt.show()
