This is the description of the data. If you don't understand some part of it, and it seems critical, let me know. If you don't understand something and it doesn't seem critical, just ignore it. Your goal is to do SOMETHING meaningful about this data, so you definitely don't have to understand every aspect of it.

In this archive you can find 5 subfolders. Each subfolder contains data from one experiment. 

In each experiment a brain of a tadpole was imaged using a high speed camera and a fluorescent scope (see Xu 2011 for details). At the same time, an optic fiber was brought to the tadpole eye (see Khakhalin 2014 for details), and three kinds of stimuli were shown to the tadpole again and again, in a cycle: first a looming stimulus, then a full-field flash, and finally a "scrambled" looming stimulus, in which pixels of a looming stimulus were randomly rearranged (see Jang 2016 for a visualization of all 3 stimuli). The random rearrangement was different for each experiment, but same in each presentation within the experiment.

After each video was recorded, I manually found neurons in each video, and defined a region of interest around each neuron. I then measured fluorescence within each region of interest over time (again, see Xu 2011 for a longer description of this procedure). I then ran a deconvolution algorithm to infer possible spiking from each fluorescence record (see Vogelstein 2010).

In each folder you will find several files:

* readme - practically useless
* xy - contains X and Y coordinates of each neuron within the image of the brain.
* xxxxxxF - fluorescence data (raw data)
* xxxxxxS - inferred spiking data

Both F and S files are comma-separated files with many columns. In F-files, each row describes fluorescence within one frame of the original video. The first column is the presentation number (starting from 1 and up); the 2nd column is time within this presentation (each response was recorded for 4 seconds); all other columns show fluorescence of different neurons, in some arbitrary units. Neurons are given in the same order as in the XY file. S-files are arranged similarly: first a column with a stimulus number, then a column of time stamps within this stimulus, and then many columns for many neurons, except in S-files these columns contain inferred spikes, and not raw fluorescence. As described above, the stimuli always cycled through a sequence of looming-flash-scrambled, so stimulus #1 is always looming, #2 is flash, #3 is scrambled, #4 is looming again, and so on.


Notes:

* The fluorofore was bleaching (becoming less shiny) over time, so each next stimulus is slightly weaker than the previous one. Every now and then though (usually about every 12 stimuli) I would adjust either the light intensity, or the signal amplification up. It means that the signal/noise ratio is different in different trials. At the same time, it is safe to assume that signal/noise ratio was the same within each "triad" of stimuli (looming-flash-scrambled). That means that the signal/noise ratio should be the same within presentations 1-2-3 as well as, say, 13-14-15, but not necessarily between these groups of presentations.
* Different cells were filled with fluorofore to a different extent (and unpredictably so), and also some cells were more in focus, and some were more blurry, which means that signal/noise ratio is also different in different cells.
* The inferred spiking in this dataset is probabilistic, which means that the data I share shows not how strong the spiking was, but how probable it was that there was some spiking in this cell at each frame. It doesn't change the math though; you can (probably?) treat it exactly as you would treat instantaneous spike-frequency.
* There may be artifacts of motion in some trials when the tadpole moved its tail, and so the entire image moved periodically. When it happens, it usually happens in early trials (those closer to the very beginning of the experiment).
* As the experiment went on, the latency of signals (the time between the stimulus and the neural response) frequently shifted a bit. I don't really know why.
* You obviuosly don't have to process all 5 experiments. You may, but you don't have to.
* You don't have to look at all neurons; you can selected best ones if you wish. You may look at all, but you don't have to.
* You don't have to compare responses to different stimuli to each other. You may, and it's one of simpler questions to ask, but you can also ask some other question and use responses indiscriminately. It all depends on your question.
* Same for the position (coordinates) of different neurons. If you want to use this information - use it. If you don't want to use it, just ignore it.


References:

Jang, E. V., Ramirez-Vizcarrondo, C., Aizenman, C. D., & Khakhalin, A. S. (2016). Emergence of selectivity to looming stimuli in a spiking network model of the optic tectum. Frontiers in Neural Circuits, 10.

Khakhalin, A. S., Koren, D., Gu, J., Xu, H., & Aizenman, C. D. (2014). Excitation and inhibition in recurrent networks mediate collision avoidance in Xenopus tadpoles. European Journal of Neuroscience, 40(6), 2948-2962.

Vogelstein, J. T., Packer, A. M., Machado, T. A., Sippy, T., Babadi, B., Yuste, R., & Paninski, L. (2010). Fast nonnegative deconvolution for spike train inference from population calcium imaging. Journal of neurophysiology, 104(6), 3691-3704.

Xu, H., Khakhalin, A. S., Nurmikko, A. V., & Aizenman, C. D. (2011). Visual experience-dependent maturation of correlated neuronal activity patterns in a developing visual system. Journal of Neuroscience, 31(22), 8025-8036.