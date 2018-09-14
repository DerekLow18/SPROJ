# SPROJ
My year-long Senior Project at Bard College

## A Brief Summary
In brief, this is my senior project that I spent a year researching and writing.
For those of you unfamiliar with graduation requirements at Bard, Bard requires that
every senior complete a senior project for their respective majors. In my case, I
elected to do a joint major instead of a double major, largely due to the sheer number
of mandatory classes I still had to complete prior to graduation. This meant that I
could complete one senior project that would fulfill the requirements of both the
Biology and Computer Science departments at Bard College.

What exactly does "completeing the requirements of both departments" mean?
Essentially, I had to complete a project that contained enough computer
science in it to be considered a computer science project, and with enough biology
in it to be considered a biology project. Therefore, I opted to pursure a project
involving neuroscience and machine learning.

The project entailed several steps:
1. Simulation of neurons in a network to generate simulated brain activity
2. Conversion of that activity into a spike-time matrix
3. Analysis of the spike-time matrix with a neural network
4. Analysis of the weights of the neural network to detect for patterns
5. Application of network science concepts to uncover any possible patterns in those weights

Therefore, the project tries to cover a wide range of topics, from machine learning
and neural networks, to simulation and modeling, to analysis via network science.

Currently, while the project passed and I graduated with a joint B.A. in Computer Science and 
Biology, the repository is a mess of different files and data of immense size, as well as
write-ups in LaTex. This README.md marks the beginning of the process of cleaning up the
repository of unnecessary things and possible reorganizing code.

## What are the important sections?

For now, I'll leave you with a little more direction of how to navigate this mess. The
**Experiments** folder is where most of the important code is. All code is in **Python**,
though there is a mix of Python2 and Python3, which largely depended on when the code was
written. At some point during my work, I decided to switch to Python3 to allow easier interfacing
with some modern Python packages, and Python2 code stayed as Python2 code because those programs
contained libraries and packages that were built during the reign of Python2, and were not
compatible with Python3.
