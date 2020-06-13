import matplotlib.pyplot as plt
import numpy as np

events = np.loadtxt("output_stg_full13_minus_lp-py/lp_1.dat")
#print(events.shape)
events = events[:1000]
events2 = np.loadtxt("output_stg_full13_minus_lp-py/py_1.dat")
events2 = events2[:1000]
#events3 = np.loadtxt("output_stg_full11_surrogates/lp_1_surrogate_2.dat")
#events3 = events3[:1000]
#print(events2.shape)

events = np.vstack((events, events2))

plt.eventplot(events[:, 700:1000])
#plt.eventplot(events2[:100])
plt.show()
