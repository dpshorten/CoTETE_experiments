import random
import math
import pickle
import numpy as np

RATE_Y = 1.0
NUM_Y_eventS = 3e6
RATE_X_MAX = 10
NUM_FILES = 1

for file_num in range(NUM_FILES):

    event_train_y = []
    event_train_x = []

    event_train_x.append(0)

    event_train_y = np.random.uniform(0, int(NUM_Y_eventS / RATE_Y), int(NUM_Y_eventS))
    event_train_y.sort()

    most_recent_y_index = 0
    previous_x_candidate = 0
    while most_recent_y_index < (len(event_train_y) - 1):

        if most_recent_y_index % int(1e5) == 0:
            print(file_num + 1, int(most_recent_y_index/1e5))

        this_x_candidate = previous_x_candidate + random.expovariate(RATE_X_MAX)

        while most_recent_y_index < (len(event_train_y) - 1) and this_x_candidate > event_train_y[most_recent_y_index + 1]:
            most_recent_y_index += 1

        delta_t = this_x_candidate - event_train_y[most_recent_y_index]

        rate = 0

        if delta_t > 1:
            rate = 0.5
        else:
            rate = 0.5 + 5.0 * math.exp(-50 * (delta_t - 0.5)**2) - 5.0 * math.exp(-50 * (0.5)**2)
        if random.random() < rate/float(RATE_X_MAX):
            event_train_x.append(this_x_candidate)
        previous_x_candidate = this_x_candidate

    f = open("train_" + str(file_num + 1) + ".pkl", 'wb')
    pickle.dump(event_train_x, f)
    pickle.dump(event_train_y, f)
    f.close()
