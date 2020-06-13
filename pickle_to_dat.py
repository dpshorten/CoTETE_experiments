import pickle

for k in range(5):
    print(k)
    
    pickled_events_file = open("canonical_example_pickles/train_" + str(k + 1) + ".pkl", 'rb')
    event_train_x = pickle.load(pickled_events_file)
    event_train_y = pickle.load(pickled_events_file)

    dat_x = open("canonical_example_dats/train_x_" + str(k + 1) + ".dat", 'w')
    dat_y = open("canonical_example_dats/train_y_" + str(k + 1) + ".dat", 'w')

    for i in range(len(event_train_x)):
        dat_x.write(str(event_train_x[i]) + "\n")
        if i % int(1e7) == 0:
            print("x", i)

    for i in range(len(event_train_y)):
        dat_y.write(str(event_train_y[i]) + "\n")
        if i % int(1e7) == 0:
            print("y", i)

    dat_x.close()
    dat_y.close()
