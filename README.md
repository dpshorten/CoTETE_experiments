This repository contains code for reproducing the figures in
[Estimating Transfer Entropy in Continuous Time Between Neural Spike Trains or Other Event-Based Data](https://doi.org/10.1101/2020.06.16.154377).

Please first setup the [CoTETE.jl](https://github.com/dpshorten/CoTETE.jl) package, as described
in it's documentation, before attempting to run these scripts.

Once CoTETE.jl has been setup, you should be able to run the julia script associated with a figure. These scripts
assume that there is a folder named "run outputs", where the results can be saved in an HDF5 file.

```console
david@home:~$ mkdir run_outputs
david@home:~$ julia figure_2_experiment.jl
```

The saved results can then be used by the plotting script.

```console
david@home:~$ python3 figure_2_plotting.py
```
