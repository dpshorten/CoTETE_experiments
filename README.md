This repository contains code for reproducing the figures in
[Estimating Transfer Entropy in Continuous Time Between Neural Spike Trains or Other Event-Based Data](https://doi.org/10.1101/2020.06.16.154377).

Please first setup the [CoTETE.jl](https://github.com/dpshorten/CoTETE.jl) package, as described
in it's documentation, before attempting to run these scripts.

Once CoTETE.jl has been setup, you should be able to run the julia script associated with a figure. 

```console
david@home:~$ julia figure_2_experiment.jl
```

The saved results can then be used by the plotting script.

```console
david@home:~$ python3 figure_2_plotting.py
```

For the figure 4 experiments you will first have to run `figure_4_data_generator.py`
and `figure_4_data_converter.py` to create the example data to work with.

You can download the data for figure 8
[here](https://unisyd-my.sharepoint.com/:u:/r/personal/david_shorten_sydney_edu_au/Documents/stg_spike_files.zip?csf=1&web=1&e=2XkX6n).

Note that many of the parameters used for the figures in the paper have been commented out and
replaced with parameter that make the experiments run faster. Running the experiments for some of the figures (eg: figure 6) required an overnight run on a cluster.
