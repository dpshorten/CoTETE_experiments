Please first setup the [CoTETE.jl](https://github.com/dpshorten/CoTETE.jl) package, as described
in it's documentation, before attempting to run these scripts.

At present, it does not work with the latest version of CoTETE.jl. Please use commit: e68cf12f0f2d5c657b95f620ecdc7af9b411f469. 

Once CoTETE.jl has been setup, you should be able to run the julia script associated with a figure.
eg:

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
replaced with parameters that make the experiments run faster. Running the experiments for some of the figures (eg: figure 6) required an overnight run on a cluster.
