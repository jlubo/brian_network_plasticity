import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import os.path

def plotMonitorResults(config, neuron_ids, synapse_ids,
                       spike_times, data_stacked,
                       data_dir, interactive=False):
    """
    Plots the results from given Brian state and spike monitors

    Parameters
    ----------
    config:
        configuration parameters in JSON format
    neuron_ids:
        indices of neurons to be targeted for plotting
    synapse_ids:
        indices of synapses to be targeted for plotting
    spike_times: 
        two-dimensional array containing the neuron and time of each spike
    data_stacked: 
        two-dimensional array containing the values of the membrane potential, weights, calcium amount, etc. over time
    syn_state_mon: 
        synapse state monitor
    data_dir:
        the directory in which to store the plots
    interactive [optional]:
        if True, shows the plots in an interactive mode (needed, e.g., for Jupyter Lab)
    """
    ### Plotting ###
    plt.figure(figsize=(18,4))
    figure_fmt = 'png'
    figure_dpi = 200
    delta_t = config["simulation"]["dt"] # timestep, in units of ms
    t_max = config["simulation"]["runtime"] # total duration, in units of s
    theta_p = config["synapses"]["syn_exc_calcium_plasticity"]["theta_p"] # LTP threshold
    theta_d = config["synapses"]["syn_exc_calcium_plasticity"]["theta_d"] # LTD threshold
    h_0 = config["synapses"]["syn_exc_calcium_plasticity"]["h_0"] # baseline of the synaptic weight, in units of mV

    # Time range to plot
    xlim_0 = 0
    xlim_1 = int(t_max/delta_t*1000)
    #xlim_auto = False
    tb = np.arange(xlim_0, xlim_1, 1)
    spike_mask = np.logical_and(spike_times[:,0]/delta_t >= xlim_0, spike_times[:,0]/delta_t <= xlim_1) # selects spikes in the considered time window

    # Spike raster plot
    plt.title('Spike raster')
    plt.plot(spike_times[:,0][spike_mask], spike_times[:,1][spike_mask], '.', c="purple") # alpha=0.4
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    ax = plt.gca()
    #ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('% 1.0f'))
    #plt.ylim(-0.5,1.5)
    #plt.ylim(-0.5,3.5)
    plt.savefig(os.path.join(data_dir, f"spike_raster.{figure_fmt}"), dpi=figure_dpi)
    if interactive:
        plt.show()
    plt.close()

    #plt.subplot(133)
    for i in range(len(neuron_ids)):
        # Neuronal membrane voltage
        plt.title(f'Neuron {neuron_ids[i]}')
        V = data_stacked[:,1+i*2]
        plt.plot(tb*delta_t, V[tb], c="#ff0000")
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane potential (mV)');
        plt.savefig(os.path.join(data_dir, f"voltage_neuron_{neuron_ids[i]}.{figure_fmt}"), dpi=figure_dpi)
        if interactive:
            plt.show()
        plt.close()

    for i in range(len(synapse_ids)):
        # Synaptic calcium amount and protein amount of neuron 0
        plt.figure(figsize=(12,4))
        Ca = data_stacked[:,1+len(neuron_ids)*2+i*4+2]
        p = data_stacked[:,1+len(neuron_ids)*2+i*4+3]
        plt.title(f'Synapse {synapse_ids[i]}')
        plt.plot(tb*delta_t, Ca[tb], c="#c8c896")
        plt.plot(tb*delta_t, p[tb], c="#008000")
        plt.xlabel('Time (ms)')
        plt.ylabel('Calcium or protein amount');
        for ts in spike_times[:,0][spike_mask]:
            plt.axvline(ts, ls='dotted', c="purple", alpha=0.4)
        plt.axhline(theta_p, ls='dotted', c='#969664')
        plt.axhline(theta_d, ls='dotted', c='#969696')
        plt.savefig(os.path.join(data_dir, f"calcium_synapse_{synapse_ids[i]}_and_protein_neuron_0.{figure_fmt}"), dpi=figure_dpi)
        if interactive:
            plt.show()
        plt.close()

        # Early-/late-phase synaptic weight
        plt.figure(figsize=(12,4))
        h = data_stacked[:,1+len(neuron_ids)*2+i*4]/h_0*100
        z = (data_stacked[:,1+len(neuron_ids)*2+i*4+1]+1)*100
        plt.plot(tb*delta_t, h[tb], c="#800000")
        plt.plot(tb*delta_t, z[tb], c="#1f77b4")
        plt.xlabel('Time (ms)')
        plt.ylabel('Synaptic weight (%)')
        plt.savefig(os.path.join(data_dir, f"weights_synapse_{synapse_ids[i]}_and_protein_neuron_0.{figure_fmt}"), dpi=figure_dpi)
        if interactive:
            plt.show()
        plt.close()
