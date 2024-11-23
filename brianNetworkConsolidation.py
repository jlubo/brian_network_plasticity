#!/bin/python3

# Brian simulation of memory consolidation in recurrent spiking neural networks, consisting of leaky integrate-and-fire neurons that are
# connected via current-based, plastic synapses (early phase based on calcium dynamics and late phase based on synaptic tagging and capture).
# Reproduces the results of Luboeinski and Tetzlaff, Commun. Biol., 2021 (https://doi.org/10.1038/s42003-021-01778-y).

# (c) Jannik Luboeinski 2022-2024
# Contact: mail[at]jlubo.net

import brian2 as b
from brian2.units import msecond, second, mvolt, namp, ncoulomb, Mohm, pfarad, hertz, farad
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import json
import os
import time
from datetime import datetime
import re
import argparse
from plotMonitorResults import plotMonitorResults
b.set_device('cpp_standalone')

###############################################################################
# getTimestamp
# Returns a previously determined timestamp, or the timestamp of the current point in time
# - refresh [optional]: if True, forcibly retrieves a new timestamp; else, only returns a new timestamp if no previous one is known
# - return: timestamp in the format YY-MM-DD_HH-MM-SS
# TODO outsourcing
def getTimestamp(refresh = False):
	global timestamp_var # make this variable static
		
	try:
		if timestamp_var and refresh == True:
			timestamp_var = datetime.now() # forcibly refresh timestamp
	except NameError:
		timestamp_var = datetime.now() # get timestamp for the first time
		
	return timestamp_var.strftime("%y-%m-%d_%H-%M-%S")
	
###############################################################################
# getFormattedTime
# Returns a string indicating the hours, minutes, and seconds of a time given in seconds
# - time_el: time in seconds
# - return: formatted time string
# TODO outsourcing
def getFormattedTime(time_el):
	if time_el < 1:
		time_el_str = "<1 s"
	elif time_el < 60:
		time_el_str = str(time_el) + " s"
	elif time_el < 3600:
		time_el_str = str(time_el // 60) + " m " + str(time_el % 60) + " s"
	else:
		time_el_str = str(time_el // 3600) + " h " + str((time_el % 3600) // 60) + " m " + str((time_el % 3600) % 60) + " s"
	
	return time_el_str		
	
###############################################################################
# getDataPath
# Consumes a general description for the simulation and a file description, and returns 
# a path to a timestamped file in the output directory;  if no file description is provided, 
# returns the path to the output directory
# - sim_description [optional]: general description for the simulation (only has to be set once)
# - file [optional]: specific name and extension for the currently considered file
# - refresh [optional]: if True, enforces the retrieval of a new timestamp
# - return: path to a file in the output directory
# TODO outsourcing
def getDataPath(sim_description = None, file = "", refresh = False):
	global sim_description_var # make this variable static
	
	timestamp = getTimestamp(refresh)
	out_path = "data_" + timestamp 
	
	if sim_description is None:
		try:
			sim_description = sim_description_var # use previously defined default description
		except NameError:
			sim_description = ""
	
	if sim_description != "":
		out_path = out_path + " " + sim_description
		sim_description_var = sim_description # set new default description

	if file == "":
		return out_path
		
	return os.path.join(out_path, timestamp + "_" + file)

###############################################################################
# initLog
# Initializes the global log file
# - desc: description in the data path
def initLog(desc):
	global logf

	logf = open(getDataPath(desc, "log.txt"), "w")
	
###############################################################################
# closeLog
# Close the global log file
def closeLog():
	global logf

	logf.close()

###############################################################################
# writeLog
# Writes string to the global log file 'logf' and prints it to the console
# - ostrs: the string(s) to be written/printed
# TODO outsourcing
def writeLog(*ostrs):

	for i in range(len(ostrs)):
		ostr = str(ostrs[i])
		ostr = re.sub(r'\x1b\[[0-9]*m', '', ostr) # remove console formatting
		if i == 0:
			logf.write(ostr)
		else:
			logf.write(" " + ostr)
	logf.write("\n")
	
	print(*ostrs)


###############################################################################
# completeProt
# Takes a, possibly incomplete, stimulation protocol dictionary and returns a complete
# protocol after adding keys that are missing. This allows to leave out unnecessary keys
# when defining protocols in the JSON config file.
# - prot: stimulation protocol, possibly incomplete
# - return: complete stimulation protocol
def completeProt(prot):
	
	if prot.get("scheme") is None:
		prot["scheme"] = ""

	if prot.get("time_start") is None:
		prot["time_start"] = 0

	#if prot.get("duration") is None:
	#	prot["duration"] = 0

	if prot.get("freq") is None:
		prot["freq"] = 0

	if prot.get("N_stim") is None:
		prot["N_stim"] = 0

	if prot.get("I_0") is None:
		prot["I_0"] = 0

	if prot.get("sigma_WN") is None:
		prot["sigma_WN"] = 0

	if prot.get("explicit_input") is None \
	  or prot["explicit_input"].get("receivers") is None \
	  or prot["explicit_input"].get("stim_times") is None:
		prot["explicit_input"] = { "receivers": [ ], "stim_times": [ ] }

	return prot


###############################################################################
# protFormatted
# Takes a complete stimulation protocol dictionary and creates a formatted string for
# human-readable output.
# - prot: stimulation protocol
# - return: formatted string with information about the protocol
def protFormatted(prot):
	if not prot['scheme']:
		fmstr = "none"
	elif prot['scheme'] == "EXPLICIT":
		fmstr = f"{prot['scheme']}, pulses at {prot['explicit_input']['stim_times']} ms, to neurons {prot['explicit_input']['receivers']}"
	elif prot['scheme'] == "STET":
		fmstr = f"{prot['scheme']} (Poisson) to neurons {prot['explicit_input']['receivers']}, starting at {prot['time_start']} s"
	elif prot['I_0'] != 0:
		fmstr = f"{prot['scheme']} (OU), I_0 = {prot['I_0']} nA, sigma_WN = {prot['sigma_WN']} nA s^1/2"
	#elif prot['duration'] != 0:
	#	fmstr = f"{prot['scheme']} (OU), {prot['N_stim']} input neurons at {prot['freq']} Hz, starting at {prot['time_start']} s, lasting for {prot['duration']} s"
	else:
		fmstr = f"{prot['scheme']} (OU), {prot['N_stim']} input neurons at {prot['freq']} Hz, starting at {prot['time_start']} s"
	
	return fmstr


#####################################
# run
# Runs simulation of a recurrent neural network with consolidation dynamics
# - config: configuration of model and simulation parameters (as a dictionary from JSON format)
# - neuron_observables: list of the neuronal observables to be monitored
# - synapse_observables: list of the synaptic observables to be monitored
# - interactive [optional]: if True, shows the plots in an interactive mode (needed, e.g., for Jupyter Lab)
def run(config, 
		neuron_observables=['V', 'p'],
		synapse_observables=['h', 'z', 'Ca'],
		interactive=False):
	
	#return spike_mon, neur_state_mon, syn_state_mon
	# initialize
	# Sets the parameter values according to provided dictionary
	# - config: dictionary containing configuration data
	#def initialize(config):
		#...
		
	# get_model
	# Defining the mathematical equations describing the network model
	# - req: string specifying which model is wanted
	# - return: the requested equation (or set of equations)
	def get_model(req):
		
		# neuron dynamics: LIF membrane potential and amount of plasticity-related proteins (for excitatory neurons)
		if req == "neuron_full":
			return '''
			dV/dt = ( -(V - V_rev) + V_psp + R_mem*I_bg + V_stim_0 + V_stim_1 ) / tau_mem: volt (unless refractory)
			dI_bg/dt = ( -I_bg + I_0 + sigma_wn*xi_bg ) / tau_OU : amp
			V_stim_0 : volt
			V_stim_1 : volt
			sum_h_diff : volt
			dp/dt = ( -p + alpha_c*0.5*(1+sign(sum_h_diff - theta_pro_c)) ) / tau_pro_c : 1
			dV_psp/dt = -V_psp / tau_syn : volt
			'''
		# neuron dynamics: LIF membrane potential (simplified, for inhibitory neurons) 
		if req == "neuron_simple":
			return '''
			dV/dt = ( -(V - V_rev) + V_psp + R_mem*I_bg ) / tau_mem: volt (unless refractory)
			dI_bg/dt = ( -I_bg + I_0 + sigma_wn*xi_bg ) / tau_OU : amp
			dV_psp/dt = -V_psp / tau_syn : volt
			'''
		# plastic synapse dynamics: calcium amount, early-phase weight, tag, late-phase weight
		elif req == "plastic_synapse":
			return '''
			dh/dt = ( 0.1*(h_0 - h) + gamma_p*(10*mvolt - h)*0.5*(1+sign(Ca - theta_p)) - gamma_d*h*0.5*(1+sign(Ca - theta_d)) + sqrt(tau_h * (0.5*(1+sign(Ca - theta_p)) + 0.5*(1+sign(Ca - theta_d)))) * sigma_pl * xi_pl ) / tau_h : volt (clock-driven)
			dCa/dt = -Ca/tau_Ca : 1 (clock-driven)
			sum_h_diff_post = abs(h - h_0) : volt (summed)
			dz/dt = ( alpha_c*p_post *(1-z)*0.5*(1+sign((h - h_0) - theta_tag_p)) - alpha_c*p_post *(z + 0.5) * 0.5*(1+sign((h_0 - h) - theta_tag_d)) ) / tau_z : 1 (clock-driven)
			w = h + z*h_0 : volt
			'''
		# plastic synapse dynamics: calcium amount, early-phase weight
		elif req == "plastic_early_only_synapse":
			return '''
			dh/dt = ( 0.1*(h_0 - h) + gamma_p*(10*mvolt - h)*0.5*(1+sign(Ca - theta_p)) - gamma_d*h*0.5*(1+sign(Ca - theta_d)) + sqrt(tau_h * (0.5*(1+sign(Ca - theta_p)) + 0.5*(1+sign(Ca - theta_d)))) * sigma_pl * xi_pl ) / tau_h : volt (clock-driven)
			dCa/dt = -Ca/tau_Ca : 1 (clock-driven)
			sum_h_diff_post = 0*volt : volt (summed)
			dz/dt = 0 / tau_z : 1 (clock-driven)
			w = h : volt
			'''
		# plastic synapse dynamics: postsynaptic effects following presynaptic spike
		elif req == "plastic_synapse_pre":
			return {
			'pre_voltage': 'V_psp_post += w',
			'pre_calcium': 'Ca += Ca_pre'
			}
		# plastic synapse dynamics: postsynaptic effects following postsynaptic spike
		elif req == "plastic_synapse_post":
			return 'Ca += Ca_post'
		# non-plastic synapse dynamics: weight
		elif req == "static_synapse":
			return '''
			w : volt
			'''
		# non-plastic synapse dynamics: postsynaptic potential following presynaptic spike 
		elif req == "static_synapse_pre":
			return {
			'pre_voltage': 'V_psp_post += w',
			}
		# stimulus input #0
		elif req == "stim_input_0":
			return '''
			V_stim_0_post = V_stim_pre : volt (summed)
			'''
		# stimulus input #1
		elif req == "stim_input_1":
			return '''
			V_stim_1_post = V_stim_pre : volt (summed)
			'''
		# unknown specifier
		raise ValueError(f"Unknown model specifier: {req}")

	# get_ou_stimulus
	# Defining the mathematical description of an Ornstein-Uhlenbeck stimulus input
	# - req: string specifying which paradigm is wanted
	# - return: the requested equation or set of equations; or `None` if unknown paradigm has been specified
	def get_ou_stimulus(req):
		
		# modeling putative population input for the full duration
		if req == "FULL":
			return '''
			dV/dt = ( -V + (N_stim*f_stim + sqrt(N_stim*f_stim)*xi_stim) * 1*second*h_0 ) / tau_OU : volt
			V_stim = V : volt
			f_stim : hertz
			N_stim : 1
			'''
		# modeling putative population input with pulse triplet
		elif req == "TRIPLET":
			return '''
			dV/dt = ( -V + ((N_stim*f_stim + sqrt(N_stim*f_stim)*xi_stim) * 1*second*h_0 ) ) / tau_OU : volt
			V_stim = ( int(t>=t_start) * int(t<=t_start + 0.1*second)
					  + int(t>=t_start + 0.5*second) * int(t<=t_start + 0.6*second)
					  + int(t>=t_start + 1.0*second) * int(t<=t_start + 1.1*second) ) * V : volt
			f_stim : hertz
			N_stim : 1
			t_start : second
			'''
		# modeling putative population input with one pulse of 0.1 s duration
		elif req == "ONEPULSE":
			return '''
			dV/dt = ( -V + ((N_stim*f_stim + sqrt(N_stim*f_stim)*xi_stim) * 1*second*h_0 ) ) / tau_OU : volt
			V_stim = ( int(t>=t_start) * int(t<=t_start + 0.1*second) ) * V : volt
			f_stim : hertz
			N_stim : 1
			t_start : second
			'''
		return None
	
	# get_explicit_stimulus
	# Check and retrieve the specification of an explicit stimulus input (explicitly specified pulses for
	# stimulation to defined receiving neurons, implemented below via SpikeGeneratorGroup)
	# - prot: stimulation protocol dictionary (including the key "explicit_input")
	# - return: dictionary of receiving neurons and stimulation pulse times if explicit protocol is given, `None` if not
	def get_explicit_stimulus(prot):
		# Check scheme
		if prot["scheme"] != "EXPLICIT":
			return None
		# Check abundance of stimulation
		if len(prot["explicit_input"]["receivers"]) <= 0 or \
		   len(prot["explicit_input"]["stim_times"]) <= 0:
			return None
		return prot["explicit_input"]

	#####################################
	# make certain parameter values externally accessible
	
	#global theta_p, theta_d, theta_pro_c, theta_tag_p, theta_tag_d, h_0, delta_t

	#####################################
	# set the parameter values
	
	#initialize(config)
	
	minute = 60*second
	hour = 60*minute

	# simulation parameters
	delta_t = config["simulation"]["dt"]*msecond # timestep for detailed computation
	b.defaultclock.dt = delta_t
	s_desc = config['simulation']['short_description'] # short description of the simulation
	t_max = config["simulation"]["runtime"]*second # biological duration of the simulation
	#output_period = config['simulation']['output_period'] # sampling size in timesteps (every "output_period-th" timestep, data will be recorded for plotting)
	neuron_ids = np.array(config['simulation']['sample_neuron_ids']) # identifier(s) of the neuron(s) to be probed ([]: none)
	synapse_ids = np.array(config['simulation']['sample_synapse_ids']) # identifier(s) of the synapse(s) to be probed ([]: none)
	learn_prot = completeProt(config["simulation"]["learn_protocol"]) # protocol for learning stimulation as a dictionary with the keys "time_start" (starting time), "scheme" (scheme of pulses), and "freq" (stimulation frequency)
	recall_prot = completeProt(config["simulation"]["recall_protocol"]) # protocol for recall stimulation as a dictionary with the keys "time_start" (starting time), "scheme" (scheme of pulses), and "freq" (stimulation frequency)
	bg_prot = completeProt(config["simulation"]["bg_protocol"]) # protocol for background input as a dictionary with the keys "time_start" (starting time), "scheme" (scheme of pulses), "I_0" (mean), and "sigma_WN" (standard deviation)
	N_CA = config["populations"]["N_CA"]
	p_r = config["populations"]["p_r"]

	# network parameters
	N_exc = config["populations"]["N_exc"]
	N_inh = config["populations"]["N_inh"]
	N_tot = N_exc + N_inh
	tau_syn = config["synapses"]["tau_syn"]*msecond
	t_syn_delay = config["synapses"]["t_ax_delay"]*msecond
	h_0 = config["synapses"]["syn_exc_calcium_plasticity"]["h_0"]*mvolt
	w_ei = config["populations"]["w_ei"]*h_0
	w_ie = config["populations"]["w_ie"]*h_0
	w_ii = config["populations"]["w_ii"]*h_0
	p_c = config["populations"]["p_c"]
	"""
	if config["populations"]["conn_file"]: # if a connections file is specified, load the connectivity matrix from that file
		conn_matrix = np.loadtxt(config["populations"]["conn_file"]).transpose()
	else: # there is no pre-defined connectivity matrix -> generate one
		rng = np.random.default_rng() # random number generator # TODO refactor with Brian functions? (see below)
		conn_matrix = rng.random((N_tot, N_tot)) <= p_c # two-dim. array of booleans indicating the existence of any incoming connection
		conn_matrix[np.identity(N_tot, dtype=bool)] = 0 # remove self-couplings
	"""

	# plasticity parameters
	t_Ca_delay = config["synapses"]["syn_exc_calcium_plasticity"]["t_Ca_delay"]*msecond
	Ca_pre = config["synapses"]["syn_exc_calcium_plasticity"]["Ca_pre"]
	Ca_post = config["synapses"]["syn_exc_calcium_plasticity"]["Ca_post"]
	tau_Ca = 0.0488*second
	tau_h = 688.4*second
	tau_pro_c = 1*hour
	tau_z = 1*hour
	z_max = 1
	gamma_p = 1645.6
	gamma_d = 313.1
	theta_p = config["synapses"]["syn_exc_calcium_plasticity"]["theta_p"]
	theta_d = config["synapses"]["syn_exc_calcium_plasticity"]["theta_d"]
	nu_th_LTP = 0*hertz
	nu_th_LTD = 0*hertz
	sigma_pl = config["synapses"]["syn_exc_calcium_plasticity"]["sigma_pl"]*mvolt #2.90436*mvolt
	alpha_c = 1
	theta_pro_c = config["synapses"]["syn_exc_calcium_plasticity"]["theta_pro"]*mvolt
	theta_tag_p = config["synapses"]["syn_exc_calcium_plasticity"]["theta_tag"]*mvolt
	theta_tag_d = config["synapses"]["syn_exc_calcium_plasticity"]["theta_tag"]*mvolt

	# neuron parameters
	R_mem = config["neuron"]["R_leak"]*Mohm
	C_mem = config["neuron"]["C_mem"]*farad
	tau_mem = R_mem * C_mem # membrane time constant
	V_rev = config["neuron"]["V_rev"]*mvolt
	t_ref = config["neuron"]["t_ref"]*msecond + delta_t # refractory period, add one time step to comply with other implementations
	V_reset = config["neuron"]["V_reset"]*mvolt
	V_th = config["neuron"]["V_th"]*mvolt
	V_spike = 35*mvolt
	tau_OU = tau_syn
	w_spike = 100*(V_th-V_reset)*np.exp(delta_t/tau_mem) # value that is sufficently large to cause spike in postsyn. neuron
	
	#parameter_nmspc = {'delta_t': 2. * usecond, 'delta_t_ff': 60. * second, 's_desc': 'Small net', 't_max': 3. * second, 'f_stim': 1, 'I_0': 0, 'sigma_WN': 0, 'N_CA': 1, 'p_r': 0.5, 'N_stim': 1000, 'N_exc': 4, 'N_inh': 0, 'N_tot': 4, 'tau_syn': 5. * msecond, 't_syn_delay': 3. * msecond, 'h_0': 4.20075 * mvolt, 'w_ee': 4.20075 * mvolt, 'w_ei': 8.4015 * mvolt, 'w_ie': 16.803 * mvolt, 'w_ii': 16.803 * mvolt, 'p_c': -1, 't_Ca_delay': 18.8 * msecond, 'Ca_pre': 0.6, 'Ca_post': 0.1655, 'tau_Ca': 48.8 * msecond, 'tau_h': 0.6884 * ksecond, 'tau_pro_c': 3.6 * ksecond, 'tau_z': 3.6 * ksecond, 'z_max': 1, 'gamma_p': 1645.6, 'gamma_d': 313.1, 'theta_p': 3.0, 'theta_d': 1.2, 'nu_th_LTP': 0. * hertz, 'nu_th_LTD': 0. * hertz, 'sigma_pl': 2.90436 * mvolt, 'alpha_c': 1, 'theta_pro_c': 2.10037 * mvolt, 'theta_tag_p': 0.840149 * mvolt, 'theta_tag_d': 0.840149 * mvolt, 'R_mem': 10. * Mohm, 'C_mem': 1. * nfarad, 'tau_mem': 10. * msecond, 'V_rev': -65. * mvolt, 't_ref': 2. * msecond, 'V_reset': -70. * mvolt, 'V_th': -55. * mvolt, 'V_spike': 35. * mvolt, 'tau_OU': 5. * msecond, 'I_0': 0. * amp, 'sigma_wn': 0. * second ** 0.5 * amp, 'w_spike': 1.50030003 * volt}
		

	#####################################
	# create output directory, save code and config, and open log file
	
	out_path = getDataPath(s_desc, refresh=True)
	if not os.path.isdir(out_path): # if the directory does not exist yet
		os.mkdir(out_path)
	os.system("cp -r *.ipynb  \"" + out_path + "\"") # archive the interactive Python code
	os.system("cp -r *.py  \"" + out_path + "\"") # archive the Python code
	#os.system("cp -r mechanisms/ \"" + out_path + "\"") # archive the NMODL mechanism code
	#os.system("cp -r custom/ \"" + out_path + "\"") # archive the C++ mechanism code
	json.dump(config, open(getDataPath(file="config.json"), "w"), indent="\t")
	global logf # global handle to the log file (to need less code for output commands)
	logf = open(getDataPath(file="log.txt"), "w")
	
	#####################################
	# output of key parameters
	writeLog(f"\x1b[31mBrian network simulation {getTimestamp()} (Brian version: {b.__version__})\n" +
	          "|\n"
	         f"\x1b[35mSimulated timespan:\x1b[37m {config['simulation']['runtime']} s\n" +
	         f"\x1b[35mPopulation parameters:\x1b[37m N_exc = {config['populations']['N_exc']}, N_inh = {config['populations']['N_inh']}\n" +
	         f"\x1b[35mLearning protocol:\x1b[37m {protFormatted(learn_prot)}\n" +
	         f"\x1b[35mRecall protocol:\x1b[37m {protFormatted(recall_prot)}\n" +
	         f"\x1b[35mBackground input:\x1b[37m {protFormatted(bg_prot)}\x1b[0m\n\x1b[35m" +
	          "|\x1b[0m")
	
	#####################################
	# start taking the total time
	t_0 = time.time()

	#####################################
	# set background input
	if bg_prot["scheme"] == "FULL":
		I_0 = config["simulation"]["bg_protocol"]["I_0"]*namp #0.15*namp
		sigma_wn = config["simulation"]["bg_protocol"]["sigma_WN"]*namp*second**(1/2) #0.05*namp*second**(1/2)
	else:
		I_0 = 0*namp
		sigma_wn = 0*namp*second**(1/2)
	
	#####################################
	# set up the neuron populations of the network
	neur_pop_exc = b.NeuronGroup(N_exc, get_model("neuron_full"), threshold='V>V_th', reset='V=V_reset', refractory='t_ref', method='euler')#, namespace=parameter_nmspc)
	neur_pop_exc.V = V_rev # initialize membrane potential at reversal potential
	neur_pop_exc.V_psp = 0*mvolt # initialize postsynaptic potential at zero
	neur_pop_exc.I_bg = I_0 # initialize background current at mean value
	#neur_pop_exc.I_stim = 0*namp #N_stim*learn_prot["freq"]*h_0/R_mem # initialize stimulus current at mean value??
	#neur_pop_exc.f_stim = learn_prot["freq"]*hertz

	if N_inh > 0:
		neur_pop_inh = b.NeuronGroup(N_inh, get_model("neuron_simple"), threshold='V>V_th', reset='V=V_reset', refractory='t_ref', method='euler')#, namespace=parameter_nmspc)
		neur_pop_inh.V = V_rev # initialize membrane potential at reversal potential
		neur_pop_inh.V_psp = 0*mvolt # initialize postsynaptic potential at zero
		neur_pop_inh.I_bg = I_0 # initialize background current at mean value

	#####################################
	# set up the stimulus groups (OU and spike generator)
	#stimulus = b.TimedArray(np.hstack([[c, c, c, 0, 0]
    #                        for c in np.random.rand(1000)]),
    #                        dt=delta_t)
	
	ou_stimulus_learn = get_ou_stimulus(learn_prot["scheme"])
	if ou_stimulus_learn:
		neur_pop_ou_stim_learn = b.NeuronGroup(N_CA, ou_stimulus_learn, method='heun')#, namespace=parameter_nmspc)
		neur_pop_ou_stim_learn.f_stim = learn_prot["freq"]*hertz
		neur_pop_ou_stim_learn.N_stim = learn_prot["N_stim"]
		if hasattr(neur_pop_ou_stim_learn, "t_start"):
			neur_pop_ou_stim_learn.t_start = learn_prot["time_start"]*second
		neur_pop_ou_stim_learn.V = "N_stim*f_stim*1*second*h_0"

	ou_stimulus_recall = get_ou_stimulus(recall_prot["scheme"])
	if ou_stimulus_recall:
		neur_pop_ou_stim_recall = b.NeuronGroup(int(p_r*N_CA), ou_stimulus_recall, method='heun')
		neur_pop_ou_stim_recall.f_stim = recall_prot["freq"]*hertz
		neur_pop_ou_stim_recall.N_stim = recall_prot["N_stim"]
		if hasattr(neur_pop_ou_stim_recall, "t_start"):
			neur_pop_ou_stim_recall.t_start = recall_prot["time_start"]*second
		neur_pop_ou_stim_recall.V = "N_stim*f_stim*1*second*h_0"

	explicit_stimulus_learn = get_explicit_stimulus(learn_prot)
	if explicit_stimulus_learn:
		spike_gen_times = explicit_stimulus_learn["stim_times"]*msecond # spike times
		spike_gen = b.SpikeGeneratorGroup(1, [0]*len(explicit_stimulus_learn["stim_times"]), spike_gen_times) # one spike-generating neuron
		#spike_gen.set_spikes(spike_gen_indices, spike_gen_times)
		#print(f"spike_gen_times = {spike_gen_times}, target neurons = {explicit_input['receivers']}")

	#####################################
	# set up the synapse populations
	#syn_exc_exc = b.Synapses(neur_pop_exc, neur_pop_exc, model=get_model("plastic_synapse"), on_pre=get_model("plastic_synapse_pre"), on_post=get_model("plastic_synapse_post"), method='milstein') # model of synapses within excitatory population ('neur_pop_exc')
	syn_exc_exc = b.Synapses(neur_pop_exc, neur_pop_exc, model=get_model("plastic_synapse"),
							      on_pre=get_model("plastic_synapse_pre"), on_post=get_model("plastic_synapse_post"), method='heun') # model of synapses within excitatory population ('neur_pop_exc')
	#syn_exc_exc = b.Synapses(neur_pop_exc, neur_pop_exc, model=get_model("plastic_synapse"), on_pre='V_post += (h + z*h_0)',method='euler') # WORKS # model of synapses within 'neur_pop_exc' 
	#syn_exc_exc = b.Synapses(neur_pop_exc, neur_pop_exc, on_pre='''V_post += h_0''',method='euler') # WORKS # model of synapses within 'neur_pop_exc' 
	#syn_exc_exc = b.Synapses(neur_pop_exc, neur_pop_exc, model="h : volt", on_pre='''V_post += h''',method='euler') # WORKS # model of synapses within 'neur_pop_exc' 
	#if simtype_basic_early or simtype_2N1S:
	#    syn_exc_exc.connect(i=1,j=0) # connect neurons in 'neur_pop_exc': 1 -> 0
	#neur_pop_exc_recurr_conn = b.Synapses(neur_pop_exc, neur_pop_exc, on_pre='''V_post+=h''') # recurrent synapses within 'neur_pop_exc'

	if N_inh > 0:
		syn_exc_inh = b.Synapses(neur_pop_exc, neur_pop_inh, model=get_model("static_synapse"),
								      on_pre=get_model("static_synapse_pre"), method='euler')#, namespace=parameter_nmspc) # model of synapses from excitatory to inhibitory neurons
		#syn_exc_inh.w = w_ei
		syn_inh_exc = b.Synapses(neur_pop_inh, neur_pop_exc, model=get_model("static_synapse"),
								      on_pre=get_model("static_synapse_pre"), method='euler')#, namespace=parameter_nmspc) # model of synapses from inhibitory to excitatory neurons
		#syn_inh_exc.w = w_ie
		syn_inh_inh = b.Synapses(neur_pop_inh, neur_pop_inh, model=get_model("static_synapse"),
								      on_pre=get_model("static_synapse_pre"), method='euler')#, namespace=parameter_nmspc) # model of synapses within inhibitory population ('neur_pop_inh')
		#syn_inh_inh.w = w_ii
	
	if ou_stimulus_learn:
		# synapses from 'neur_pop_ou_stim_learn' to 'neur_pop_exc'
		syn_ou_stim_learn = b.Synapses(neur_pop_ou_stim_learn, neur_pop_exc, model=get_model("stim_input_0"))
	if ou_stimulus_recall:
		# synapses from 'neur_pop_ou_stim_recall' to 'neur_pop_exc'
		syn_ou_stim_recall = b.Synapses(neur_pop_ou_stim_recall, neur_pop_exc, model=get_model("stim_input_1"))
	if explicit_stimulus_learn:
		# synapses from 'spike_gen' to 'neur_pop_exc'
		syn_spike_gen_inp_exc = b.Synapses(spike_gen, neur_pop_exc, on_pre='V_post+=w_spike')
		if N_inh > 0:
			 # synapses from 'spike_gen' to 'neur_pop_inh'
			syn_spike_gen_inp_inh = b.Synapses(spike_gen, neur_pop_inh, on_pre='V_post+=w_spike')
	
	#####################################
	# create the connections that provide input to the network
	if ou_stimulus_learn:
		for j_ in range(N_CA): # input to all neurons in the assembly core
			syn_ou_stim_learn.connect(i=j_,j=j_) # connect neurons in `neur_pop_ou_stim_learn` one-by-one to the first
			                                     # `N_CA` neurons of `neur_pop_exc`
	if ou_stimulus_recall:
		for j_ in range(int(p_r*N_CA)): # input to all neurons in the assembly core
			syn_ou_stim_recall.connect(i=j_,j=j_) # connect neurons in `neur_pop_ou_stim_recall` one-by-one to the first
			                                      # `p_r*N_CA` neurons of `neur_pop_exc`
	if explicit_stimulus_learn:
		for j_ in explicit_stimulus_learn["receivers"]:
			# connect the explicit input neuron (in 'spike_gen') to neuron 'j_' in 'neur_pop_exc'
			if j_ < N_exc:
				syn_spike_gen_inp_exc.connect(i=0,j=j_)
			# connect the explicit input neuron (in 'spike_gen') to neuron 'j_' in 'neur_pop_inh'
			else:
				syn_spike_gen_inp_inh.connect(i=0,j=j_-N_exc)

	#####################################
	# create the synaptic connections within in the network
	
	# if a connections file is specified, load the connectivity matrix from that file
	if config["populations"]["conn_file"]: 
		writeLog(f"Loading connections from file.")
		conn_matrix = np.loadtxt(config["populations"]["conn_file"])
		'''
		connections_counter = 0 # for testing / output
		for post_neuron_id in range(N_tot): # loop over all neurons in the network
			connections = conn_matrix[post_neuron_id]
			assert connections[post_neuron_id] == 0 # check that there are no self-couplings

			#print("Expected connections (mean): " + str(round(p_c * (N_tot - 1), 1)) + ", obtained connections: " + str(np.sum(connections)))
			connections_counter += np.sum(connections)

			exc_connections = np.array(connections*np.concatenate((np.ones(N_exc, dtype=np.int8), np.zeros(N_inh, dtype=np.int8)), axis=None), dtype=bool) # array of booleans indicating the existence of any incoming excitatory connection
			inh_connections = np.array(connections*np.concatenate((np.zeros(N_exc, dtype=np.int8), np.ones(N_inh, dtype=np.int8)), axis=None), dtype=bool) # array of booleans indicating the existence of any incoming inhibitory connection

			assert not np.any(np.logical_xor(np.logical_or(exc_connections, inh_connections), connections)) # test if 'exc_connections' and 'inh_connections' together yield 'connections' again
			exc_pre_neurons = np.arange(N_tot)[exc_connections] # array of excitatory presynaptic neurons indicated by their id
			inh_pre_neurons = np.arange(N_tot)[inh_connections] - 1600 # array of inhibitory presynaptic neurons indicated by their id
			
			assert np.logical_and(np.all(exc_pre_neurons >= 0), np.all(exc_pre_neurons < N_exc)) # test if the excitatory neuron numbers are in the correct range
			assert np.logical_and(np.all(inh_pre_neurons >= 0), np.all(inh_pre_neurons < N_inh)) # test if the inhibitory neuron numbers are in the correct range

			# excitatory neurons
			if post_neuron_id < N_exc:

				# incoming excitatory synapses
				for pre_neuron_id in exc_pre_neurons:
					syn_exc_exc.connect(i=pre_neuron_id, j=post_neuron_id) # connect neurons in 'neur_pop_exc'

				# incoming inhibitory synapses
				for pre_neuron_id in inh_pre_neurons:
					syn_inh_exc.connect(i=pre_neuron_id, j=post_neuron_id) # connect inhibitory to excitatory neuron

			# inhibitory neurons
			else:
				# incoming excitatory synapses
				for pre_neuron_id in exc_pre_neurons:
					syn_exc_inh.connect(i=pre_neuron_id, j=post_neuron_id) # connect excitatory to inhibitory neuron

				# incoming inhibitory synapses
				for pre_neuron_id in inh_pre_neurons:
					syn_inh_inh.connect(i=pre_neuron_id, j=post_neuron_id) # connect neurons in 'neur_pop_inh'
		'''
		#sources, targets = conn_matrix.nonzero()
		#syn_exc_exc.connect(i=sources[sources<N_exc], j=targets[targets<N_exc])
		#if N_inh > 0:
		#	syn_exc_inh.connect(i=sources[sources<N_exc], j=targets[targets>=N_exc])
		#	syn_inh_exc.connect(i=sources[sources>=N_exc], j=targets[targets<N_exc])
		#	syn_inh_inh.connect(i=sources[sources>=N_exc], j=targets[targets>=N_exc])
		sources, targets = conn_matrix[:N_exc,:N_exc].nonzero()
		syn_exc_exc.connect(i=sources, j=targets)
		if N_inh > 0:
			sources, targets = conn_matrix[:N_exc,N_exc:].nonzero()
			syn_exc_inh.connect(i=sources, j=targets)
			sources, targets = conn_matrix[N_exc:,:N_exc].nonzero()
			syn_inh_exc.connect(i=sources, j=targets)
			sources, targets = conn_matrix[N_exc:,N_exc:].nonzero()
			syn_inh_inh.connect(i=sources, j=targets)
	# Otherwise, use Brian's own random generation mechanisms
	else:
		writeLog(f"Creating connections with probability p_c = {p_c}.")
		syn_exc_exc.connect(condition='i != j', p=p_c)
		if N_inh > 0:
			syn_exc_inh.connect(condition='i != j', p=p_c)
			syn_inh_exc.connect(condition='i != j', p=p_c)
			syn_inh_inh.connect(condition='i != j', p=p_c)
	
	# Set initial values for synapse-related variables (is not allowed to be done earlier)
	syn_exc_exc.h = h_0
	syn_exc_exc.z = 0
	syn_exc_exc.Ca = 0
	syn_exc_exc.pre_voltage.delay = t_syn_delay
	syn_exc_exc.pre_calcium.delay = t_Ca_delay
	if N_inh > 0:
		syn_exc_inh.w = w_ei
		syn_inh_exc.w = -w_ie
		syn_inh_inh.w = -w_ii
		syn_exc_inh.pre_voltage.delay = t_syn_delay
		syn_inh_exc.pre_voltage.delay = t_syn_delay
		syn_inh_inh.pre_voltage.delay = t_syn_delay
		
	#####################################
	# set up monitors and run
	spike_mon = b.SpikeMonitor(neur_pop_exc)
	neuron_ids_exc = neuron_ids[neuron_ids<N_exc]
	neur_state_mon_exc = b.StateMonitor(neur_pop_exc, neuron_observables,
								        record=neuron_ids_exc)
	if N_inh > 0:
		neuron_ids_inh = neuron_ids[neuron_ids>=N_exc] - N_exc
		neur_state_mon_inh = b.StateMonitor(neur_pop_inh, "V",
											record=neuron_ids_inh)
	else:
		neuron_ids_inh = np.array([])
	syn_state_mon = b.StateMonitor(syn_exc_exc, synapse_observables, record=synapse_ids)
	# TODO fix plotting of synapse variables

	b.run(t_max, report="stdout")#, namespace=parameter_nmspc)
	
	t_1 = time.time()
	writeLog("Simulation completed (in " + getFormattedTime(round(t_1 - t_0)) + ").")

	#####################################
	# assemble and store the data in files
	data_stacked = np.array([neur_state_mon_exc.t/msecond]).T

	for i in range(len(neuron_ids_exc)):
		data_stacked = np.column_stack(
			[data_stacked,
				neur_state_mon_exc.V[i]/mvolt,
				np.nan*np.zeros(len(neur_state_mon_exc.t))])
	for i in range(len(neuron_ids_inh)):
		data_stacked = np.column_stack(
			[data_stacked,
				neur_state_mon_inh.V[i]/mvolt,
				np.nan*np.zeros(len(neur_state_mon_inh.t))])
	for i in range(len(synapse_ids)):
			data_stacked = np.column_stack(
				[data_stacked,
					syn_state_mon.h[i]/mvolt,
					syn_state_mon.z[i],
					syn_state_mon.Ca[i],
					neur_state_mon_exc.p[0]])
			# TODO recording of protein from different neurons

	spike_times = np.column_stack((spike_mon.t/msecond, spike_mon.i))

	np.savetxt(getDataPath(file='traces.txt'), data_stacked, fmt="%.4f")
	np.savetxt(getDataPath(file='spikes.txt'), spike_times, fmt="%.4f %.0f") # integer formatting for neuron number

	json.dump(config, open(getDataPath(file='config.json'), "w"), indent="\t")

	t_2 = time.time()
	writeLog("Data stored (in " + getFormattedTime(round(t_2 - t_1)) + ").")

	#####################################
	# plotting the data
	plotMonitorResults(config, neuron_ids, synapse_ids,
	                   spike_times, data_stacked,
	                   getDataPath(), interactive)
	#for i in range(len(neuron_ids)):
	#	plotResults(config, data_stacked, getTimestamp(), i, 
	#		        neuron=neuron_ids[i], synapse="n", 
	#				store_path = getDataPath(), figure_fmt = 'svg')
	t_3 = time.time()
	writeLog("Plotting completed (in " + getFormattedTime(round(t_3 - t_2)) + ").")

	#####################################
	# Read out weights and store connectivity matrix
	# -- doing this before running is not possible with `cpp_standalone` device
	W_exc = np.zeros((N_exc, N_exc))
	W_exc_inh = np.zeros((N_exc, N_inh))
	W_inh_exc = np.zeros((N_inh, N_exc))
	W_inh = np.zeros((N_inh, N_inh))
	W_exc[syn_exc_exc.i[:], syn_exc_exc.j[:]] = (syn_exc_exc.h[:]/mvolt != 0)
	if N_inh > 0:
		W_exc_inh[syn_exc_inh.i[:], syn_exc_inh.j[:]] = (syn_exc_inh.w[:]/mvolt != 0)
		W_inh_exc[syn_inh_exc.i[:], syn_inh_exc.j[:]] = (syn_inh_exc.w[:]/mvolt != 0)
		W_inh[syn_inh_inh.i[:], syn_inh_inh.j[:]] = (syn_inh_inh.w[:]/mvolt != 0)
	W = np.vstack((np.column_stack((W_exc, W_exc_inh)),
	               np.column_stack((W_inh_exc, W_inh))))
	np.savetxt(getDataPath(file='connections.txt'), W, fmt="%1.0f")
	t_4 = time.time()
	writeLog("Retrieval of connectivity matrix completed (in " + getFormattedTime(round(t_4 - t_3)) + ").")
	writeLog("Number of connections: "+ str(int(np.sum(W))) + " (expected value: " + str(round(p_c * (N_tot**2 - N_tot), 1)) + ").")

	#####################################
    # close the log file
	writeLog("Total elapsed time: " + getFormattedTime(round(t_4 - t_0)) + ".")
	closeLog()
	

#####################################
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-s_desc', type=str, help="short description")
	parser.add_argument('-config_file', required=True, help="configuration of the simulation parameters (JSON file)")
	parser.add_argument('-runtime', type=float, help="runtime of the simulation in s")
	parser.add_argument('-dt', type=float, help="duration of one timestep in ms")
	parser.add_argument('-recall_start', type=float, help="beginning of recall stimulation in seconds")
	parser.add_argument('-N_CA', type=int, help="number of neurons in the cell assembly")
	parser.add_argument('-w_ie', type=float, help="inhibitory to excitatory coupling strength in units of h_0")
	parser.add_argument('-w_ii', type=float, help="inhibitory to excitatory coupling strength in units of h_0")
	# TODO introduce setting of learning and recall protocols, update run scripts for smallnet2
	args = parser.parse_args()
	
	# load parameter configuration as dictionary from JSON file
	if (args.config_file is not None): 
		config_file = args.config_file
	else:
		config_file = "config_smallnet2.json"
		
	config = json.load(open(config_file, "r"))

	# overwrite parameter values
	if (args.s_desc is not None): config['simulation']['short_description'] = args.s_desc
	if (args.runtime is not None): config['simulation']['runtime'] = args.runtime
	if (args.dt is not None): config['simulation']['dt'] = args.dt
	if (args.recall_start is not None): config['simulation']['recall_protocol']['time_start'] = args.recall_start
	if (args.N_CA is not None): config['populations']['N_CA'] = args.N_CA
	if (args.w_ie is not None): config['populations']['w_ie'] = args.w_ie
	if (args.w_ii is not None): config['populations']['w_ii'] = args.w_ii

	run(config)
	"""neuron_monitored_variables = ['V', 'p']
	neuron_monitored_ids = [0,1,2,3]
	synapse_monitored_variables = ['h', 'z', 'Ca']
	synapse_monitored_ids = [0,1,2,3,4] # strangely, more indices can be added than available (e.g., [0,1,2,3,4,5,6,7])
	run(config, 
	    neuron_monitored_variables, neuron_monitored_ids,
		synapse_monitored_variables, synapse_monitored_ids)"""