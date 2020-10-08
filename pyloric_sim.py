from brian2 import *
import sys
import random
import sys

set_device('cpp_standalone', directory = "brain_output_" + sys.argv[1], debug=False)

TEST_RUN = True
OUTPUT_FOLDER = "output_pyloric_noisy4"
if TEST_RUN:
    RUN_LENGTH = 100 * second
else:
    RUN_LENGTH = 10000 * second


defaultclock.dt = 0.005*ms
Cm = (0.628*nfarad)
F = 96485.33212
R = 8.314462618
T = 20
tau_g = 50*ms
F_Ca = 14.96
Ca_EQ = 0.05
TAU_Ca = 200 * ms
EXTRACELLULAR_Ca = 3000

noise_sigma_I = 9e-9
noise_theta_I = 1000

time_multiplier = 1*second

ABPD_properties = {
    'tau_g_Na' : 0.25 * time_multiplier,
    'tau_g_CaT' : 40 * time_multiplier,
    'tau_g_CaS' : 16.67 * time_multiplier,
    'tau_g_A' : 2 * time_multiplier,
    'tau_g_KCa' : 10 * time_multiplier,
    'tau_g_Kd' : 1 * time_multiplier,
    'tau_g_H' : 10000 * time_multiplier,
    'tau_g_s_chol' : 1e15 * time_multiplier,
    'tau_g_s_glut_1' : 2e4 * time_multiplier,
    #'tau_g_s_glut_1' : 1e15 * time_multiplier,
    'tau_g_s_glut_2' :  1e15 * time_multiplier,
    'Ca_tgt' : 90,
    'g_l' : 0.000 * usiemens,
    'E_l' : -50*mV,
    'E_Na' : 50*mV,
    'E_A' : -80*mV,
    'E_H' : -20*mV,
    'E_Kd' : -80*mV,
    'E_KCa' : -80*mV,
}

LP_properties = {
    'tau_g_Na' : 1 * time_multiplier,
    'tau_g_CaT' :  1e15 * time_multiplier,
    'tau_g_CaS' : 25 * time_multiplier,
    'tau_g_A' : 5 * time_multiplier,
    'tau_g_KCa' : 20 * time_multiplier,
    'tau_g_Kd' : 4 * time_multiplier,
    'tau_g_H' : 2000 * time_multiplier,
    'tau_g_s_chol' : 500 * time_multiplier,
    'tau_g_s_glut_1' : 1e4 * time_multiplier,
    'tau_g_s_glut_2' : 1e4 * time_multiplier,
    #'tau_g_s_glut_2' : 1e15 * time_multiplier,
    'Ca_tgt' : 20,
    'g_l' : 0.02 * 0.628 * usiemens,
    'E_l' : -50*mV,
    'E_Na' : 50*mV,
    'E_A' : -80*mV,
    'E_H' : -20*mV,
    'E_Kd' : -80*mV,
    'E_KCa' : -80*mV,
}

PY_properties = {
    'tau_g_Na' : 1 * time_multiplier,
    'tau_g_CaT' : 40 * time_multiplier,
    'tau_g_CaS' : 40 * time_multiplier,
    'tau_g_A' :  2 * time_multiplier,
    'tau_g_KCa' : 1e15 * time_multiplier,
    'tau_g_Kd' : 0.8 * time_multiplier,
    'tau_g_H' : 2000 * time_multiplier,
    'tau_g_s_chol' : 5e3 * time_multiplier,
    'tau_g_s_glut_1' : 250 * time_multiplier,
    'tau_g_s_glut_2' : 1e6 * time_multiplier,
    #'tau_g_s_glut_2' : 1e15 * time_multiplier,
    'Ca_tgt' : 20,
    'g_l' : 0.01 * 0.628 * usiemens,
    'E_l' : -50*mV,
    'E_Na' : 50*mV,
    'E_A' : -80*mV,
    'E_H' : -20*mV,
    'E_Kd' : -80*mV,
    'E_KCa' : -80*mV,
}



eqs = Equations('''
dv/dt = (-I_l - I_Na - I_CaT - I_CaS - I_A - I_KCa  - I_Kd - I_H - I_glut_1 - I_glut_2 - I_chol - noise_I*amp)/Cm : volt
dnoise_I/dt = (-noise_theta_I*noise_I)/(1*second) + (noise_sigma_I*xi)*sqrt(1/(1*second)) : 1
I_glut_1 : amp
I_glut_2 : amp
I_chol : amp
I_l = g_l*(v-E_l) : amp
I_Na = g_Na*((m_Na**3)*h_Na)*(v-E_Na) : amp
I_CaT = g_CaT*((m_CaT**3)*h_CaT)*(v-E_Ca) : amp
I_CaS = g_CaS*((m_CaS**3)*h_CaS)*(v-E_Ca) : amp
I_A = g_A*((m_A**3)*h_A)*(v-E_A) : amp
I_KCa = g_KCa*(m_KCa**4)*(v-E_KCa) : amp
I_Kd = g_Kd*(m_Kd**4)*(v-E_Kd) : amp
I_H = g_H*(m_H**1)*(v-E_H) : amp
dCa/dt = (-F_Ca*((I_CaT/(1e-9*amp)) + (I_CaS/(1e-9*amp))) - Ca + Ca_EQ)/(TAU_Ca) : 1
E_Ca = ((R * (T + 273.15))/(2 * F)) * log10(EXTRACELLULAR_Ca/Ca) * 1e3 * mV : volt
dm_Na/dt = (m_Na_inf - m_Na)/tau_m_Na : 1
dm_CaT/dt = (m_CaT_inf - m_CaT)/tau_m_CaT : 1
dm_CaS/dt = (m_CaS_inf - m_CaS)/tau_m_CaS : 1
dm_A/dt = (m_A_inf - m_A)/tau_m_A : 1
dm_KCa/dt = (m_KCa_inf - m_KCa)/tau_m_KCa : 1
dm_Kd/dt = (m_Kd_inf - m_Kd)/tau_m_Kd : 1
dm_H/dt = (m_H_inf - m_H)/tau_m_H : 1
dh_Na/dt = (h_Na_inf - h_Na)/tau_h_Na : 1
dh_CaT/dt = (h_CaT_inf - h_CaT)/tau_h_CaT : 1
dh_CaS/dt = (h_CaS_inf - h_CaS)/tau_h_CaS : 1
dh_A/dt = (h_A_inf - h_A)/tau_h_A : 1
v_n = v/mV : 1
m_Na_inf = 1/(1 + exp((v_n+25.5)/(-5.29))) : 1
m_CaT_inf = 1/(1 + exp((v_n+27.1)/(-7.2))) : 1
m_CaS_inf = 1/(1 + exp((v_n+33)/(-8.1))) : 1
m_A_inf = 1/(1 + exp((v_n+27.2)/(-8.7))) : 1
m_KCa_inf = (Ca/(Ca + 3))*(1/(1 + exp((v_n+28.3)/(-12.6)))) : 1
m_Kd_inf = 1/(1 + exp((v_n+12.3)/(-11.8))) : 1
m_H_inf = 1/(1 + exp((v_n+75)/(5.5))) : 1
h_Na_inf = 1/(1 + exp((v_n+48.9)/(5.18))) : 1
h_CaT_inf = 1/(1 + exp((v_n+32.1)/5.5)) : 1
h_CaS_inf = 1/(1 + exp((v_n+60)/6.2)) : 1
h_A_inf = 1/(1 + exp((v_n+56.9)/4.9)) : 1
tau_m_Na = (2.64 - (2.52/(1 + exp((v_n+120)/(-25.0)))))*ms : second
tau_m_CaT = (43.4 - (42.6/(1 + exp((v_n+68.1)/(-20.5)))))*ms : second
tau_m_CaS = (2.8 + (14/(exp((v_n+27)/10) + exp((v_n+70)/(-13)))))*ms : second
tau_m_A = (23.2 - (20.8/(1 + exp((v_n+32.9)/(-15.2)))))*ms : second
tau_m_KCa = (180.6 - (150.2/(1 + exp((v_n+46)/(-22.7)))))*ms : second
tau_m_Kd = (14.4 - (12.8/(1 + exp((v_n+28.3)/(-19.2)))))*ms : second
tau_m_H = (2/(exp((v_n+169.7)/(-11.6)) + exp((v_n-26.7)/14.3)))*ms : second
tau_h_Na = ((1.34/(1 + exp((v_n+62.9)/(-10.0)))) * (1.5 + (1/(1 + exp((v_n + 34.9)/3.6)))))*ms : second
tau_h_CaT = (210 - (179.6/(1 + exp((v_n + 55)/(-16.9)))))*ms : second
tau_h_CaS = (120 + (300/(exp((v_n+55)/9) + exp((v_n+65)/(-16)))))*ms : second
tau_h_A = (77.2 - (58.4/(1 + exp((v_n + 38.9)/(-26.5)))))*ms : second
dg_Na/dt = ((mRNA_Na * usiemens) - g_Na)/tau_g : siemens
dmRNA_Na/dt = -(Ca - Ca_tgt)/tau_g_Na : 1
dg_CaT/dt = ((mRNA_CaT * usiemens) - g_CaT)/tau_g : siemens
dmRNA_CaT/dt = -(Ca - Ca_tgt)/tau_g_CaT : 1
dg_CaS/dt = ((mRNA_CaS * usiemens) - g_CaS)/tau_g : siemens
dmRNA_CaS/dt = -(Ca - Ca_tgt)/tau_g_CaS : 1
dg_A/dt = ((mRNA_A * usiemens) - g_A)/tau_g : siemens
dmRNA_A/dt = -(Ca - Ca_tgt)/tau_g_A : 1
dg_KCa/dt = ((mRNA_KCa * usiemens) - g_KCa)/tau_g : siemens
dmRNA_KCa/dt = -(Ca - Ca_tgt)/tau_g_KCa : 1
dg_Kd/dt = ((mRNA_Kd * usiemens) - g_Kd)/tau_g : siemens
dmRNA_Kd/dt = -(Ca - Ca_tgt)/tau_g_Kd : 1
dg_H/dt = ((mRNA_H * usiemens) - g_H)/tau_g : siemens
dmRNA_H/dt = -(Ca - Ca_tgt)/tau_g_H : 1
dg_s_chol/dt = ((mRNA_s_chol * usiemens) - g_s_chol)/tau_g : siemens
dmRNA_s_chol/dt = -(Ca - Ca_tgt)/tau_g_s_chol : 1
dg_s_glut_1/dt = ((mRNA_s_glut_1 * usiemens) - g_s_glut_1)/tau_g : siemens
dmRNA_s_glut_1/dt = -(Ca - Ca_tgt)/tau_g_s_glut_1 : 1
dg_s_glut_2/dt = ((mRNA_s_glut_2 * usiemens) - g_s_glut_2)/tau_g : siemens
dmRNA_s_glut_2/dt = -(Ca - Ca_tgt)/tau_g_s_glut_2 : 1
''')

ABPD = NeuronGroup(1, model=eqs, method = 'euler', threshold='v>0*mV', refractory='v>-0*mV',
                   namespace = ABPD_properties)

LP = NeuronGroup(1, model=eqs, method = 'euler', threshold='v>0*mV', refractory='v>-0*mV',
                namespace = LP_properties)

PY = NeuronGroup(1, model=eqs, method = 'euler', threshold='v>0*mV', refractory='v>-0*mV',
                namespace = PY_properties)

for neuron, neuron_properties in zip([ABPD, LP, PY], [ABPD_properties, LP_properties, PY_properties]):
    neuron.v = '-50*mV'
    neuron.Ca = '0.05'
    neuron.m_Na = '0.0097'
    neuron.m_CaT = '0.04'
    neuron.m_CaS = '0.109'
    neuron.m_A = '0.068'
    neuron.m_KCa = '0.002'
    neuron.m_Kd = '0.039'
    neuron.m_H = '0.011'
    neuron.h_Na = '0.553'
    neuron.h_CaT = '0.963'
    neuron.h_CaS = '0.166'
    neuron.h_A = '0.197'

    # Set initial values, maintaining the conductance ratios
    multiplier = 0.5
    neuron.g_Na = (multiplier * second * usiemens) / neuron_properties["tau_g_Na"]
    neuron.g_CaT = (multiplier * second * usiemens) / neuron_properties["tau_g_CaT"]
    neuron.g_CaS = (multiplier * second * usiemens) / neuron_properties["tau_g_CaS"]
    neuron.g_A = (multiplier * second * usiemens) / neuron_properties["tau_g_A"]
    neuron.g_KCa = (multiplier * second * usiemens) / neuron_properties["tau_g_KCa"]
    neuron.g_Kd = (multiplier * second * usiemens) / neuron_properties["tau_g_Kd"]
    neuron.g_H =(multiplier * second * usiemens) / neuron_properties["tau_g_H"]
    neuron.g_s_chol = (multiplier * second * usiemens) / neuron_properties["tau_g_s_chol"]
    neuron.g_s_glut_1 = (multiplier * second * usiemens) / neuron_properties["tau_g_s_glut_1"]
    neuron.g_s_glut_2 = (multiplier * second * usiemens) / neuron_properties["tau_g_s_glut_2"]

    neuron.mRNA_Na = neuron.g_Na / usiemens
    neuron.mRNA_CaT = neuron.g_CaT / usiemens
    neuron.mRNA_CaS = neuron.g_CaS / usiemens
    neuron.mRNA_A = neuron.g_A / usiemens
    neuron.mRNA_KCa = neuron.g_KCa / usiemens
    neuron.mRNA_Kd = neuron.g_Kd / usiemens
    neuron.mRNA_H = neuron.g_H / usiemens
    neuron.mRNA_s_chol = neuron.g_s_chol / usiemens
    neuron.mRNA_s_glut_1 = neuron.g_s_glut_1 / usiemens
    neuron.mRNA_s_glut_2 = neuron.g_s_glut_2 / usiemens

tau_s_bar = 50*ms
eqs_glut_synapse_1 = Equations('''
ds/dt = (s_bar - s)/tau_s : 1
ds_bar/dt = (0 - s_bar)/tau_s_bar : 1
tau_s = (1 - s_bar)/(1/(40*ms)) : second
I_glut_1_post = g_s_glut_1_post * (s) * (v_post - (-70*mV)) : amp (summed)
''')

eqs_glut_synapse_2 = Equations('''
ds/dt = (s_bar - s)/tau_s : 1
ds_bar/dt = (0 - s_bar)/tau_s_bar : 1
tau_s = (1 - s_bar)/(1/(40*ms)) : second
I_glut_2_post = g_s_glut_2_post * (s) * (v_post - (-70*mV)) : amp (summed)
''')

eqs_chol_synapse = Equations('''
ds/dt = (s_bar - s)/tau_s : 1
ds_bar/dt = (0 - s_bar)/tau_s_bar : 1
tau_s = (1 - s_bar)/(1/(100*ms)) : second
I_chol_post = g_s_chol_post * (s) * (v_post - (-80*mV)) : amp (summed)
''')

s_bar_post = 0.99
ABPD_LP_glut = Synapses(ABPD, LP, model = eqs_glut_synapse_1, on_pre = 's_bar = s_bar_post')
ABPD_LP_glut.connect()
ABPD_PY_glut = Synapses(ABPD, PY, model = eqs_glut_synapse_1, on_pre = 's_bar = s_bar_post')
ABPD_PY_glut.connect()
ABPD_LP_chol = Synapses(ABPD, LP, model = eqs_chol_synapse, on_pre = 's_bar = s_bar_post')
ABPD_LP_chol.connect()
ABPD_PY_chol = Synapses(ABPD, PY, model = eqs_chol_synapse, on_pre = 's_bar = s_bar_post')
ABPD_PY_chol.connect()
LP_ABPD_glut = Synapses(LP, ABPD, model = eqs_glut_synapse_1, on_pre = 's_bar = s_bar_post')
LP_ABPD_glut.connect()
LP_PY_glut = Synapses(LP, PY, model = eqs_glut_synapse_2, on_pre = 's_bar = s_bar_post')
LP_PY_glut.connect()
PY_LP_glut = Synapses(PY, LP, model = eqs_glut_synapse_2, on_pre = 's_bar = s_bar_post')
PY_LP_glut.connect()



if TEST_RUN:
    trace_ABPD = StateMonitor(ABPD, 'v', record=True, dt = 0.5 * ms, )
    trace_LP = StateMonitor(LP, 'v', record=True, dt = 0.5 * ms)
    trace_PY = StateMonitor(PY, 'v', record=True, dt = 0.5 * ms)
spikes_ABPD = SpikeMonitor(ABPD)
spikes_LP = SpikeMonitor(LP)
spikes_PY = SpikeMonitor(PY)

run(RUN_LENGTH, report='text')

print(len(spikes_ABPD.spike_trains()[0]))
print(len(spikes_LP.spike_trains()[0]))
print(len(spikes_PY.spike_trains()[0]))

if TEST_RUN:
    plot(trace_ABPD.t/ms, trace_ABPD[0].v/mV + 300, linewidth = 3, color = (0, 0.39216, 0.78431))
    plot(trace_LP.t/ms, trace_LP[0].v/mV + 150, linewidth = 3, color = (0.90196, 0.1568, 0))
    plot(trace_PY.t/ms, trace_PY[0].v/mV + 0, linewidth = 3, color = (0.23529, 0.58823, 0.31372))
    xlabel('t (ms)')
    ylabel('v (mV)')
    show()
else:
    out_spikes = open(OUTPUT_FOLDER + "/abpd_" + sys.argv[1] + ".dat", "w")
    for spike in spikes_ABPD.spike_trains()[0]:
        out_spikes.write(str(float(spike)) + "\n")
    out_spikes.close()

    out_spikes = open(OUTPUT_FOLDER + "/lp_" + sys.argv[1] + ".dat", "w")
    for spike in spikes_LP.spike_trains()[0]:
        out_spikes.write(str(float(spike)) + "\n")
    out_spikes.close()

    out_spikes = open(OUTPUT_FOLDER + "/py_" + sys.argv[1] + ".dat", "w")
    for spike in spikes_PY.spike_trains()[0]:
        out_spikes.write(str(float(spike)) + "\n")
    out_spikes.close()
