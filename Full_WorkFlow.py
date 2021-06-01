# # Full Workflow
from funcs import *

RepoPath = '/home/shreejit/WORK/CBCLikelihood'
mass_min = 5.
mass_max = 445. # Heavier than this will lower the merger freq of heaviest BBH (225, 225) below f_min
templ_fmin = 10. # start freq of templates
i_f_start = 2**5 # = 2048. Starting INDEX for freq of the PSD
Mtot = np.arange(10, mass_min+mass_max, 10)
Data = {'Total Mass': Mtot, 'Sample Frequencies': [], 'PSD': {}, 'Redshift': {}, 'Max Horizon': {}}
RUN='Aplus'

## Finding Horizon
first_time = True

Data['Redshift'][RUN] = []
Data['Max Horizon'][RUN] = {}
Data['PSD'][RUN] = {}
for det in ['H1', 'L1']:
    # Read PSD
    path=RepoPath+"/Data/PSD_"+RUN+".txt"
    psd = np.genfromtxt(path, delimiter=" ")
    if first_time:
        Data['Sample Frequencies'] = psd[:, 0]
        first_time = False
    Data['PSD'][RUN][det] = psd[:, 1]

for M in tqdm(Data['Total Mass']):
    # ATTN: Only L1 PSD used for calculating horizon redshift
    Data['Redshift'][RUN].append(horizon_redshift(Data['Sample Frequencies'][i_f_start:], Data['PSD'][RUN]['L1'][i_f_start:], omega=OMEGA, m1=M/2., m2=M/2.))

Data['Redshift'][RUN] = np.array(Data['Redshift'][RUN])
# catch max redshift, corresponding mass and luminosity distance
i_max = Data['Redshift'][RUN].argmax()

Data['Max Horizon'][RUN]['Total Mass'] = Data['Total Mass'][i_max]
Data['Max Horizon'][RUN]['Redshift'] = Data['Redshift'][RUN][i_max]
Data['Max Horizon'][RUN]['Luminosity Distance'] = lal.LuminosityDistance(OMEGA, Data['Redshift'][RUN][i_max])

Zmax = {}

#print "{}: Maximum Horizon Redshift is {} for a binary with total mass {}".format(RUN, Data['Max Horizon'][RUN]['Redshift'], Data['Max Horizon'][RUN]['Total Mass'])
# Conservatively setting the max z 0.1 higher than obtained z
Zmax[RUN] = Data['Max Horizon'][RUN]['Redshift'] + 0.1


# # Injections
N = 100000 # Number of injections
df = Data['Sample Frequencies'][1] - Data['Sample Frequencies'][0]
Distributions = {"Uniform": Uniform, "Log_Flat": log_flat, "Power_Law": pow_law}


# ## Create HDF file
DtTime = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')[:-10]
Path = "{}/Data/{}_Injection_Data_{}.hdf5".format(RepoPath, DtTime, N)
# Create HDF5 file
inj_data = hp.File(Path, "a")
print("Data File: " + Path)

# ## Random sky position, orientation, spins and polarization
alpha, delta, iota, spin1, spin2, pol, inj_t = sample_params(N)
Iota = inj_data.create_dataset("iota", data=iota)
inj_data.create_dataset("alpha", data=alpha)
inj_data.create_dataset("delta", data=delta)
Spin1 = inj_data.create_dataset("spin1", data=spin1)
Spin2 = inj_data.create_dataset("spin2", data=spin2)
inj_data.create_dataset("polarization", data=pol)
inj_data.create_dataset("injection_time", data=gps_time)


# ## Evaluating SNRs

# Draw redshift values such that the injections are uniform in comoving vol
z = sample_redshifts(Zmax[RUN], N)
# Convert to luminosity distances
dlum = convert_to_dlum(z)
# Find effective distances for corresponding detectors
d_eff_dict = find_eff_d(dlum, alpha, delta, pol, iota, inj_t)

inj_data_for_run = inj_data.create_group(RUN)

psd_H1 = pf.FrequencySeries(Data['PSD'][RUN]['H1'], delta_f=df)
psd_L1 = pf.FrequencySeries(Data['PSD'][RUN]['L1'], delta_f=df)

# In each distribution
for distrib in Distributions:
    in_distrib = inj_data_for_run.create_group(distrib)
    # Masses M1 and M2 generated according to "Distribution"
    m1, m2 = Distributions[distrib](N, mass_min, mass_max)
    Mass1 = in_distrib.create_dataset("Mass1", data=m1)
    Mass2 = in_distrib.create_dataset("Mass2", data=m2)
    # Calculating SNRs
    SNR = []
    for i in tqdm(range(N)):
        # print("mass1=", m1[i], " mass2=", m2[i], " z=", z[i], " spin1z=", spin1[i], \
        #      " spin2z=", spin2[i], " inclination=", iota[i])
        # Get each template
        sptilde, _ = get_fd_waveform(approximant="IMRPhenomD", mass1=m1[i]*(1+z[i]), mass2=m2[i]*(1+z[i]), spin1z=spin1[i], spin2z=spin2[i], distance=1., inclination=iota[i], delta_f=df, f_lower=templ_fmin, f_final=Data['Sample Frequencies'][-1])
        sptilde.resize(len(Data['PSD']['O1']['L1']))
        rho = mf.sigmasq(sptilde, psd=psd_H1, low_frequency_cutoff=Data['Sample Frequencies'][0]) / (d_eff_dict['H1'][i] ** 2.)
        rho += mf.sigmasq(sptilde, psd=psd_L1,                               low_frequency_cutoff=Data['Sample Frequencies'][0]) / (d_eff_dict['L1'][i] ** 2.)
        rho **= 0.5
        SNR.append(rho)
    SNR = np.array(SNR)
    # Save the SNR array for the current distribution
    in_distrib.create_dataset("SNR", data=SNR)

inj_data.close()
