import numpy as np

TCMB = 2.725  # Kelvin
hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS

datadir = '/Users/abitbol/code/self_calibration_fg/data/'

def truncate(data, lmin, lmax):
    mask = (data['ells'] >= lmin) * (data['ells'] <= lmax)
    ndata = {} 
    for ps in data:
        ndata[ps] = data[ps][mask]
    return ndata

def load_SO_noise():
    SO_freqs = [ 27.,  39.,  93., 145., 225., 280.]
    fdata = np.load(datadir+'SO_calc_mode2-1_SATyrsLF1_noise_SAT_P.npy', encoding='bytes')
    SO_ell = fdata[0]
    noise_data = fdata[1]
    unit = SO_ell * (SO_ell + 1.) / (2. * np.pi)
    SO_noise = {}
    SO_noise['ells'] = SO_ell
    for k, fs in enumerate(SO_freqs):
        SO_noise[fs] = noise_data[k] * unit
    return SO_noise

def load_cmb():
    # data is Dl in uK^2    
    fdata = np.loadtxt(datadir+'camb_lens_nobb.dat')
    cmb_cls = {}
    cmb_cls['ells'] = fdata[:, 0]
    cmb_cls['TT'] = fdata[:, 1]
    cmb_cls['EE'] = fdata[:, 2]
    cmb_cls['BB'] = fdata[:, 3]
    cmb_cls['TE'] = fdata[:, 4]
    cmb_cls['TB'] = np.zeros(len(fdata[:, 0]))
    cmb_cls['EB'] = np.zeros(len(fdata[:, 0]))
    return cmb_cls

def load_dust(ells, m=1., EBfrac=0.03):
    # At 353 GHz
    alpha_EE = -2.28
    alpha_BB = -2.16
    A_EE = 34.3 / m
    rA_BB_EE = 0.48
    dust = {}
    dust['EE'] = dustfit(ells, A_EE, alpha_EE)
    dust['BB'] = dustfit(ells, A_EE * rA_BB_EE, alpha_BB)
    dust['EB'] = EBfrac * dust['EE']
    return dust

def load_synch(ells, m=1., EBfrac=0.03):
    # At 2.3 GHz 
    alpha_EE = -3.3
    alpha_BB = -3.18
    A_EE = 4.e-3 * 1e6    #mK^2 to uK^2
    A_BB = 1.7e-3 * 1e6
    Ap_EE = 72.
    Ap_BB = 66
    synch = {}
    synch['EE'] = synchfit(ells, A_EE, Ap_EE, alpha_EE)
    synch['BB'] = synchfit(ells, A_BB, Ap_BB, alpha_BB)
    synch['EB'] = EBfrac * synch['EE']
    return synch

def dustfit(ell, A, alpha):
    return A * (ell / 80.)**(alpha + 2.)

def synchfit(ell, As, Ap, alpha):
    unit = ell * (ell + 1.) / (2. * np.pi)
    return (As * (ell / 80.)**alpha + Ap) * unit

def scale_dust(dust, nu):
    ndust = {}
    for ps in dust:
        unit = normed_dust(nu) * normed_cmb_thermo_units(353.e9) / normed_cmb_thermo_units(nu) 
        ndust[ps] = dust[ps] * unit**2
    return ndust

def scale_synch(synch, nu):
    nsynch = {}
    for ps in synch:
        unit = normed_synch(nu) * normed_cmb_thermo_units(2.3e9) / normed_cmb_thermo_units(nu) 
        nsynch[ps] = synch[ps] * unit**2
    return nsynch

def normed_dust(nu, beta=1.53):
    Td = 19.6 
    nu0 = 353.e9
    X = hplanck * nu / (kboltz * Td)
    X0 = hplanck * nu0 / (kboltz * Td)
    return (nu/nu0)**(3.+beta) * (np.exp(X0) - 1.) / (np.exp(X) - 1.)

def normed_cmb_thermo_units(nu):
    X = hplanck * nu / (kboltz * TCMB)
    eX = np.exp(X)
    return eX * X**4 / (eX - 1.)**2 

def normed_synch(nu, beta=-3.2):
    nu0 = 2.3e9
    return (nu/nu0)**(2.+beta)



