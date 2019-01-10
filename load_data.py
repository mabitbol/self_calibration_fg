import numpy as np

TCMB = 2.725  # Kelvin
hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS

def load_cmb():
    # data is Dl in uK^2    
    fname = 'data/camb_lens_nobb.dat'
    fdata = np.loadtxt(fname)
    cmb_cls = {}
    cmb_cls['ells'] = fdata[:, 0]
    cmb_cls['TT'] = fdata[:, 1]
    cmb_cls['EE'] = fdata[:, 2]
    cmb_cls['BB'] = fdata[:, 3]
    cmb_cls['TE'] = fdata[:, 4]
    cmb_cls['TB'] = np.zeros(len(ells))
    cmb_cls['EB'] = np.zeros(len(ells))
    return cmb_cls

def load_dust(ells, EBfrac=0.03):
    # At 353 GHz
    alpha_EE = -2.28
    alpha_BB = -2.16
    A_EE = 34.3 
    rA_BB_EE = 0.48
    dust = {}
    dust['EE'] = dustfit(ells, A_EE, alpha_EE)
    dust['BB'] = dustfit(ells, A_EE * rA_BB_EE, alpha_BB)
    dust['EB'] = EBfrac * dust['EE']
    dust['ells'] = ells
    return dust

def dustfit(ell, A, alpha):
    return A * (ell / 80.)**(alpha + 2.)

def truncate(data, lmin, lmax):
    mask = (data['ells'] >= lmin) * (data['ells'] <= lmax)
    for ps in data:
        data[ps] = data[ps][mask]
    return data

def scale_dust(dust, nu):
    for ps in dust:
        unit = normed_dust(nu) * normed_cmb_thermo_units(353.e9) / normed_cmb_thermo_units(nu) 
        dust[ps] = dust[ps] * unit
    return dust


def normed_dust(nu, beta=1.53):
    Td = 19.6 # K
    nu0 = 353.e9
    X = hplanck * nu / (kboltz * Td)
    X0 = hplanck * nu0 / (kboltz * Td)
    return (nu/nu0)**(3.+beta) * (np.exp(X0) - 1.) / (np.exp(X) - 1.)

def normed_cmb_thermo_units(nu):
    X = hplanck * nu / (kboltz * TCMB)
    eX = np.exp(X)
    return eX * X**4 / (eX - 1.)**2 

def normed_synch(nu, beta):
    nu0 = 30.e9
    return (nu/nu0)**(2.+beta)

def normed_plaw(ell, alpha):
    ell0 = 80.
    return (ell/ell0)**alpha 


def load_SO_noise():
    SO_freqs = [ 27.,  39.,  93., 145., 225., 280.]
    fdata = np.load('../self_calibration_fg/data/SO_calc_mode1-1_SATyrsLF5_fsky0.1_noise_SAT_P.npy')
    SO_ell = fdata[0]
    noise_data = fdata[1]
    unit = SO_ell * (SO_ell + 1.) / (2. * pi)
    SO_noise = {}
    for k, fs in enumerate(SO_freqs):
        SO_noise[fs] = noise_data[k] * unit
    return SO_ell, SO_noise



