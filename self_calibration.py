import numpy as np
from numpy import sin, cos

import load_data as ld

[TT, EE, BB, TE, TB, EB] = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
SO_freqs = [ 27.,  39.,  93., 145., 225., 280.]
d2r = np.pi / 180.

def full_rotate_cls(psi, cmb, dust):
    obs={} 
    obs['ells'] = cmb['ells']
    #obs[TT] = cmb[TT]+dust[TT]
    obs[EE] = (sin(2*psi)**2)*(cmb[BB]+dust[BB]) + (cos(2*psi)**2)*(cmb[EE]+dust[EE]) + sin(4*psi)*dust[EB]
    obs[BB] = (cos(2*psi)**2)*(cmb[BB]+dust[BB]) + (sin(2*psi)**2)*(cmb[EE]+dust[EE]) - sin(4*psi)*dust[EB]
    #obs[TE] = cos(2*psi)*(cmb[TE]+dust[TE]) + sin(2*psi)*dust[TB]
    #obs[TB] = -sin(2*psi)*(cmb[TE]+dust[TE]) + cos(2*psi)*dust[TB]
    obs[EB]  = 0.5*sin(4*psi)*(cmb[BB]-cmb[EE]+dust[BB]-dust[EE]) + cos(4*psi)*dust[EB]
    return obs

def prepare_cmb_so(truncate_30=False):
    cmb_cls = ld.load_cmb()
    so_noise = ld.load_SO_noise()
    so_ell = so_noise['ells']
    
    if truncate_30:
        so_noise = ld.truncate(so_noise, lmin=200., lmax=so_ell.max())
        so_ell = so_noise['ells']

    cmb_cls = ld.truncate(cmb_cls, lmin=so_ell.min(), lmax=so_ell.max())
    assert np.all(so_ell == cmb_cls['ells'])
    return cmb_cls, so_noise

def calc_eb_var(cmb, so_noise, fsky=0.1):
    C_EE_tot = cmb['EE'] + so_noise
    C_BB_tot = cmb['BB'] + so_noise
    delta_EB_var = 1. / (2. * cmb['ells'] + 1.) * C_EE_tot * C_BB_tot / fsky
    return delta_EB_var

def prepare_foregrounds(ells, nu):
    dust_353 = ld.load_dust(ells)
    dust = ld.scale_dust(dust_353, nu)
    
    synch_2 = ld.load_synch(ells)
    synch = ld.scale_synch(synch_2, nu)
    return dust, synch

def make_fgs(dust, synch):
    fgs = {}
    for ps in dust:
        fgs[ps] = dust[ps] + synch[ps]
    return fgs

def prepare_data(psi0_deg, nu):
    psi0 = psi0_deg * d2r
    cmb_cls, so_noise = prepare_cmb_so(True)

    dust, synch = prepare_foregrounds(cmb_cls['ells'], nu*1e9)
    fgs = make_fgs(dust, synch)
    obs = full_rotate_cls(psi0, cmb_cls, fgs)

    eb_var = calc_eb_var(cmb_cls, so_noise[nu])
    return cmb_cls, so_noise, dust, eb_var, obs

def eb_likelihood(psis, cmb, obs, eb_var):
    eb_like = []
    for psi in psis:
        numerator = obs['EB'] + 0.5 * sin(4.*psi) * (cmb['EE'] - cmb['BB'])
        lnlike = -np.sum(0.5*numerator**2 / eb_var)
        eb_like.append(lnlike)
    return eb_like

def run_self_calibration(psi0_deg, nu=145.):
    cmb_cls, so_noise, dust, eb_var, obs = prepare_data(psi0_deg=psi0_deg, nu=nu)
    psis = np.linspace(psi0_deg-5., psi0_deg+5., 5000) * d2r
    eb_like = eb_likelihood(psis, cmb_cls, obs, eb_var)
    likep = eb_like - np.max(eb_like)
    bias = psis[np.where(likep == np.max(likep))[0]][0]
    y = np.cumsum(np.exp(likep))
    y /= np.max(y)
    try:
        sigma = psis[y>=0.6827][0] - psis[y<=0.3173][-1]
    except:
        sigma=np.inf
    return bias, sigma
    
#x = False
x = True
if x:
    print("\Delta\Psi=0")
    for fnu in SO_freqs:
        bias, sigma = run_self_calibration(0, fnu)
        bias = float("%0.2f" %(bias/d2r))
        sigma = float("%0.2f" %(sigma/d2r))
        print(fnu, bias, sigma)

    print("\Delta\Psi=2")
    for fnu in SO_freqs:
        bias, sigma = run_self_calibration(2., fnu)
        bias = float("%0.2f" %(bias/d2r-2.))
        sigma = float("%0.2f" %(sigma/d2r))
        print(fnu, bias, sigma)

    print("\Delta\Psi=-2")
    for fnu in SO_freqs:
        bias, sigma = run_self_calibration(-2., fnu)
        bias = float("%0.2f" %(bias/d2r+2.))
        sigma = float("%0.2f" %(sigma/d2r))
        print(fnu, bias, sigma)


