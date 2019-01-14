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

def prepare_cmb_so():
    cmb_cls = ld.load_cmb()
    so_ell, so_noise = ld.load_SO_noise()
    cmb_cls = ld.truncate(cmb_cls, lmin=so_ell.min(), lmax=so_ell.max())
    assert np.all(so_ell == cmb_cls['ells'])
    return cmb_cls, so_noise

def calc_eb_var(cmb, so_noise, fsky=0.1):
    C_EE_tot = cmb['EE'] + so_noise
    C_BB_tot = cmb['BB'] + so_noise
    delta_EB_var = 2. / (2. * cmb['ells'] + 1.) * C_EE_tot * C_BB_tot / fsky
    return delta_EB_var

def prepare_foregrounds(ells, nu):
    dust_353 = ld.load_dust(ells)
    dust = ld.scale_dust(dust_353, nu)
    
    # load synch 
    # scale synch
    return dust

def prepare_data(psi0_deg=2., nu=145.):
    psi0 = psi0_deg * d2r
    cmb_cls, so_noise = prepare_cmb_so()
    dust = prepare_foregrounds(cmb_cls['ells'], nu*1e9)
    obs = full_rotate_cls(psi0, cmb_cls, dust)
    eb_var = calc_eb_var(obs, so_noise[nu], fsky=0.1)
    return cmb_cls, so_noise, dust, eb_var, obs

def eb_likelihood(psis, cmb, obs, eb_var):
    eb_like = []
    for psi in psis:
        numerator = obs['EB'] + 0.5 * sin(4.*psi) * (cmb['EE'] - cmb['BB'])
        lnlike = -np.sum(numerator**2 / eb_var)
        eb_like.append(lnlike)
    return eb_like

def run_self_calibration(psi0_deg, nu=145.):
    cmb_cls, so_noise, dust, eb_var, obs = prepare_data(psi0_deg=psi0_deg, nu=145.)
    psis = np.linspace(psi0_deg-.5, psi0_deg+.5, 1000) * d2r
    eb_like = eb_likelihood(psis, cmb_cls, obs, eb_var)
    likep = eb_like - np.max(eb_like)
    bias = psis[np.where(likep == np.max(likep))[0]][0]
    y = np.cumsum(np.exp(likep))
    y /= np.max(y)
    sigma = psis[y>=0.6827][0] - psis[y<=0.3173][-1]
    return bias, sigma
    
bias, sigma = run_self_calibration(0)
print(bias/d2r, sigma/d2r)

