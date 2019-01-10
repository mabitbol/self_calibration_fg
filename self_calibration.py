import numpy as np

import load_data as ld

[TT, EE, BB, TE, TB, EB] = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']

def full_rotate_cls(psi, cmb):
    obs={} 
    obs[TT] = cmb[TT]+dust[TT]
    obs[EE] = (sin(2*psi)**2)*(cmb[BB]+dust[BB]) + (cos(2*psi)**2)*(cmb[EE]+dust[EE]) + sin(4*psi)*dust[EB]
    obs[BB] = (cos(2*psi)**2)*(cmb[BB]+dust[BB]) + (sin(2*psi)**2)*(cmb[EE]+dust[EE]) - sin(4*psi)*dust[EB]
    obs[TE] = cos(2*psi)*(cmb[TE]+dust[TE]) + sin(2*psi)*dust[TB]
    obs[TB] = -sin(2*psi)*(cmb[TE]+dust[TE]) + cos(2*psi)*dust[TB]
    obs[EB]  = 0.5*sin(4*psi)*(cmb[BB]-cmb[EE]+dust[BB]-dust[EE]) + cos(4*psi)*dust[EB]
    return 

def fast_rotate(psis, cmb):
    rot_EB = 0.5*sin(4*psi)*(cmb[BB]-cmb[EE]+dust[BB]-dust[EE]) + cos(4*psi)*dust[EB]
    


def prepare_cmb_SO(nu):
    cmb_cls = ld.load_data()
    so_ell, so_noise = ld.load_SO_noise()
    cmb_cls = ld.truncate(cmb_cls, lmin=so_ell.min(), lmax=so_ell.max())
    assert (so_ell == cmb_cls['ell'])
    C_EE_tot = cmb_cls['EE'] + so_noise[nu]
    C_BB_tot = cmb_cls['BB'] + so_noise[nu]
    delta_EB_var = 2. / (2. * so_ell + 1.) * C_EE_tot * C_BB_tot / fsky
    return cmb_cls, delta_EB_var

def prepare_foregrounds(nu):
    dust_353 = ld.load_dust(cmb_cls['ell'])
    dust = ld.scale_dust(dust_353, nu)
    
    # load synch 
    # scale synch
    return dust



    
