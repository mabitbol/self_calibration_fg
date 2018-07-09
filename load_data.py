#! /usr/bin/python

import numpy as np
from numpy import sin, cos, arcsin, exp
import healpy as hp
from math import pi
from astropy.io import fits
from scipy import optimize
from show_data import *

r2d = 180./pi
d2r = pi/180.
arcmin2d = 1./60.
arcmin2r = pi/(180.*60.)
qnames =['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
[TT, EE, BB, TE, TB, EB] = qnames


def load_cls(show_cls=False, kelvin=True):
    fname = "data/camb_lens_nobb.dat"
    with open(fname) as f:
        clsfile = list(f)
    N = len(clsfile)
    data = np.zeros([N,5])
    for i in range(N):
        data[i] = clsfile[i].split()
    data = np.array(data)
    ell = data[:,0]
    if kelvin:
        data = data * 1.e-12
    cmb = dict(zip(qnames[:4], data[:,1:].T))
    cmb['TB'] = np.zeros(N)
    cmb['EB'] = np.zeros(N)
    if show_cls:
        plot_cls(ell, cmb, "camb_loadcls")
    return ell, cmb


def load_bb(kelvin=True):
    fname = "data/camb_lens_r1.dat"
    with open(fname) as f:
        clsfile = list(f)
    N = len(clsfile)
    data = np.zeros([N,5])
    for i in range(N):
        data[i] = clsfile[i].split()
    data = np.array(data)
    ell = data[:,0]
    if kelvin:
        data = data * 1.e-12
    cmb = dict(zip(qnames[:4], data[:,1:].T))
    return cmb[BB]


def load_colin_dust(show_cls=False):
    fname = "data/dustcls.txt"
    binbounds = [40, 120, 250, 400, 1000]
    with open(fname) as f:
        dustfile = list(f)
    data = np.zeros([13,4])
    for i in range(len(dustfile)):
        data[:,i] = dustfile[i].split()
    ell = data[0]
    dust_raw = dict(zip(qnames, data[1:7]))
    errors_raw = dict(zip(qnames, data[7:]))
    #uk = 1.e12
    #print "raw EB ", dust_raw[EB][0]*uk, errors_raw[EB][0]*uk
    #print "raw TB ", dust_raw[TB][0]*uk, errors_raw[TB][0]*uk
    if show_cls:
        plot_dust_cls(ell, dust_raw, name="rawdust", nobars=False, errors=errors_raw)
    return ell, dust_raw, errors_raw


def load_extrap_dust(l, show_cls=False):
    ell, dust_raw, errors_raw = load_colin_dust(show_cls)
    return extrapolate(ell, dust_raw, errors_raw, l)


def extrapolate(ell, dust, errors,  l):
    dustnew = {}
    errorsnew = {}
    for k in dust.keys():
        Amp, sig = optimize.curve_fit(dustfit, ell, dust[k], p0=dust[k][0], \
                                        sigma=errors[k], absolute_sigma=True)
        err = np.sqrt(sig[0][0])
        dustnew[k] = dustfit(l, Amp)
        errorsnew[k] = dustfit(l, err)
    return dustnew, errorsnew
     

def dustfit(ell, A):
    return A*np.power(ell/80., -0.42)


def generate_names(num_curves, sky_params, dust_params, name):
    title_name = " Likelihood Curves"
    save_name = name+"varied_"
    label_names = []
    for key, val in sky_params.items():
        if type(val) == list:
            save_name += key
            if len(label_names) == 0:
                for i in range(num_curves):
                    lns = key+"="+str(val[i])
                    if key == "beam":
                        lns = "$\Theta=%s^{\prime}$"%str(int(val[i]/arcmin2r))
                    if key == "fsky":
                        continue
                        lns = "$f_{sky}=$"+str(val[i])
                    label_names.append(lns)
            else:
                for i in range(num_curves):
                    lns = key+"="+str(val[i])
                    if key == "beam":
                        lns = "$\Theta=%s^{\prime}$"%str(int(val[i]))
                    if key == "fsky":
                        continue
                        lns = "$f_{sky}=$"+str(val[i])
                    label_names[i] += ", "+lns
    for key, val in dust_params.items():
        if type(val) == list:
            save_name += key
            if len(label_names) == 0:
                for i in range(num_curves):
                    lns = key+"="+str(val[i])
                    if key == "frac":
                        lns = "$f_c=$"+str(val[i])
                    if key == "level":
                        lns = "$m=%s$"%str(val[i])
                    label_names.append(lns)
            else:
                for i in range(num_curves):
                    lns = key+"="+str(val[i])
                    if key == "frac":
                        lns = "$f_c=$"+str(val[i])
                    if key == "level":
                        lns = "$m=%s$"%str(val[i])
                    label_names[i] += ", "+lns
    save_name += str(num_curves)
    return [title_name, save_name, label_names]


def make_params_list(num, sky_params, dust_params):
    params_list = []
    params = {}
    params.update(sky_params)
    params.update(dust_params)
    for j in range(num):
        tmp = {}
        tmp.update(params)
        for k in params:
            if type(params[k])==list:
                tmp[k] = tmp[k][j]
        params_list.append(tmp)
    return params_list


def frac_dust(dust, errors, frac):
    dust[EB] = frac*np.sqrt(dust[EE]*dust[BB])
    dust[TB] = frac*np.sqrt(dust[TT]*dust[BB])
    errors[EB] = np.sqrt((dust[BB]*errors[EE])**2+(dust[EE]*errors[BB])**2)
    errors[EB] = 0.5*(frac**2)*errors[EB]/dust[EB]
    errors[TB] = np.sqrt((dust[BB]*errors[TT])**2+(dust[TT]*errors[BB])**2)
    errors[TB] = 0.5*(frac**2)*errors[TB]/dust[TB]
    #errors[TB] = frac*np.sqrt(errors['TT']*errors['BB'])
    return dust, errors


def mult_cls(data, level):
    for k in data.keys():
        data[k] = data[k]*level
    return data


def truncate(data, lmax):
    for i in range(len(data)):
        for k in data[i].keys():
            data[i][k] = data[i][k][:lmax]
    return data



##### Test functions #####

def testcls():
    names =['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
    [TT, EE, BB, TE, TB, EB] = names
    xname = '$\ell$'
#    yname = '$\ell (\ell+1) C_{\ell}/ 2\pi [\mu K^2]$'
    yname = '$[\ell (\ell+1) C_{\ell}/ 2\pi]^{1/2} [\mu K]$'

    ell, cmb = load_cls()
    #plot_cls(ell, cmb, "primordial")
    obs, dust, errors = rotate_dust(ell, cmb, 2., True, 0,0,False)
    obs2, dust, errors = rotate_dust(ell, cmb, 2., False,0,0,False)
#    dust = load_colin_dust(ell, False)
    #plot_dust_cls(ell, dust, "dust")

    beam = 5./(60.)*pi/180.
    window = (hp.sphtfunc.gauss_beam(beam, lmax=ell[-1])**2)[2:]
    nl = 5      #microK arcmin
    noise = ((ell*(ell+1)/(2*pi))*((pi/10800.)*nl)**2)
    #nl2 = 10      #microK arcmin
    #noise2 = ((ell*(ell+1)/(2*pi))*((pi/10800.)*nl2)**2)*1e-12

    pl.loglog(ell, np.sqrt(cmb['EE']*window*1.e12), label="Primordial EE")
    pl.loglog(ell, np.sqrt(cmb['BB']*window*1.e12), label="Lensing BB, r=0")
    pl.loglog(ell, np.sqrt(noise), label="noise at %s $\mu$K arcmin" % str(nl))
    pl.loglog(ell, np.sqrt(obs['BB']*window*1.e12+noise), label="Rotated BB, $\psi=2.0^{\circ}$")
    pl.loglog(ell, np.sqrt(obs2['BB']*window*1.e12+noise), label="Rotated BB with dust")
    #pl.loglog(ell, (noise2), label="noise at %s $\mu$K arcmin" % str(nl2))
    #pl.loglog(ell, np.sqrt(dust['EE']*window*1.e12), label="Dust EE")
    pl.loglog(ell, np.sqrt(dust['BB']*window*1.e12), label="Dust BB")
    pl.xlim([2, ell[-1]])
    pl.xlabel(xname)
    pl.ylabel(yname)
    pl.grid()
    #pl.title("Power Spectra Amplitudes")
    pl.legend(loc=4, fontsize=12)
    sname = "fig2_sqrt_rotate_dustplot.eps"
    pl.savefig(sname, format='eps')
    return 


#testcls()
