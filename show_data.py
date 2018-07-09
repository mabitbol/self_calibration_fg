#! /usr/bin/python

import numpy as np
import pylab as pl

    
def all_cls(ell, cls, name, save):
    xname = '$\ell$'
    pl.figure()
    for cl in cls:
        if cl != "TT":
            pl.plot(ell, cls[cl]*1.e12, label=cl)
    pl.xlabel(xname)
    pl.ylabel("$\ell (\ell +1) C_{\ell}^{%s} / 2\pi [\mu K^2]$" % cl)
    pl.title("%s Power Spectrum" %name)
    pl.legend(fontsize=14)
    savename = "all"+name+save
    pl.savefig(savename)
    pl.close()
    return
        

def all_cls_log(ell, cls, name, save):
    xname = '$\ell$'
    pl.figure()
    for cl in cls:
        if cl != "TT":
            pl.semilogx(ell, cls[cl]*1.e12, label=cl)
    pl.xlabel(xname)
    pl.ylabel("$\ell (\ell +1) C_{\ell}^{%s} / 2\pi [\mu K^2]$" % cl)
    pl.title("%s Power Spectrum" %name)
    pl.legend(fontsize=14)
    savename = "all_log"+name+save
    pl.savefig(savename)
    pl.close()
    return


def plot_cls(ell, cls, name, save):
    xname = '$\ell$'
    for i, cl in enumerate(cls):
        pl.figure(i)
        pl.plot(ell, cls[cl]*1.e12, label=str(cl))
        pl.axhline(y=0)
        pl.xlabel(xname)
        pl.ylabel("$\ell (\ell +1) C_{\ell}^{%s} / 2\pi [\mu K^2]$" % cl)
        pl.title("%s %s Power Spectrum" % (name, cl))
        pl.legend()
        savename = name+save+cl
        pl.savefig(savename)
        pl.close(i)
    return


def plot_rotations(ell, cmb, obs, dust, name):
    xname = '$\ell$'
    for cl in cmb:
        pl.figure()
        pl.plot(ell, cmb[cl], label="rotated without dust")
        pl.plot(ell, obs[cl], '--', label="rotated with 10$\%$ correlation")
        pl.plot(ell, dust[cl], label="dust")
        pl.xlabel(xname)
        pl.ylabel("$\ell (\ell +1) C_{\ell}^{%s} / 2\pi [K^2]$" % cl)
        pl.title("%s Rotated Power Spectrum" % cl)
        pl.legend(fontsize=14)
        savename = name+cl
        pl.savefig(savename)
        pl.close()
    #for cl in cmb:
    #    pl.figure()
    #    pl.loglog(ell, np.abs(cmb[cl]), '--', label="cmb")
    #    pl.loglog(ell, np.abs(obs[cl]), '.-', label="rotated")
    #    pl.loglog(ell, np.abs(dust[cl]), label="dust")
    #    pl.xlabel(xname)
    #    pl.ylabel("$\ell (\ell +1) C_{\ell}^{%s} / 2\pi [K^2]$" % cl)
    #    pl.title("%s Rotated Power Spectrum" % cl)
    #    pl.legend(fontsize=15)
    #    savename = name+"log"+cl
    #    pl.savefig(savename)
    #   pl.close()
    return


def plot_dust_cls(ell, dust, name, nobars=True, errors=1):
    xname = '$\ell$'
    yname = '$\ell (\ell+1) C_{\ell}/ 2\pi (\mu K^2)$'
    if nobars:
        for i, cl in enumerate(dust):
            pl.figure(i)
            pl.plot(ell, dust[cl]*1.e12, label=str(cl))
            pl.xlabel(xname)
            pl.ylabel("$\ell (\ell +1)C_{\ell}^{%s} / 2\pi [\mu K^2]$" % cl)
            pl.title("Dust %s Power Spectrum" % cl)
            pl.legend()
            pl.grid()
            savename = name+cl
            pl.savefig(savename)
            pl.close(i)
    else:
        for i, cl in enumerate(dust):
            pl.figure(i)
#            pl.xscale("log")
            pl.errorbar(ell, dust[cl], yerr=errors[cl], fmt='o', color='g', label=str(cl))
            pl.axhline(y=0, color='k')
            pl.xlabel(xname)
            pl.ylabel("$\ell (\ell +1)C_{\ell}^{%s} / 2\pi [K^2]$" % cl)
            pl.title("Dust %s Power Spectrum" % cl)
            pl.legend()
            pl.grid()
            savename = name+cl
            pl.savefig(savename)
            pl.close(i)
    return


def plot_fit(ell, EE, BB, TE, dEE, dBB, dTE, l, fitEE, fitBB, fitTE, fitEEerr, fitBBerr, fitTEerr):
    pl.figure(1)
    #pl.xscale("log")
    #pl.yscale("log")
    pl.errorbar(ell, EE, yerr=dEE, fmt='o', color='r', label='EE')
    pl.errorbar(l, fitEE, yerr=fitEEerr, color='b', label='fit')
    #pl.plot(l, fitEE, color='b', label='fit')
    pl.axhline(y=0, color='k')
    pl.xlabel("$\ell$")
    pl.ylabel("$\ell (\ell +1)C_{\ell}^{EE} / 2\pi [K^2]$")
    pl.title("Dust EE Power Spectrum")
    pl.legend()
    pl.grid()
    savename = "EE_fit"
    pl.savefig(savename)
    pl.close(1)

    pl.figure(2)
    #pl.xscale("log")
    pl.errorbar(ell, BB, yerr=dBB, fmt='o', color='r', label='BB')
    pl.errorbar(l, fitBB, yerr=fitBBerr, color='b', label='fit')
    #pl.plot(l, fitBB, color='b', label='fit')
    pl.axhline(y=0, color='k')
    pl.xlabel("$\ell$")
    pl.ylabel("$\ell (\ell +1)C_{\ell}^{BB} / 2\pi [K^2]$")
    pl.title("Dust BB Power Spectrum")
    pl.legend()
    pl.grid()
    savename = "BB_fit"
    pl.savefig(savename)
    pl.close(2)

    pl.figure(2)
    #pl.xscale("log")
    pl.errorbar(ell, TE, yerr=dTE, fmt='o', color='r', label='TE')
    pl.errorbar(l, fitTE, yerr=fitTEerr, color='b', label='fit')
    #pl.plot(l, fitTE, color='b', label='fit')
    pl.axhline(y=0, color='k')
    pl.xlabel("$\ell$")
    pl.ylabel("$\ell (\ell +1)C_{\ell}^{TE} / 2\pi [K^2]$")
    pl.title("Dust TE Power Spectrum")
    pl.legend()
    pl.grid()
    savename = "TE_fit"
    pl.savefig(savename)
    pl.close(2)
    return


