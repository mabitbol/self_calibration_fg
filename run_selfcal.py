#! /usr/bin/python

import numpy as np
import matplotlib as mpl
from matplotlib import cm
#mpl.use('gtkagg')
mpl.use('agg')
from matplotlib import rcParams
rcParams['ps.fonttype'] = 42
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
import pylab as pl
import seaborn as sns
sns.set(style="whitegrid")
sns.set_context("notebook", font_scale=2.0, rc={"lines.linewidth":1.0})
sns.axes_style(rc={"axes.edgecolor":'black'})
pl.matplotlib.rc('font', size=40)
pl.matplotlib.rc('figure', autolayout=True)
pl.matplotlib.rc('axes', edgecolor='0.57')
pl.matplotlib.rc('grid', color='0.57')
#import healpy as hp
from math import pi
from numpy import sin, cos, arcsin, exp
from scipy import optimize, integrate
from old_load_data import *
from show_data import *
from fast_calc import fast_resid_eb, fast_resid_tb

r2d = 180./pi
d2r = pi/180. 
arcmin2d = 1./60.
arcmin2r = pi/(180.*60.)
quicknames = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
[TT, EE, BB, TE, TB, EB] = quicknames


def plot_likeli(psi_a, L, name, plot_names, params_list):
    title_name, save_name, label_names = plot_names
    N = len(params_list)
    w = 0.4
    psi = params_list[0]["psi"]*r2d
    psi_ad = psi_a*r2d
    blue = cm.get_cmap('Blues')
    norm = mpl.colors.Normalize(-1, N)
    scalar = cm.ScalarMappable(norm=norm, cmap=blue)
    red = cm.get_cmap('Reds')
    norm = mpl.colors.Normalize(-1, N)
    scalar1 = cm.ScalarMappable(norm=norm, cmap=red)
    pl.figure()
    print(name, save_name)
    for i in range(N):
        #if i==0:
        #    c = 'r'
        #    label_names[i] = "clean sky"
        #else:
        #    c = scalar.to_rgba(i)
        #elif i==1:
        #    c = 'b'
        #    label_names[i] = "dusty sky"
        #elif i==2:
        #    c = 'g'
        #    label_names[i] = "corrected Likelihood"
        if i<3:
            c = scalar1.to_rgba(i)
        else:
            c = scalar.to_rgba(i)
        #if i==0 or i==1:
        #    c = 'r'
        #else:
        #    c = 'b'
        #if i==1 or i==2:
        #    fmt = '--'
        #else:
        #    fmt = '-'
        #pl.plot(psi_ad, L[i], color=c, linestyle=fmt, label=label_names[i])
        pl.plot(psi_ad, L[i], color=c, label=label_names[i])
        print(label_names[i] )
        print("peak ", psi_ad[L[i]==max(L[i])][0] * 60.)
        print("sigma ", calc_sigma(psi_ad, L[i]) * 60.)
        
    pl.axvline(x=psi, color='k', label="Input angle")
    #pl.xlim([psi-w, psi+w])
    pl.ylim([-0.05, 1.05])
    #pl.xlim([psi-1.5, psi+1.5])
    #pl.xlim([psi-.5, psi+1.5])
    pl.xlim([psi-.4, psi+.6])
    pl.xlabel("$\Delta\psi$ [deg]")
    pl.ylabel("Normalized "+name+" Likelihood")
    #pl.title(name+title_name)
    pl.legend(loc=1, fontsize=20)
    savename = "L"+name+save_name+".eps"
    #savename = "L"+name+save_name
    pl.savefig(savename, format='eps')
    #pl.savefig(savename)
    pl.close()
    return

def calc_sigma(x, L):
    xmean = np.where(L == max(L))[0][0]
    L_lhs = L[:xmean+1]
    cdf_lhs = np.cumsum(L_lhs)
    cdf_lhs /= np.max(cdf_lhs)
    m_lhs = np.where(cdf_lhs >= 0.3173)[0][0]
    sigma_lhs = x[xmean] - x[m_lhs]

    L_rhs = L[xmean:]
    cdf_rhs = np.cumsum(L_rhs)
    cdf_rhs /= np.max(cdf_rhs)
    m_rhs = np.where(cdf_rhs >= 0.6827)[0][0]
    sigma_rhs = x[xmean+m_rhs] - x[xmean]
    return (sigma_rhs + sigma_lhs)/2.

def make_data(plots, sky_params, dust_params, num_curves, name):
    plot_names = generate_names(num_curves, sky_params, dust_params, name)
    params_list = make_params_list(num_curves, sky_params, dust_params)
    make_likelihoods(num_curves, plots, plot_names, params_list)
    return

def make_likelihoods(num_curves, plots, plot_names, params_list):
    title_name, save_name, label_names = plot_names
    num = 1000
    #bounds = [-pi/800, pi/800]
    psi = params_list[0]["psi"]
    w = pi/200
    bounds = [psi-w, psi+w]
    psi_a = np.linspace(bounds[0], bounds[1], endpoint = True, num=num)  
    print("step size: ", psi_a[1] - psi_a[0])
    Ltb = []
    Leb = []
    for k in range(num_curves):
        if params_list[k]["use_full"] == 0:
            Ltb_k, Leb_k = calculate_likelihood(plots, save_name+str(k), \
                                                psi_a, **params_list[k])
        else:
            Ltb_k, Leb_k = calculate_full_likelihood(plots, save_name+str(k), \
                                                      psi_a, **params_list[k])
        Ltb.append(Ltb_k)
        Leb.append(Leb_k)
    #np.save("sLEBd", Leb)
    #np.save("sLTBd", Ltb)
    #np.save("spsia", psi_a)
    plot_likeli(psi_a, Leb, "EB", plot_names, params_list)
    plot_likeli(psi_a, Ltb, "TB", plot_names, params_list)
    return

def calculate_likelihood(plots, names, psi_a, psi, fsky, lmax, beam, \
                            nl, nodust, frac, level, frac_on, use_full):
    #check if/when sine wraps
    ell, cmb = load_cls(False)
    obs, dust, errors = rotate_dust(ell, cmb, psi, nodust, frac, level, \
                                    frac_on, plots)
    if plots:
        plot_everything(ell, cmb, dust, obs, names)
    cmb, obs, dust = truncate([cmb, obs, dust], lmax)
    ell = ell[:lmax]
    TB_list, EB_list = setup_calc_naive(ell, cmb, obs, dust, fsky, beam, nl)
    Ltb = fast_resid_tb(psi_a, *TB_list)
    Leb = fast_resid_eb(psi_a, *EB_list)
    Ltb = np.exp(Ltb)
    Ltb /= max(Ltb) 
    Leb = np.exp(Leb)
    Leb /= max(Leb) 
    return Ltb, Leb

def calculate_full_likelihood(plots, names, psi_a, psi, fsky, lmax, beam, \
                            nl, nodust, frac, level, frac_on, use_full):
    ell, cmb = load_cls()
    obs, dust, errors = rotate_dust(ell, cmb, psi, nodust, frac, level, \
                                    frac_on, plots)
    if plots:
        plot_everything(ell, cmb, dust, obs, names)
    cmb, obs, dust, errors = truncate([cmb, obs, dust, errors], lmax)
    ell = ell[:lmax]
    unit, noise = setup_full(ell, fsky, beam, nl)
    return full_likeli(psi_a, ell, cmb, obs, dust, errors, unit, noise)

def setup_calc_naive(ell, cmb, obs, dust, fsky, beam, nl):
    unit, noise = setup_full(ell, fsky, beam, nl)
    #TB 
    A = cmb[TE]
    C = obs[TB]
    dtb = unit*((obs[TT]+noise)*(obs[BB]+noise))
    #EB
    X = cmb[EE]-cmb[BB]
    Y = obs[EB]
    deb = unit*((obs[EE]+noise)*(obs[BB]+noise))
    return [A,C,dtb], [X,Y,deb]

def setup_full(ell, fsky, beam, nl):
    #window = (hp.sphtfunc.gauss_beam(beam, lmax=ell[-1])**2)[2:]
    #noise = (((ell*(ell+1)/(2*pi))*((pi/10800.)*nl)**2)*1e-12)/window

    noise = np.loadtxt('data/N_ell_SA_Pol_Basel_lkneepess_1yrsLF_145GHz.txt')[:, 1]
    noise *= (ell*(ell+1)/(2*pi)) * 1e-12

    #noise = np.loadtxt('data/N_ell_SA_Pol_Goal_lkneeopt_1yrsLF_145GHz.txt')[:, 1]
    #noise *= (ell*(ell+1)/(2*pi)) *1e-12

    unit = 1./((2*ell+1)*fsky)
    return unit, noise

def rotate_dust(ell, cmb, psi, nodust, frac, level, frac_on, plot=False):
    #psi = psi*d2r
    N = len(cmb[TT])
    if nodust:
        dust = {}
        errors = {}
        for k in quicknames:
            dust[k] = np.zeros(N)
            errors[k] = np.zeros(N)
    else:
        dust, errors = load_extrap_dust(ell, False)
        if level>0.:
            dust = mult_cls(dust, level)
            errors = mult_cls(errors, level)
        if frac_on:
            dust, errors = frac_dust(dust, errors, frac)
    obs={} 
    obs[TT] = cmb[TT]+dust[TT]
    obs[EE] = (sin(2*psi)**2)*(cmb[BB]+dust[BB]) + (cos(2*psi)**2)*(cmb[EE]+dust[EE]) + sin(4*psi)*dust[EB]
    obs[BB] = (cos(2*psi)**2)*(cmb[BB]+dust[BB]) + (sin(2*psi)**2)*(cmb[EE]+dust[EE]) - sin(4*psi)*dust[EB]
    obs[TE] = cos(2*psi)*(cmb[TE]+dust[TE]) + sin(2*psi)*dust[TB]
    obs[TB] = -sin(2*psi)*(cmb[TE]+dust[TE]) + cos(2*psi)*dust[TB]
    obs[EB]  = 0.5*sin(4*psi)*(cmb[BB]-cmb[EE]+dust[BB]-dust[EE]) + cos(4*psi)*dust[EB]
    return obs, dust, errors

def full_likeli(psi_a, ell, cmb, obs ,dust, errors, unit, noise):
    sigma = {}
    amp = {}
    for k in dust.keys():
        amp[k] = (dust[k]*np.power(ell/80., 0.42))[1]
        sigma[k] = (errors[k]*np.power(ell/80., 0.42))[1]
    lower, upper = bounds(amp, sigma)
    N = len(psi_a)
    Leb = np.zeros(N)
    Ltb = np.zeros(N)
    deb = unit*(obs[EE]+noise)*(obs[BB]+noise)
    dtb = unit*(obs[TT]+noise)*(obs[BB]+noise)
    X = (cmb[EE]-cmb[BB])+(dust[EE]-dust[BB])
    Y = obs[EB]
    #Z = dust[EB]
    A = cmb[TE]+dust[TE]
    C = obs[TB]
    #D = dust[TB]
    for i in range(N):
        p = psi_a[i]
        Leb[i],err = integrate.quad(sloweb_exp,lower[EB],upper[EB],\
                                    args=(p,ell,X,Y,amp[EB],sigma[EB],deb))
        Ltb[i],err1 = integrate.quad(slowtb_exp,lower[TB],upper[TB],\
                                    args=(p,ell,A,C,amp[TB],sigma[TB],dtb))
        #Leb[i],err = integrate.nquad(full_eb_exp,[[lower[EE],upper[EE]],\
        #                        [lower[BB],upper[BB]],[lower[EB],upper[EB]]],\
        #                            args=(p,ell,cmb,obs,amp,sigma,deb))
        #Ltb[i],err1 = integrate.nquad(full_tb_exp,[[lower[TE],upper[TE]],\
        #                            [lower[TB],upper[TB]]],\
        #                            args=(p,ell,cmb,obs,amp,sigma,dtb))
    Leb /= max(Leb)
    Ltb /= max(Ltb)
    print(psi_a[Ltb==max(Ltb)][0]*r2d)
    print(psi_a[Leb==max(Leb)][0]*r2d)
    return Ltb, Leb

def full_tb_exp(fte, ftb, psi, ell, cmb, obs, amp, sigma, dtb):
    A = cmb[TE]+dustfit(ell,fte)
    C = obs[TB]
    D = dustfit(ell,ftb)
    x = C + A*np.sin(2*psi) - D*np.cos(2*psi)
    g1 = -np.sum((x*x)/(2*dtb))
    g2 = -(fte-amp[TE])**2/(2*sigma[TE]**2)
    g3 = -(ftb-amp[TB])**2/(2*sigma[TB]**2)
    return np.exp(g1+g2+g3)

def full_eb_exp(fee, fbb, feb, psi, ell, cmb, obs, amp, sigma, deb):
    X = (cmb[EE]-cmb[BB])+(dustfit(ell,fee)-dustfit(ell,fbb))
    Y = obs[EB]
    Z = dustfit(ell,feb)
    x = Y + 0.5*X*np.sin(4*psi) - Z*np.cos(4*psi)
    g1 = -np.sum((x*x)/(2*deb))
    g2 = -(fee-amp[EE])**2/(2*sigma[EE]**2)
    g3 = -(fbb-amp[BB])**2/(2*sigma[BB]**2)
    g4 = -(feb-amp[EB])**2/(2*sigma[EB]**2)
    return np.exp(g1+g2+g3+g4)

def bounds(amp, sigma):
    upper = {}
    lower = {}
    for k in amp.keys():
        upper[k] = amp[k] + 5.*sigma[k] 
        lower[k] = amp[k] - 5.*sigma[k] 
    return lower, upper 

def sloweb_exp(feb, psi, ell, X, Y, ampeb, erreb, deb):
    Zf = dustfit(ell,feb)
    x = Y + 0.5*X*np.sin(4*psi) - Zf*np.cos(4*psi)
    g1 = -np.sum((x*x)/(2*deb))
    g4 = -(feb-ampeb)**2/(2*erreb**2)
    return np.exp(g1+g4)

def slowtb_exp(ftb, psi, ell, A, C, amptb, errtb, dtb):
    Df = dustfit(ell,ftb)
    x = C + A*np.sin(2*psi) - Df*np.cos(2*psi)
    g1 = -np.sum((x*x)/(2*dtb))
    g4 = -(ftb-amptb)**2/(2*errtb**2)
    return np.exp(g1+g4)


######################################################################
def foo2():
    bins = [40, 120, 250, 400, 1000]
    xname = '$\ell$'
    yname = '$\ell (\ell+1) C^{XB}_{\ell}/ 2\pi [\mu K^2]$'
    ell, cmb = load_cls()
    l, dustraw, errorsraw = load_colin_dust(False)
    dustfit, errorsfit = load_extrap_dust(ell, False)
    uk = 1.e12
    f, ax = pl.subplots()
    #ax.semilogx(ell, dustfit[EB]*uk, color='b', label='$EB$ Fit')
    ax.plot(ell, dustfit[EB]*uk, color='b', label='$EB$ Fit')
    ax.fill_between(ell, dustfit[EB]*uk-errorsfit[EB]*uk, \
                dustfit[EB]*uk+errorsfit[EB]*uk, facecolor='b', alpha=0.2)
    ax.errorbar(l, dustraw[EB]*uk, yerr=errorsraw[EB]*uk, fmt='o', color='b', \
                label='Measured $EB$')
    #ax.semilogx(ell, dustfit[TB]*uk, color='g', label='$TB$ Fit')
    ax.plot(ell, dustfit[TB]*uk, color='g', label='$TB$ Fit')
    ax.fill_between(ell, dustfit[TB]*uk-errorsfit[TB]*uk, \
                dustfit[TB]*uk+errorsfit[TB]*uk, facecolor='g', alpha=0.2)
    ax.errorbar(l+8, dustraw[TB]*uk, yerr=errorsraw[TB]*uk, fmt='^', color='g', \
                label='Measured $TB$')
    ax.set_xlim(0, 1000)
    #ax.axhline(y=0,color='k')
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.legend(fontsize=12, loc=4)
    #ax.grid()
    #ax.set_title("Measured and Fit TB")
    sname = 'EBTBspec.pdf'
    #pl.show()
    pl.savefig(sname, format='pdf')
    return

def foo1():
    bins = [40, 120, 250, 400, 1000]
    xname = '$\ell$'
    yname = '$\ell (\ell+1) C^{TB}_{\ell}/ 2\pi [\mu K^2]$'
    ell, cmb = load_cls()
    l, dustraw, errorsraw = load_colin_dust(False)
    dustfit, errorsfit = load_extrap_dust(ell, False)
    uk = 1.e12
    f, ax = pl.subplots()
    ax.semilogx(ell, dustfit[TB]*uk, color='b', label='Fit')
    ax.fill_between(ell, dustfit[TB]*uk-errorsfit[TB]*uk, dustfit[TB]*uk+errorsfit[TB]*uk)
    ax.errorbar(l, dustraw[TB]*uk, yerr=errorsraw[TB]*uk, fmt='o', color='r', \
                label='Measured')
    frac = [.1, .2, .5]
    dust = mult_cls(dustfit, 5.*uk)
    for i in range(3):
        dust, errors = frac_dust(dust, errorsfit, frac[i])
        l = 'level=5x, $f_c$='+str(frac[i])
        ax.semilogx(ell, dust[TB], color='g', label=l)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.legend(fontsize=12)
    #ax.grid()
    #ax.set_title("Measured and Fit TB")
    sname = 'TBspecfrac.eps'
    pl.savefig(sname, format='eps')
    #pl.show()
    return

def foo():
    bins = [40, 120, 250, 400, 1000]
    xname = '$\ell$'
    yname = '$\ell (\ell+1) C^{EB}_{\ell}/ 2\pi [\mu K^2]$'
    ell, cmb = load_cls()
    l, dustraw, errorsraw = load_colin_dust(False)
    dustfit, errorsfit = load_extrap_dust(ell, False)
    uk = 1.e12
    f, ax = pl.subplots()
    ax.semilogx(ell, dustfit[EB]*uk, color='b', label='Fit')
    ax.fill_between(ell, dustfit[EB]*uk-errorsfit[EB]*uk, dustfit[EB]*uk+errorsfit[EB]*uk)
    ax.errorbar(l, dustraw[EB]*uk, yerr=errorsraw[EB]*uk, fmt='o', color='r', \
                label='Measured')
    frac = [.1, .2, .5]
    dust = mult_cls(dustfit, 5.*uk)
    for i in range(3):
        dust, errors = frac_dust(dust, errorsfit, frac[i])
        l = 'level=5x, $f_c$='+str(frac[i])
        ax.semilogx(ell, dust[EB], color='g', label=l)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.legend(fontsize=12)
    #ax.grid(both=True)
    #ax.set_title("Measured and Fit EB")
    sname = 'EBspecfrac.eps'
    pl.savefig(sname, format='eps')
    return

def plot_everything(ell, cmb, dust, obs, names):
    #plot_rotations(ell, obs, obs2, dust, "psi2wwo1fracEBTB")
    print(names)
    all_cls(ell, cmb, "cmb", names) 
    all_cls(ell, dust, "dust", names) 
    all_cls(ell, obs, "rotated", names) 
    all_cls_log(ell, cmb, "cmb", names) 
    all_cls_log(ell, dust, "dust", names) 
    all_cls_log(ell, obs, "rotated", names) 
    plot_cls(ell, cmb, "cmb", names)
    plot_cls(ell, obs, "rotated", names)
    plot_cls(ell, dust, "dust", names)
    return 

def fit_r():
    xname = '$\ell$'
    yname = '$\ell (\ell+1) C^{BB}_{\ell}/ 2\pi [\mu K^2]$'
    no_lens = False
    fsky = 0.01
    beam = 5*arcmin2r
    lmax = 200
    nl = 5.
    n = 3
    psis = np.array([.5, 1., 2.])*d2r
    r = np.zeros(n)
    ell, cmb = load_cls()
    bb = load_bb()
    #no lensing
    if no_lens:
        bb = bb-cmb[BB]
        cmb[BB] = np.zeros(len(ell))
    unit, noise = setup_full(ell, fsky, beam, nl)
    #bb = bb[:lmax]
    #unit = unit[:lmax]
    #noise = noise[:lmax]
    uk = 1.e12
    plotbb = 0.01*(bb-cmb[BB])+cmb[BB]
    pl.loglog(ell, plotbb*uk, label="Theoretical: $r=0.01$")
    for i in range(n):
        obs, dust, errors = rotate_dust(ell, cmb, psis[i], True, 0, 0, 0)
        pl.loglog(ell, obs[BB]*uk, label="Observed: $r=0.0$, $\Delta\psi=%s^{\circ}$"\
               % str(round(psis[i]*r2d,2)))
        #obsbb = obs[BB][:lmax]
        #dobs = 2*unit*(obsbb+noise)*(obsbb+noise)
        #r[i] = sum(obsbb*bb/dobs)/sum(bb*bb/dobs)
        r[i] = obs[BB][80]/bb[80]
        #r[i] = sum(obsbb)/sum(bb)
    pl.xlim([2, ell[-1]])
    pl.xlabel(xname)
    pl.ylabel(yname)
    pl.legend(loc=2, fontsize=12)
    sname = "spuriousB.eps"
    pl.savefig(sname, format='eps')
    #pl.show()
    #print psis*r2d
    #print r
    return 

def testcls2():
    uk = 1.e12
    level = [1, 5, 10]
    frac = [.1, .2, .5]

    ell, cmb = load_cls()
    #print ell[78]
    l, dustraw, errorsraw = load_colin_dust(False)
    #print l[0]
    #print "measured EB ", dustraw[EB][0]*uk, errorsraw[EB][0]*uk
    #print "measured TB ", dustraw[TB][0]*uk, errorsraw[TB][0]*uk
    dustfit, errorsfit = load_extrap_dust(ell, False)
    dust = mult_cls(dustfit, uk)
    errors = mult_cls(errorsfit, uk)
    #print "level1 EB ", dust[EB][78], errors[EB][78]
    #print "level1 TB ", dust[TB][78], errors[TB][78]
    dust = mult_cls(dust, 5.)
    errors = mult_cls(errors, 5.)
    #print "level5 EB ", dust[EB][78], errors[EB][78]
    #print "level5 TB ", dust[TB][78], errors[TB][78]
    dust = mult_cls(dust, 2.)
    errors = mult_cls(errors, 2.)
    #print "level10 EB ", dust[EB][78], errors[EB][78]
    #print "level10 TB ", dust[TB][78], errors[TB][78]

    #dust, errors = frac_dust(dust, errorsfit, 0.01)
    #print "frac.01 EB ", dust[EB][78], errors[EB][78]
    #print "frac.01 TB ", dust[TB][78], errors[TB][78]
    #dust, errors = frac_dust(dust, errorsfit, 0.1)
    #print "frac.1 EB ", dust[EB][78], errors[EB][78]
    #print "frac.1 TB ", dust[TB][78], errors[TB][78]
    #dust, errors = frac_dust(dust, errorsfit, 0.5)
    #print "frac.5 EB ", dust[EB][78], errors[EB][78]
    #print "frac.5 TB ", dust[TB][78], errors[TB][78]
    return 

def testcls1():
    xname = '$\ell$'
    yname = '$\ell (\ell+1) \hat{C}^{EB}_{\ell}/ 2\pi [\mu K^2]$'
    uk = 1.e12
    f, ax = pl.subplots()

    ell, cmb = load_cls()
    l, dustraw, errorsraw = load_colin_dust(False)
    dustfit, errorsfit = load_extrap_dust(ell, False)

    psi_pm = np.arange(-1, 1.5, 0.5)*d2r
    color = iter(cm.rainbow(np.linspace(0,1,len(psi_pm))))
    for i, psi in enumerate(psi_pm):
        c = next(color)
        obs, dust, errors = rotate_dust(ell, cmb, psi, True, 0,0,False)
        ax.plot(ell, obs[EB]*uk, c=c, label="$\Delta\psi=%s^{\circ}$" %str(round(psi*r2d,2)))
        #ax.semilogx(ell, obs[EB]*uk, c=c, label="$\Delta\psi=%s^{\circ}$" %str(round(psi*r2d,2)))
        #obs2, dust, errors = rotate_dust(ell, cmb, psi, False,0.5,5,True)
        #ax.semilogx(ell, obs2[EB]*uk, linestyle='--', c=c, label="Dusty, $\Delta\psi=%s^{\circ}$" %str(round(psi*r2d,2)))
        #ax.plot(ell, obs[TB]*uk, linestyle='--', c=c, label="$\Delta\psi=%s^{\circ}$" %str(round(psi*r2d,2)))

    #frac = [.01, .1, .5]
    frac = [.5]
    n = len(frac)
    dust = mult_cls(dustfit, 5.*uk)
    for i in range(n):
        dust, errors = frac_dust(dust, errorsfit, frac[i])
        l = '$m=5$, $f_c$='+str(frac[i])
        ax.plot(ell, dust[EB], color='b', linestyle='--', label=l)
        #ax.semilogx(ell, dust[EB], color='b', linestyle='--')
        #ax.semilogx(ell, dust[EB], color='g', label=l)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_ylim(-2.5, 2)
    ax.legend(fontsize=11, loc=3)
    #ax.set_xticks(np.logspace(0,4,10))
    #ax.grid(both=True)
    #ax.set_title("Measured and Fit EB")
    sname = 'dustfracrotatedEB.eps'
    pl.savefig(sname, format='eps')
    return 

def testcls():
    names =['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
    [TT, EE, BB, TE, TB, EB] = names
    xname = '$\ell$'
#    yname = '$\ell (\ell+1) C_{\ell}/ 2\pi [\mu K^2]$'
    yname = '$[\ell (\ell+1) C_{\ell}/ 2\pi]^{1/2} [\mu K]$'

    ell, cmb = load_cls()
    #plot_cls(ell, cmb, "primordial")
    obs, dust, errors = rotate_dust(ell, cmb, 2*d2r, True, 0,0,False)
    obs2, dust, errors = rotate_dust(ell, cmb, 2*d2r, False,0,0,False)
#    dust = load_colin_dust(ell, False)
    #plot_dust_cls(ell, dust, "dust")

    beam = 5./(60.)*pi/180.
    window = (hp.sphtfunc.gauss_beam(beam, lmax=ell[-1])**2)[2:]
    nl = 5      #microK arcmin
    noise = ((ell*(ell+1)/(2*pi))*((pi/10800.)*nl)**2)
    #nl2 = 10      #microK arcmin
    #noise2 = ((ell*(ell+1)/(2*pi))*((pi/10800.)*nl2)**2)*1e-12

    pl.loglog(ell, np.sqrt(cmb['EE']*1.e12), label="Primordial EE")
    pl.loglog(ell, np.sqrt(cmb['BB']*1.e12), label="Lensing BB, r=0")
    #pl.loglog(ell, np.sqrt(noise), label="noise at %s $\mu$K arcmin" % str(nl))
    pl.loglog(ell, np.sqrt((obs['BB']*1.e12+noise)), label="Rotated BB, $\Delta\psi=2.0^{\circ}$")
    #pl.loglog(ell, np.sqrt((obs2['BB']*1.e12+noise)*window), label="Rotated BB with dust")
    #pl.loglog(ell, (noise2), label="noise at %s $\mu$K arcmin" % str(nl2))
    #pl.loglog(ell, np.sqrt(dust['EE']*window*1.e12), label="Dust EE")
    pl.loglog(ell, np.sqrt(dust['BB']*1.e12), label="Dust BB")
    pl.xlim([2, ell[-1]])
    pl.ylim([1e-4, 1e1])
    pl.xlabel(xname)
    pl.ylabel(yname)
    #pl.grid()
    #pl.title("Power Spectra Amplitudes")
    pl.legend(loc=4, fontsize=16)
    sname = "fig2_sqrt_rotate_dustplot.pdf"
    #pl.savefig(sname, format='eps')
    pl.savefig(sname, format='pdf')
    return 


def run():
    name = "inttest"
    num_curves = 3
    lmax = 2000
    nl = 5.
    plots = False

    psi = 0.0*d2r
    fsky = 0.01
    beam = 5*arcmin2r
    nodust = 1
    frac = 0
    frac_on = 0
    level = 0
    use_full = 0

    if True:
        name = "repeatable"
        num_curves = 3
        psi = 0.0*d2r
        fsky = 0.1
        beam = 17.*arcmin2r
        nodust = 0
        frac = [0.01, 0.1, 0.5]
        frac_on = 1
        level = 10
        use_full = 0
        #nl = [2.1, 3.3]
        nl = 5.
        lmax = 498

    if False:
        name = "intfaster"
        num_curves = 6
        psi = 0.0*d2r
        #fsky = 0.7
        fsky = [0.01,0.01,0.01,0.7,0.7,0.7]
        #beam = 60*arcmin2r
        beam = np.array([5,5,5,60,60,60])*arcmin2r
        beam = beam.tolist()
        #nodust = [1,0,0]
        nodust = 0
        frac = 0
        frac_on = 0
        #level = 5
        level = [1,5,10,1,5,10]
        #use_full = [0,0,1]
        use_full = 1

    if False:
        name = "fraceb"
        #name = "fractb"
        num_curves = 6
        fsky = [0.01,0.01,0.01,0.7,0.7,0.7]
        beam = np.array([5,5,5,60,60,60])*arcmin2r
        beam = beam.tolist()
        nodust = 0
        frac = [0.01,0.1,0.5,0.01,0.1,0.5]
        frac_on = 1
        level = 5
        use_full = 0

    if False:
        #name = "beamsn2"
        name = "beamsn2dust"
        num_curves = 4
        psi = -2.*d2r
        fsky = [0.01,0.1,0.1,0.7]
        beam = np.array([5,5,60,60])*arcmin2r
        beam = beam.tolist()
        #nodust = 1
        nodust = 0
        frac = 0
        frac_on = 0
        level = 0
        use_full = 0

    if False:
        name = "p0levelsm"
        num_curves = 6
        psi = 0*d2r
        fsky = [0.01,0.01,0.01,0.7,0.7,0.7]
        beam = np.array([5,5,5,60,60,60])*arcmin2r
        beam = beam.tolist()
        nodust = 0
        frac = 0
        frac_on = 0
        level = [1,5,10,1,5,10]
        use_full = 0

    sky_params = {"psi": psi, "fsky": fsky, "lmax": lmax, "beam": beam, "nl": nl}
    dust_params = {"nodust": nodust, "frac": frac, "level": level, \
                    "frac_on": frac_on, "use_full":use_full}
    make_data(plots, sky_params, dust_params, num_curves,  name)
    return 

if __name__ == "__main__":
    #foo2()
    #foo1()
    #foo()
    #fit_r()
    #testcls2()
    #testcls1()
    #testcls()
    run()
    
