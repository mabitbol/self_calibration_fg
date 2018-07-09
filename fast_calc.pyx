import numpy as np
cimport numpy as np


def tb_resid_n(psi, A, C, dtb):
    x = C + A*np.sin(2*psi)
    return np.sum((x*x)/(2*dtb))

def eb_resid_n(psi, A, C, deb):
    x = C + 0.5*A*np.sin(4*psi)
    return np.sum((x*x)/(2*deb))

def fast_resid_tb(psi_a, A, C, db):
    cdef int i, N
    N = len(psi_a)
    Ltb = np.zeros(N)
    for i in range(N):
        Ltb[i] = -tb_resid_n(psi_a[i], A, C, db)
    return Ltb

def fast_resid_eb(psi_a, A, C, db):
    cdef int i, N
    N = len(psi_a)
    Leb = np.zeros(N)
    for i in range(N):
        Leb[i] = -eb_resid_n(psi_a[i], A, C, db)
    return Leb


def fast_tb_exp(fte, ftb, psi, cmb, obs, dust, errors, dtb):
    names = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
    [TT, EE, BB, TE, TB, EB] = names
    A = cmb[TE]+fte*dust[TE]
    C = obs[TB]
    D = ftb*dust[TB]
    x = C + A*np.sin(2*psi) - D*np.cos(2*psi)
    g1 = np.exp(-np.sum((x*x)/(2*dtb)))
    g2 = np.exp(-np.sum(((fte*dust[TE]-dust[TE])/(2*errors[TE]))**2))
    g4 = np.exp(-np.sum(((ftb*dust[TB]-dust[TB])/(2*errors[TB]))**2))
    return g1*g2*g4


def fast_eb_exp(fee, fbb, feb, psi, cmb, obs, dust, errors, deb):
    names = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
    [TT, EE, BB, TE, TB, EB] = names
    X = (cmb[EE]-cmb[BB])+(fee*dust[EE]-fbb*dust[BB])
    Y = obs[EB]
    Z = feb*dust[EB]
    x = Y + 0.5*X*np.sin(4*psi) - Z*np.cos(4*psi)
    g1 = np.exp(-np.sum((x*x)/(2*deb)))
    g2 = np.exp(-np.sum(((fee*dust[EE]-dust[EE])/(2*errors[EE]))**2))
    g3 = np.exp(-np.sum(((fbb*dust[BB]-dust[BB])/(2*errors[BB]))**2))
    g4 = np.exp(-np.sum(((feb*dust[EB]-dust[EB])/(2*errors[EB]))**2))
    return g1*g2*g3*g4


def cLint(N,steps,lims,cmb,obs,dust,errors,deb,dtb,psi_a):
    Leb = np.zeros(N)
    Ltb = np.zeros(N)
    eelim, bblim, eblim, telim, tblim = lims
    for i in range(N):
        p = psi_a[i]
        x = np.zeros(steps)
        x1 = np.zeros(steps)
        for j in range(steps):
            eel = eelim[j]
            tel = telim[j] 
            z = np.zeros(steps)
            z1 = np.zeros(steps)
            for k in range(steps):
                bbl = bblim[k]
                tbl = tblim[l]
                z1[l] = fast_tb_exp(tel,bbl,tbl,p,cmb,obs,dust,errors,dtb)
                y = np.zeros(steps)
                for l in range(steps):
                    ebl = eblim[l]
                    y[l] = fast_eb_exp(eel,bbl,ebl,p,cmb,obs,dust,errors,deb)
                z[k] = np.trapz(y,eblim)
            x[j] = np.trapz(z,bblim)
            x1[j] = np.trapz(z1,bblim)
        Leb[i] = np.trapz(x,eelim)
        Ltb[i] = np.trapz(x1,telim)
    return Leb, Ltb


