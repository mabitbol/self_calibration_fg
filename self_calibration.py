import numpy as np
from numpy import sin, cos

import load_data as ld
d2r = np.pi / 180.

class SelfCalibrationSO:

    def __init__(self, fname, ell_min_cut=None, ell_max_cut=None):
        self.so_freqs = [ 27.,  39.,  93., 145., 225., 280.]

        self.prepare_cmb_so(fname, ell_min_cut, ell_max_cut)
        self.prepare_foregrounds()
        return

    def run_self_calibration(self, psi0_deg, nu, doprint=False, fsky=0.1):
        psi0 = psi0_deg * d2r
        observed_cls, eb_var = self.preform_observation(psi0, nu, fsky)

        self.xpsis = np.linspace(psi0_deg-5., psi0_deg+5., 100000) * d2r
        self.calculate_eb_lnlike(observed_cls, eb_var)
        self.get_bias_sigma()

        bias = float("%0.2f" %(self.bias/d2r - psi0_deg))
        sigma = float("%0.2f" %(self.sigma/d2r))
        if doprint:
            print("%d GHz: bias=%0.3f, sigma=%0.3f" %(nu, bias, sigma))
        return bias, sigma

    def calculate_eb_lnlike(self, obs, eb_var):
        eb_lnlike = []
        for psi in self.xpsis:
            numerator = obs['EB'] + 0.5 * sin(4.*psi) * (self.cmb['EE'] - \
                        self.cmb['BB'])
            lnlike = -np.sum(0.5*numerator**2 / eb_var)
            eb_lnlike.append(lnlike)
        self.eb_lnlike = eb_lnlike
        return

    def get_bias_sigma(self):
        likep = self.eb_lnlike - np.max(self.eb_lnlike)
        self.bias = self.xpsis[np.where(likep == np.max(likep))[0]][0]
        y = np.cumsum(np.exp(likep))
        y /= np.max(y)
        try:
            self.sigma = self.xpsis[y>=0.6827][0] - self.xpsis[y<=0.3173][-1]
        except:
            self.sigma=np.inf
        if False:
            if np.abs(self.bias) >= np.diff(self.xpsis)[0]:
                print("Psi resolution not fine enough to resolve bias.")
            if self.sigma >= np.diff(self.xpsis)[0]:
                print("Psi resolution not fine enough to calculate sigma! \
                        Potentially bad sigma.")
            if (self.bias + self.sigma) > np.max(self.xpsis):
                print("Psi array not wide enough for bias and sigma! \
                        Potentially bad! (1)")
            if (self.bias - self.sigma) > np.min(self.xpsis):
                print("Psi array not wide enough for bias and sigma! \
                        Potentially bad! (2)")
        return

    def prepare_cmb_so(self, fname, ell_min_cut, ell_max_cut):
        cmb_cls = ld.load_cmb()
        so_noise = ld.load_SO_noise(fname)

        so_noise = ld.truncate(so_noise, lmin=ell_min_cut, lmax=ell_max_cut)
        ellmin = so_noise['ells'].min()
        ellmax = so_noise['ells'].max()
        cmb_cls = ld.truncate(cmb_cls, lmin=ellmin, lmax=ellmax)

        assert np.all(so_noise['ells'] == cmb_cls['ells'])

        self.ells = cmb_cls['ells']
        self.cmb = cmb_cls
        self.noise = so_noise
        return

    def prepare_foregrounds(self):
        self.dust_353 = ld.load_dust(self.ells)
        self.synch_spass = ld.load_synch(self.ells)

        self.dust_scaling = {}
        self.synch_scaling = {}
        for nu in self.so_freqs:
            freq = nu * 1.e9
            self.dust_scaling[nu] = ld.scale_dust(freq)
            self.synch_scaling[nu] = ld.scale_synch(freq)
        return

    def preform_observation(self, psi0, nu, fsky):
        fgs = ld.make_fgs(self.dust_353, self.dust_scaling[nu], \
                            self.synch_spass, self.synch_scaling[nu])
        observed_cls = self.rotate_data(psi0, fgs)
        eb_var = self.calculate_eb_var(observed_cls, self.noise[nu], fsky)
        return observed_cls, eb_var

    def rotate_data(self, psi, fg):
        cmb = self.cmb
        obs={}
        obs['EE'] = (sin(2*psi)**2)*(cmb['BB']+fg['BB']) + \
                    (cos(2*psi)**2)*(cmb['EE']+fg['EE']) + sin(4*psi)*fg['EB']
        obs['BB'] = (cos(2*psi)**2)*(cmb['BB']+fg['BB']) + \
                    (sin(2*psi)**2)*(cmb['EE']+fg['EE']) - sin(4*psi)*fg['EB']
        obs['EB'] = 0.5*sin(4*psi)*(cmb['BB']-cmb['EE']+fg['BB']-fg['EE']) + \
                    cos(4*psi)*fg['EB']
        #obs['TB'] = -sin(2*psi)*(cmb['TE']+dust['TE']) + cos(2*psi)*dust['TB']
        return obs

    def calculate_eb_var(self, obs, noise, fsky=0.1):
        C_EE_tot = obs['EE'] + noise
        C_BB_tot = obs['BB'] + noise
        delta_EB_var = 1. / (2. * self.ells + 1.) * C_EE_tot * C_BB_tot / fsky
        return delta_EB_var


