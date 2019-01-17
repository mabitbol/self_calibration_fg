import numpy as np
from numpy import sin, cos
from scipy.optimize import curve_fit

import load_data as ld
d2r = np.pi / 180.

class SelfCalibrationSO:
    
    def __init__(self, low_ell_cut=None):
        self.so_freqs = [ 27.,  39.,  93., 145., 225., 280.]

        self.prepare_cmb_so(low_ell_cut)
        self.prepare_foregrounds()
        return 

    def run_self_calibration(self, psi0_deg, nu):
        psi0 = psi0_deg * d2r
        observed_cls, eb_var = self.preform_observation(psi0, nu)
        
        self.xpsis = np.linspace(psi0_deg-3., psi0_deg+3., 10000) * d2r
        self.calculate_eb_lnlike(observed_cls, eb_var)
        self.get_bias_sigma()

        bias = float("%0.3f" %(self.bias/d2r - psi0_deg))
        sigma = float("%0.3f" %(self.sigma/d2r))
        print("%d GHz: bias=%0.3f, sigma=%0.3f" %(nu, bias, sigma))
        return

    def calculate_eb_lnlike(self, obs, eb_var):
        # make this a numpy calculation so it is fast. 
        eb_lnlike = []
        for psi in self.xpsis:
            numerator = obs['EB'] + 0.5 * sin(4.*psi) * (self.cmb['EE'] - self.cmb['BB'])
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
        return 
        
    def prepare_cmb_so(self, low_ell_cut):
        cmb_cls = ld.load_cmb()
        so_noise = ld.load_SO_noise()

        if low_ell_cut:
            so_noise = ld.truncate(so_noise, lmin=low_ell_cut, lmax=10000)

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
    
    def preform_observation(self, psi0, nu):
        fgs = ld.make_fgs(self.dust_353, self.dust_scaling[nu], self.synch_spass, self.synch_scaling[nu])
        observed_cls = self.rotate_data(psi0, fgs)
        eb_var = self.calculate_eb_var(observed_cls, self.noise[nu])
        return observed_cls, eb_var
            
    def rotate_data(self, psi, fg):
        cmb = self.cmb
        obs={} 
        obs['EE'] = (sin(2*psi)**2)*(cmb['BB']+fg['BB']) + (cos(2*psi)**2)*(cmb['EE']+fg['EE']) + sin(4*psi)*fg['EB']
        obs['BB'] = (cos(2*psi)**2)*(cmb['BB']+fg['BB']) + (sin(2*psi)**2)*(cmb['EE']+fg['EE']) - sin(4*psi)*fg['EB']
        obs['EB']  = 0.5*sin(4*psi)*(cmb['BB']-cmb['EE']+fg['BB']-fg['EE']) + cos(4*psi)*fg['EB']
        return obs

    def calculate_eb_var(self, obs, noise, fsky=0.1):
        C_EE_tot = obs['EE'] + noise
        C_BB_tot = obs['BB'] + noise
        delta_EB_var = 1. / (2. * self.ells + 1.) * C_EE_tot * C_BB_tot / fsky
        return delta_EB_var
            


#selfcalibration = SelfCalibrationSO()
#print("Input \Delta\Psi = 0 degrees")
#for nu in selfcalibration.so_freqs:
#    selfcalibration.run_self_calibration(0, nu)
#print("Input \Delta\Psi = -2 degrees")
#for nu in selfcalibration.so_freqs:
#    selfcalibration.run_self_calibration(-2, nu)
#print("Input \Delta\Psi = +2 degrees")
#for nu in selfcalibration.so_freqs:
#    selfcalibration.run_self_calibration(2, nu)




"""
def make_fit(cmb):
    def eb_fit(ells, psi):
        return 0.5 * sin(4.*psi) * (cmb['EE'] - cmb['BB'])
    return eb_fit

def fit_self_calibration(psi0_deg, nu=145.):
    cmb_cls, so_noise, dust, eb_var, obs = prepare_data(psi0_deg=psi0_deg, nu=nu)
    #fitting_function = make_fit(cmb_cls)
    brange = 2. * d2r
    bounds = [psi0_deg*d2r - brange, psi0_deg*d2r + brange]
    popt, cov = curve_fit(make_fit(cmb_cls), cmb_cls['ells'], obs['EB'], sigma=np.sqrt(eb_var), absolute_sigma=True, bounds=bounds)
    bias = popt[0]
    sigma = np.sqrt(cov[0][0])
    return bias, sigma
"""




