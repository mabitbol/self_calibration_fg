from self_calibration import SelfCalibrationSO

#datadir = '/Users/abitbol/code/self_calibration_fg/data/'
datadir = '/Users/m/Projects/self_calibration_fg/data/'

#fdata = np.load(datadir+'SO_calc_mode2-1_SATyrsLF1_noise_SAT_P.npy', encoding='bytes')
#fdata = np.load(datadir+'SO_calc_mode2-1_SATyrsLF1_fsky0.1_noise_SAT_P.npy')

selfcalibration = SelfCalibrationSO()
print("Input \Delta\Psi = 0 degrees")
for nu in selfcalibration.so_freqs:
    selfcalibration.run_self_calibration(0, nu, True)

#print("Input \Delta\Psi = -2 degrees")
#for nu in selfcalibration.so_freqs:
#    selfcalibration.run_self_calibration(-2, nu, True)
#print("Input \Delta\Psi = +2 degrees")
#for nu in selfcalibration.so_freqs:
#    selfcalibration.run_self_calibration(2, nu, True)
