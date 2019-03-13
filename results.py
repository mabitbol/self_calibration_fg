from self_calibration import SelfCalibrationSO

# mode 1 baseline
# mode 2 goal

fnames = ['./data/SAT_default_noise_optimistic_mode1.npy', 
          './data/SAT_default_noise_optimistic_mode2.npy', 
          './data/LAT_default_P_noise_mode1.npy',
          './data/LAT_default_P_noise_mode2.npy']

for fname in fnames:
    selfcalibration = SelfCalibrationSO(fname)
    print(fname)
    print("Input \Delta\Psi = 0 degrees")
    for nu in selfcalibration.so_freqs:
        selfcalibration.run_self_calibration(0, nu, True)
    print()

    print("Input \Delta\Psi = -2 degrees")
    for nu in selfcalibration.so_freqs:
        selfcalibration.run_self_calibration(-2, nu, True)
    print()

    print("Input \Delta\Psi = +2 degrees")
    for nu in selfcalibration.so_freqs:
        selfcalibration.run_self_calibration(2, nu, True)
    print()
