from self_calibration import SelfCalibrationSO


psis_deg = [-2., -1., 0., 1., 2.]

print("SAT")
fnames = ['./data/SAT_default_noise_optimistic_baseline.npy', 
          './data/SAT_default_noise_optimistic_goal.npy', 
          './data/SAT_default_noise_pessimistic_baseline.npy', 
          './data/SAT_default_noise_pessimistic_goal.npy', 
          './data/SAT_pertube_peryear_noise_optimistic_baseline.npy',
          './data/SAT_pertube_peryear_noise_optimistic_goal.npy',
          './data/SAT_pertube_peryear_noise_pessimistic_baseline.npy',
          './data/SAT_pertube_peryear_noise_pessimistic_goal.npy']
for fname in fnames:
    selfcalibration = SelfCalibrationSO(fname)
    print(fname.split('/')[-1])
    for psi in psis_deg:
        print("Input \Delta\Psi = %d degrees" %psi)
        for nu in selfcalibration.so_freqs:
            selfcalibration.run_self_calibration(psi, nu, True, fsky=0.1)
        print()
print()


print("LAT")
fnames = ['./data/LAT_default_P_noise_baseline.npy',
          './data/LAT_default_P_noise_goal.npy', 
          './data/LAT_pertube_peryear_P_noise_baseline.npy', 
          './data/LAT_pertube_peryear_P_noise_goal.npy']
for fname in fnames:
    selfcalibration = SelfCalibrationSO(fname)
    print(fname.split('/')[-1])
    for psi in psis_deg:
        print("Input \Delta\Psi = %d degrees" %psi)
        for nu in selfcalibration.so_freqs:
            selfcalibration.run_self_calibration(psi, nu, True, fsky=0.4)
        print()
