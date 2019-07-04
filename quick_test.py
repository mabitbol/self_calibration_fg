from self_calibration import SelfCalibrationSO


psis_deg = [-2., -1., 0., 1., 2.]

print("SAT")
fnames = ['./data/SAT_default_noise_optimistic_baseline.npy']
'''
          './data/SAT_default_noise_optimistic_goal.npy', 
          './data/SAT_default_noise_pessimistic_baseline.npy', 
          './data/SAT_default_noise_pessimistic_goal.npy', 
          './data/SAT_pertube_peryear_noise_optimistic_baseline.npy',
          './data/SAT_pertube_peryear_noise_optimistic_goal.npy',
          './data/SAT_pertube_peryear_noise_pessimistic_baseline.npy',
          './data/SAT_pertube_peryear_noise_pessimistic_goal.npy']
'''
for fname in fnames:
    selfcalibration = SelfCalibrationSO(fname, ell_max_cut=1000)
    print(fname.split('/')[-1])
    for psi in psis_deg:
        bs = []
        sigs = []
        #print("Input \Delta\Psi = %d degrees" %psi)
        for nu in selfcalibration.so_freqs:
            bias, sigma = selfcalibration.run_self_calibration(psi, nu, False, fsky=0.1);
            bs.append(bias)
            sigs.append(sigma)
        print('$\\Psi_{{\\rm{{in}}}}={0}^{{\circ}}$ & ${1}\pm{2}$ & ${3}\pm{4}$ & ${5}\pm{6}$ & ${7}\pm{8}$ & ${9}\pm{10}$ & ${11}\pm{12}$ \\\\'.format(int(psi), bs[0], sigs[0], \
               bs[1], sigs[1], bs[2], sigs[2], bs[3], sigs[3], bs[4], sigs[4], bs[5], sigs[5]))

print("300")
for fname in fnames:
    selfcalibration = SelfCalibrationSO(fname, ell_max_cut=300)
    print(fname.split('/')[-1])
    for psi in psis_deg:
        bs = []
        sigs = []
        #print("Input \Delta\Psi = %d degrees" %psi)
        for nu in selfcalibration.so_freqs:
            bias, sigma = selfcalibration.run_self_calibration(psi, nu, False, fsky=0.1);
            bs.append(bias)
            sigs.append(sigma)
        print('$\\Psi_{{\\rm{{in}}}}={0}^{{\circ}}$ & ${1}\pm{2}$ & ${3}\pm{4}$ & ${5}\pm{6}$ & ${7}\pm{8}$ & ${9}\pm{10}$ & ${11}\pm{12}$ \\\\'.format(int(psi), bs[0], sigs[0], \
               bs[1], sigs[1], bs[2], sigs[2], bs[3], sigs[3], bs[4], sigs[4], bs[5], sigs[5]))
