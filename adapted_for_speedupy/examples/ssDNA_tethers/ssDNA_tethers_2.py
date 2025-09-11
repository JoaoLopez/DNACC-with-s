import sys
sys.path.append('/home/joaopedrolopez/Downloads/AvaliacaoExperimental/Experimentos/DNACC-with-speedupy/adapted_for_speedupy/examples/ssDNA_tethers')
from speedupy.speedupy import maybe_deterministic
import sys, os
from speedupy.speedupy import initialize_speedupy
from dnacc.derjaguin import calc_spheres_potential
import numpy as np
from math import pi
import subprocess
import scipy.interpolate
import dnacc
from dnacc.units import nm

class ssDNAStatistics(object):
    l_Kuhn = 5 * nm
    raw = np.loadtxt('interN.dat')
    interp_bridge = scipy.interpolate.interp1d(raw[:, 0] * l_Kuhn, raw[:, 1] * l_Kuhn ** 2, bounds_error=False, fill_value=0.0)
    raw = np.loadtxt('intraRed.dat')
    interp_loop = scipy.interpolate.interp1d(raw[:, 0] * l_Kuhn, raw[:, 1] * l_Kuhn ** 2, bounds_error=False, fill_value=raw[-1, 1] * l_Kuhn ** 2)
    raw = np.loadtxt('4P.dat')
    interp_exclude = scipy.interpolate.interp1d(raw[:, 0] * 1 * nm, np.exp(-raw[:, 1]), bounds_error=False, fill_value=1.0)

    @classmethod
    def calc_boltz_binding_cnf_bridge(cls, system, type_info_i, type_info_j):
        return float(cls.interp_bridge(system.separation)) * float(cls.interp_exclude(system.separation)) ** 2

    @classmethod
    def calc_boltz_binding_cnf_loop(cls, system, type_info_i, type_info_j):
        return float(cls.interp_loop(system.separation)) * float(cls.interp_exclude(system.separation)) ** 2

    @classmethod
    def calc_boltz_exclusion(cls, system, type_info_i):
        return float(cls.interp_exclude(system.separation))

    @classmethod
    def check_system(cls, system):
        if system.separation <= 0:
            raise ValueError('Invalid plate separation')

@initialize_speedupy
def main():
    plates = dnacc.PlatesMeanField(ssDNAStatistics)
    R = 550 * nm
    area = 4 * pi * R ** 2
    ALPHA = plates.add_tether_type(plate='lower', sigma=4800.0 / area, sticky_end='alpha')
    ALPHA_P = plates.add_tether_type(plate='upper', sigma=4200.0 / area, sticky_end='alphap')
    c1 = 24070.0
    c2 = 70.2964
    zr = 273.15
    hArr = np.arange(1 * nm, 300 * nm, 0.05 * nm)
    for T in np.linspace(20.0, 60.0, 41):
        beta_DeltaG0 = -(c1 / (zr + T) - c2)
        plates.beta_DeltaG0['alpha', 'alphap'] = beta_DeltaG0
        betaFPlate = []
        for h in hArr:
            aux = plates.at(h)
            betaFPlate.append(aux.free_energy_density)
        betaFRepPlate = []
        for h in hArr:
            aux = plates.at(h)
            betaFRepPlate.append(aux.rep_free_energy_density)
        with open('plates-A_B-T%.1f-G%.1f.txt' % (T, beta_DeltaG0), 'w') as f:
            f.write('# h (nm)\tF_rep (kT/nm^2)\tF_att (kT/nm^2)\tF_plate (kT/nm^2)\n')
            for (h, betaF, betaFRep) in zip(hArr, betaFPlate, betaFRepPlate):
                betaFAtt = betaF - betaFRep
                f.write('%g\t%g\t%g\t%g\n' % (h / nm, betaFRep / (1 / nm ** 2), betaFAtt / (1 / nm ** 2), betaF / (1 / nm ** 2)))
        badBetaFPlate = []
        for h in hArr:
            aux = plates.rep_free_energy_density
            aux2 = plates.at(h)
            badBetaFPlate.append(aux - aux2.sigma_bound[ALPHA, ALPHA_P])
        with open('bad-plates-A_B-T%.1f-G%.1f.txt' % (T, beta_DeltaG0), 'w') as f:
            f.write('# h (nm)\tF_rep (kT/nm^2)\tF_att (kT/nm^2)\tF_plate (kT/nm^2)\n')
            for (h, betaF, betaFRep) in zip(hArr, badBetaFPlate, betaFRepPlate):
                betaFAtt = betaF - betaFRep
                f.write('%g\t%g\t%g\t%g\n' % (h / nm, betaFRep / (1 / nm ** 2), betaFAtt / (1 / nm ** 2), betaF / (1 / nm ** 2)))
        betaFSpheres = calc_spheres_potential(hArr, betaFPlate, R)
        betaFRepSpheres = calc_spheres_potential(hArr, betaFRepPlate, R)
        with open('spheres-A_B-T%.1f-G%.1f.txt' % (T, beta_DeltaG0), 'w') as f:
            f.write('# h (nm)\tF_rep (kT)\tF_att (kT)\tF_spheres (kT)\n')
            for (h, betaF, betaFRep) in zip(hArr, betaFSpheres, betaFRepSpheres):
                betaFAtt = betaF - betaFRep
                f.write('%g\t%g\t%g\t%g\n' % (h / nm, betaFRep, betaFAtt, betaF))
        badBetaFSpheres = calc_spheres_potential(hArr, badBetaFPlate, R)
        with open('bad-spheres-A_B-T%.1f-G%.1f.txt' % (T, beta_DeltaG0), 'w') as f:
            f.write('# h (nm)\tF_rep (kT)\tF_att (kT)\tF_spheres (kT)\n')
            for (h, betaF, betaFRep) in zip(hArr, badBetaFSpheres, betaFRepSpheres):
                betaFAtt = betaF - betaFRep
                f.write('%g\t%g\t%g\t%g\n' % (h / nm, betaFRep, betaFAtt, betaF))
        Ncn = 401
        blurSigma = 3 * nm
        filtX = np.linspace(-3 * blurSigma, +3 * blurSigma, Ncn)
        filtDx = filtX[1] - filtX[0]
        filtX += 0.5 * filtDx
        filtY = np.exp(-filtX ** 2 / (2 * blurSigma ** 2))
        filtY /= np.sum(filtY)
        resampHArr = np.arange(hArr[0], hArr[-1], filtDx)

        @maybe_deterministic
        def blur(origBetaF):
            interpBetaF = scipy.interpolate.interp1d(hArr, origBetaF)
            resampBetaF = interpBetaF(resampHArr)
            resampExpMinusBetaF = np.exp(-resampBetaF)
            paddedInput = np.concatenate((np.zeros(int(Ncn / 2)), resampExpMinusBetaF, np.ones(int(Ncn / 2))))
            blurredExpMinusBetaF = np.convolve(paddedInput, filtY, mode='valid')
            blurredBetaF = -np.log(blurredExpMinusBetaF)
            return blurredBetaF
        blurredBetaFSpheres = blur(betaFSpheres)
        blurredBetaFRepSpheres = blur(betaFRepSpheres)
        with open('blurred-spheres-A_B-T%.1f-G%.1f.txt' % (T, beta_DeltaG0), 'w') as f:
            f.write('# h (nm)\tblurred F_rep (kT)\tblurred F_att (kT)\tblurred F_spheres (kT)\n')
            for (h, betaF, betaFRep) in zip(resampHArr, blurredBetaFSpheres, blurredBetaFRepSpheres):
                betaFAtt = betaF - betaFRep
                f.write('%g\t%g\t%g\t%g\n' % (h / nm, betaFRep, betaFAtt, betaF))
        blurredBadBetaFSpheres = blur(badBetaFSpheres)
        with open('blurred-bad-spheres-A_B-T%.1f-G%.1f.txt' % (T, beta_DeltaG0), 'w') as f:
            f.write('# h (nm)\tblurred F_rep (kT)\tblurred F_att (kT)\tblurred F_spheres (kT)\n')
            for (h, betaF, betaFRep) in zip(resampHArr, blurredBadBetaFSpheres, blurredBetaFRepSpheres):
                betaFAtt = betaF - betaFRep
                f.write('%g\t%g\t%g\t%g\n' % (h / nm, betaFRep, betaFAtt, betaF))
main()