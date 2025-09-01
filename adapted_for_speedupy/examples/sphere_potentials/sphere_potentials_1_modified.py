import sys
sys.path.append('/home/joaopedrolopez/Downloads/AvaliacaoExperimental/Experimentos/DNACC-with-speedupy/adapted_for_speedupy/examples/sphere_potentials')
from speedupy.speedupy import deterministic
import sys, os
from speedupy.speedupy import initialize_speedupy
from dnacc.derjaguin import calc_spheres_potential
import dnacc
from dnacc.units import nm
import numpy as np

@deterministic
def f1(R, S, betaDeltaG0, hArr, betaFPlate, L):
    filename = 'spheres-R%.1f-S%0.2f-G%.1f.dat' % (R, S, betaDeltaG0)
    betaFSphere = calc_spheres_potential(hArr, betaFPlate, R * L)
    data = [[h / L, 0, 0, V] for (h, V) in zip(hArr, betaFSphere)]
    return {filename: data}

@deterministic
def f2(S, betaDeltaG0, hArr, betaFPlate, L):
    result = {}
    for R in np.linspace(4.0, 50.0, 30):
        result.update(f1(R, S, betaDeltaG0, hArr, betaFPlate, L))
    return result

@deterministic
def f3(S, plates, betaDeltaG0, hArr, L):
    plates.beta_DeltaG0['alpha', 'alphap'] = betaDeltaG0
    betaFPlate = []
    for h in hArr:
        aux = plates.at(h)
        betaFPlate.append(aux.free_energy_density)

    filename = 'plates-S%0.2f-G%.1f.dat' % (S, betaDeltaG0)
    data = []
    for (h, V) in zip(hArr, betaFPlate):
        temp4 = plates.at(h)
        betaFRep = temp4.rep_free_energy_density
        betaFAtt = V - betaFRep
        data.append([h / L, betaFRep / (1 / L ** 2), betaFAtt / (1 / L ** 2), (betaFRep + betaFAtt) / (1 / L ** 2)])
    return betaFPlate, {filename: data}

@deterministic
def f4(S, L, plates, ALPHA, ALPHA_P, hArr):
    sigma = 1 / (S * L) ** 2
    plates.tether_types[ALPHA]['sigma'] = sigma
    plates.tether_types[ALPHA_P]['sigma'] = sigma

    result = {}
    result2 = {}
    for betaDeltaG0 in np.arange(-12, 1, 0.5):
        (betaFPlate, r2) = f3(S, plates, betaDeltaG0, hArr, L)
        result2.update(r2)
        result.update(f2(S, betaDeltaG0, hArr, betaFPlate, L))
    return result, result2

@deterministic
def f5(L, plates, ALPHA, ALPHA_P, hArr):
    result = {}
    result2 = {}
    for S in np.arange(0.1, 1.01, 0.05):
        r, r2 = f4(S, L, plates, ALPHA, ALPHA_P, hArr)
        result.update(r)
        result2.update(r2)
    return result, result2

@initialize_speedupy
def main():
    plates = dnacc.PlatesMeanField()
    L = 20 * nm
    plates.set_tether_type_prototype(sigma=0, L=L)
    ALPHA = plates.add_tether_type(plate='upper', sticky_end='alpha')
    ALPHA_P = plates.add_tether_type(plate='lower', sticky_end='alphap')
    hArr = np.linspace(1 * nm, 40 * nm, 1000)

    result, result2 = f5(L, plates, ALPHA, ALPHA_P, hArr)

    for filename, data in result2.items():
        with open(filename, 'w') as f:
            temp1 = '\t'
            f.write(temp1.join(['h / L', 'F_rep (kT/L^2)', 'F_att (kT/L^2)', 'F_plate (kT/L^2)']) + '\n')
            for (v1, v2, v3, v4) in data:
                f.write('%.7g\t%.7g\t%.7g\t%.7g\n' % (v1, v2, v3, v4))

    for filename, data in result.items():
        with open(filename, 'w') as f:
            temp2 = '\t'
            f.write(temp2.join(['h / L', '[ignore: F_rep (kT)]', '[ignore: F_att (kT)]', 'F_sphere (kT)']) + '\n')
            for (v1, v2, v3, v4) in data:
                f.write('%.7g\t%.7g\t%.7g\t%.7g\n' % (v1, v2, v3, v4))
main()