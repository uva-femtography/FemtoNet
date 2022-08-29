import numpy as np
import math
import matplotlib.pyplot as plt

from femtonet.generator.femtogen import FemtoGen

if __name__ == "__main__":
    femto = FemtoGen()
    femto.load_generator(generator='UU')
    kinematics = femto.generator.read_data_file('femtonet/generator/data/cff.csv')
    femto.generator.set_kinematics(array=kinematics[0])

    phi = math.radians(360) * np.array([np.random.random() for i in range(1000)])

    dv = np.fromiter(femto.generator.generate_cross_section(phi,
                                                            error=femto.generator.error_generator(stdev=0.0025),
                                                            type='dvcs'), dtype=float, count=phi.size)
    bh = np.fromiter(femto.generator.generate_cross_section(phi,
                                                            error=femto.generator.error_generator(stdev=0.0025),
                                                            type='bh'), dtype=float, count=phi.size)
    it = np.fromiter(femto.generator.generate_cross_section(phi,
                                                            error=femto.generator.error_generator(stdev=0.0025),
                                                            type='int'), dtype=float, count=phi.size)
    cs = np.fromiter(femto.generator.generate_cross_section(phi,
                                                            error=femto.generator.error_generator(stdev=0.0025),
                                                            type='full'), dtype=float, count=phi.size)
    df = femto.make_cross_section_date_frame(
        cs={'dvcs': (phi, dv), 'bh': (phi, bh), 'interference': (phi, it), 'total': (phi, cs)})
    femto.write_cross_section_csv(df)

    plt.scatter(phi, cs, marker='.', color='xkcd:barney purple')
    plt.grid(True)
    plt.xlabel(r'$\phi$(radians)')
    plt.ylabel(r'$\frac{{d^{5}\sigma^{I}_{unpolar}}}{dx_{bj} dQ^{2} dt d\phi d\phi_{s}}$', fontsize='x-large')
    plt.ylim(-0.05, 0.19)
    plt.xlim(0, 6.29)

    plt.show()
