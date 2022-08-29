import math
import numpy as np
import pandas as pd
import os
import os.path
import femtonet.generator.log.pylogger as log

from scipy import constants
from dataclasses import dataclass
from dataclasses import astuple
from tqdm import tqdm

from femtonet.generator.pymetric import pymetric
from femtonet.generator.physics.baseclass import FemtoGenAbstractClass


class Generator(FemtoGenAbstractClass):
    """
    Generator class for  unpolarized deeply virtual compton scattering.

    DVCS unpolarized scattering models based on work by Kriesten, Liuti, et al:
        - Theory of Deeply Virtual Compton Scattering off Unpolarized Proton (2020), 2004.08890 [hep-ph]

        - Extraction of Generalized Parton Distribution Observables from Deeply Virtual Electron Proton
          Scattering Experiments, Phys.Rev.D 101 (2020) 5, 054021 1903.05742 [hep-ph]

    """

    def __init__(self):

        self.kinematics = 0
        self.GAMMA = 0
        self.tau = 0
        self.data_file = ''
        self.proton_mass = 0.93828

        self.Ge = 0
        self.Gm = 0
        self.F2 = 0
        self.F1 = 0

        self.Q = 0
        self.nu = 0
        self.gamma = 0
        self.tau = 0
        self.y = 0
        self.q_0 = 0
        self.kp_0 = 0
        self.eps = 0
        self.xi = 0
        self.t_0 = 0

        # Trig functions
        self.cosl = 0
        self.sinl = 0
        self.coslp = 0
        self.sinlp = 0
        self.cost = 0
        self.sint = 0

        # Four-vectors
        self.q = 0
        self.qp = 0
        self.p = 0
        self.delta = 0
        self.pp = 0
        self.P = 0
        self.k = 0
        self.kp = 0

        # Contractions
        self.kk = 0
        self.PP = 0
        self.k_qp = 0

        self.kP = 0
        self.k_kp = 0
        self.kp_P = 0
        self.kp_qp = 0

        self.P_qp = 0
        self.kd = 0
        self.kp_d = 0
        self.qpd = 0

        self.kk_t = 0
        self.kqp_t = 0
        self.kkp_t = 0
        self.kpqp_t = 0
        self.kP_t = 0
        self.kpP_t = 0
        self.qpP_t = 0
        self.kd_t = 0
        self.kpd_t = 0
        self.qpd_t = 0

        self.s = 0

        self.GAMMA = 0

        self.D_plus = 0
        self.D_minus = 0

        self.ke = 8.985551e9  # N m^2 C-2 Coulomb's constant
        self.metric = pymetric.Metric()
        self.metric.set_minkowski_metric()

        self.cross_section = {'bh': 0,
                              'dvcs': 0,
                              'int': 0,
                              'full': 0}

    def update_elastic_form_factors(self, t: float) -> 'float, float, float, float':
        """

        Calculated the elastic and magnetic form factors F1, F2, Ge, Gm

        :param t: four momentum transfer of proton
        :return: The Fermi, Dirc, Electric, Magnetic form factors
        """

        try:

            self.Ge = 1 / math.pow(1 + t / 0.710649, 2)
            self.Gm = 2.792847337 * self.Ge
            self.F2 = (self.Gm - self.Ge) / (1 + (t / (4 * math.pow(0.938, 2))))
            self.F1 = self.Gm - self.F2
        except ZeroDivisionError as error:
            print('{error} in: {info}'.format(error=error, info=self.update_elastic_form_factors.__name__))
            raise

        return self.F1, self.F2, self.Ge, self.Gm


    def calculate_kinematics(self, phi: float) -> 'None':
        """

        Calculates the kinematic variables needed tp compute the unpolarized scattering cross-section for DVCS

        :param phi: scattering angle relative to the x-y plane (radians).
        :return: None
        """

        self.Q = math.sqrt(self.kinematics.Q2)
        self.nu = self.kinematics.Q2 / (2 * self.proton_mass * self.kinematics.xbj)
        self.gamma = self.Q / self.nu
        self.tau = -self.kinematics.t / (4 * math.pow(self.proton_mass, 2))
        self.y = self.Q / (self.gamma * self.kinematics.k_0)
        self.q_0 = (self.Q / self.gamma) * (1 + self.kinematics.xbj * self.kinematics.t / self.kinematics.Q2)
        self.kp_0 = self.kinematics.k_0 * (1 - self.y)
        self.eps = (1 - self.y - math.pow(0.5 * self.y * self.gamma, 2)) / (
            1 - self.y + 0.5 * math.pow(self.gamma, 2) + math.pow(0.5 * self.y * self.gamma, 2))
        self.xi = self.kinematics.xbj * (
            (1 + (self.kinematics.t / (2 * self.kinematics.Q2))) / (
            2 - self.kinematics.xbj + ((self.kinematics.xbj * self.kinematics.t) / self.kinematics.Q2)))
        self.t_0 = self.kinematics.Q2 * (1 - math.sqrt(1 + math.pow(self.gamma, 2))
                                         + 0.5 * math.pow(self.gamma, 2)) / (self.kinematics.xbj * (
            1 - math.sqrt(1 + math.pow(self.gamma, 2)) + math.pow(self.gamma, 2) / (2 * self.kinematics.xbj)))

        # Trig functions
        self.cosl = -(1 / math.sqrt(1 + math.pow(self.gamma, 2))) * (1 + 0.5 * self.y * math.pow(self.gamma, 2))
        self.sinl = (self.gamma / math.sqrt(1 + math.pow(self.gamma, 2))) * math.sqrt(
            1 - self.y - math.pow(0.5 * self.gamma * self.y, 2))
        self.coslp = (self.cosl + self.y * math.sqrt(1 + math.pow(self.gamma, 2))) / (1 - self.y)
        self.sinlp = self.sinl / (1 - self.y)
        self.cost = -(1 / math.sqrt(1 + math.pow(self.gamma, 2))) * (1 + (.5 * math.pow(self.gamma, 2)) * (
            (1 + (self.kinematics.t / self.kinematics.Q2)) / (
            1 + ((self.kinematics.xbj * self.kinematics.t) / (self.kinematics.Q2)))))
        self.sint = math.sqrt(1 - math.pow(self.cost, 2))

        # Four-vectors
        self.q = np.array([self.nu, 0, 0, -self.nu * math.sqrt(1 + math.pow(self.gamma, 2))])
        self.qp = self.q_0 * np.array([1, self.sint * math.cos(phi), self.sint * math.sin(phi), self.cost])
        self.p = np.array([self.proton_mass, 0, 0, 0])
        self.delta = self.q - self.qp
        self.pp = self.p + self.delta
        self.P = 0.5 * (self.p + self.pp)
        self.k = self.kinematics.k_0 * np.array([1, self.sinl, 0, self.cosl])
        self.kp = self.kp_0 * np.array([1, self.sinlp, 0, self.coslp])

        # Contractions
        self.kk = self.metric.contract(self.k, self.k)
        self.PP = self.metric.contract(self.p, self.p)
        self.k_qp = self.metric.contract(self.k, self.qp)

        self.kP = self.metric.contract(self.k, self.P)
        self.k_kp = self.metric.contract(self.k, self.kp)
        self.kp_P = self.metric.contract(self.kp, self.P)
        self.kp_qp = self.metric.contract(self.kp, self.qp)

        self.P_qp = self.metric.contract(self.P, self.qp)
        self.kd = self.metric.contract(self.k, self.delta)
        self.kp_d = self.metric.contract(self.kp, self.delta)
        self.qpd = self.metric.contract(self.qp, self.delta)

        self.kk_t = self.metric.contract(self.k, self.k, type='transverse')
        self.kqp_t = self.metric.contract(self.k, self.qp, type='transverse')
        self.kkp_t = self.metric.contract(self.k, self.kp, type='transverse')
        self.kpqp_t = self.metric.contract(self.kp, self.qp, type='transverse')
        self.kP_t = self.metric.contract(self.k, self.P, type='transverse')
        self.kpP_t = self.metric.contract(self.kp, self.P, type='transverse')
        self.qpP_t = self.metric.contract(self.qp, self.P, type='transverse')
        self.kd_t = self.metric.contract(self.k, self.delta, type='transverse')
        self.kpd_t = self.metric.contract(self.kp, self.delta, type='transverse')
        self.qpd_t = self.metric.contract(self.qp, self.delta, type='transverse')

        self.s = self.kk + self.metric.contract(self.p, self.p) + 2 * self.metric.contract(self.k, self.p)

        self.GAMMA = math.pow(constants.alpha, 3) / (
            16 * self.kinematics.xbj * math.pow(constants.pi, 2) * math.pow((self.s - math.pow(self.proton_mass, 2)),
                                                                            2) * math.sqrt(1 + math.pow(self.gamma, 2)))

        self.D_plus = (1 / (2 * self.kp_qp)) - (1 / (2 * self.k_qp))
        self.D_minus = -(1 / (2 * self.kp_qp)) - (1 / (2 * self.k_qp))


    def _calculate_interference(self, phi: float) -> 'float, float, float':
        """

        Calculation of coefficients for interference term in the DVCS scattering cross section.

        :param phi: scattering angle relative to the x-y plane (radians).
        :return: coefficients for the DVCS-BN interference term.
        """

        self.calculate_kinematics(phi)
        A_UU = -4 * math.cos(phi) * (self.D_plus * ((self.kqp_t - 2 * self.kk_t - 2 * self.k_qp) * self.kp_P
                                                    + (
                                                        2 * self.kp_qp - 2 * self.kkp_t - self.kpqp_t) * self.kP + self.kp_qp * self.kP_t
                                                    + self.k_qp * self.kpP_t - 2 * self.k_kp * self.kP_t)
                                     - self.D_minus * ((
                                                           2 * self.k_kp - self.kpqp_t - self.kkp_t) * self.P_qp + 2 * self.k_kp * self.qpP_t
                                                       - self.kp_qp * self.kP_t - self.k_qp * self.kpP_t))

        B_UU = -2 * self.xi * math.cos(phi) * (self.D_plus * (
            (self.kqp_t - 2 * self.kk_t - 2 * self.k_qp) * self.kp_d + (
            2 * self.kp_qp - 2 * self.kkp_t - self.kpqp_t) * self.kd
            + self.kp_qp * self.kd_t + self.k_qp * self.kpd_t - 2 * self.k_kp * self.kd_t)
                                               - self.D_minus * ((
                                                                     2 * self.k_kp - self.kpqp_t - self.kkp_t) * self.qpd + 2 * self.k_kp * self.qpd_t
                                                                 - self.kp_qp * self.kd_t - self.k_qp * self.kpd_t))
        C_UU = -2 * math.cos(phi) * (
            self.D_plus * (2 * self.k_kp * self.kd_t - self.kp_qp * self.kd_t - self.k_qp * self.kpd_t
                           + 4 * self.xi * self.k_kp * self.kP_t - 2 * self.xi * self.kp_qp * self.kP_t - 2 * self.xi * self.k_qp * self.kpP_t)
            - self.D_minus * (
                self.k_kp * self.qpd_t - self.kp_qp * self.kd_t - self.k_qp * self.kpd_t + 2 * self.xi * self.k_kp * self.qpP_t
                - 2 * self.xi * self.kp_qp * self.kpP_t - 2 * self.xi * self.k_qp * self.kpP_t))

        return A_UU, B_UU, C_UU


    def _calculate_bethe_heitler(self, phi: float) -> 'float, float, float':
        """

        Calculation of coefficients for bethe-heitler term in the DVCS scattering cross section.

        :param phi: scattering angle relative to the x-y plane (radians).
        :return: coefficients for the DVCS-bethe-heitler interference term.
        """
        self.calculate_kinematics(phi)
        const = 8 * math.pow(self.proton_mass, 2) / (self.kinematics.t * self.kinematics.t * self.k_qp * self.kp_qp)
        A_UU = -const * (4 * self.tau * (math.pow(self.kP, 2) + math.pow(self.kp_P, 2)) - (self.tau + 1) * (
            math.pow(self.kd, 2) + math.pow(self.kp_d, 2)))
        B_UU = -2 * const * (math.pow(self.kd, 2) + math.pow(self.kp_d, 2))

        return A_UU, B_UU, 0


    def _calculate_dvcs(self, phi: float) -> 'float, float, float':
        """
        Calculation of coefficients for DVCS term in the DVCS scattering cross section.

        :param phi: scattering angle relative to the x-y plane (radians).
        :return: coefficients for the DVCS interference term.
        """
        self.calculate_kinematics(phi)

        A = 4 * (1 - math.pow(self.xi, 2)) * (
            math.pow(self.kinematics.ReH, 2) + math.pow(self.kinematics.ImH, 2) + math.pow(self.kinematics.ReHt,
                                                                                           2) + math.pow(
            self.kinematics.ImHt,
            2))
        B = ((self.t_0 - self.kinematics.t) / 2 * math.pow(self.proton_mass, 2)) * (
            math.pow(self.kinematics.ReE, 2) + math.pow(self.kinematics.ImE, 2) + math.pow(self.xi, 2) * (
            math.pow(self.kinematics.ReEt, 2) + math.pow(self.kinematics.ImEt, 2)))
        C = ((2 * math.pow(self.xi, 2)) / (1 - math.pow(self.xi, 2))) * (
            self.kinematics.ReE * self.kinematics.ReH + self.kinematics.ImE * self.kinematics.ImH + self.kinematics.ReEt * self.kinematics.ReHt + self.kinematics.ImEt *
            self.kinematics.ImHt)

        F_UUT = (A + B - C)

        return F_UUT, 0, 0


    def generate_cross_section(self, phi: 'np.array', error=None, type='full') -> 'iterator':
        """

        Generator for the DVCS cross section as a function of phi

        :param phi: scattering angle relative to the x-y plane (radians).
        :param type: string key describing type of cross section to be calculated: full, dvcs, bh, int.
        :param error: Iterable error distribution. Must be a python generator function. Should be used
                      with femtogen.error_generator() function.
        :return: yield cross section value for a given phi value.
        """

        if type not in self.cross_section:
            raise Exception('Invalid cross-section choice.\n\n{}'.format(self.generate_cross_section.__doc__))

        self.update_elastic_form_factors(-self.kinematics.t)

        for p in tqdm(phi):
            # DVCS Term
            conversion = math.pow(.197326, 2) * 1e7 * (2 * math.pi)
            ADVCS, _, _ = self._calculate_dvcs(p)

            A_DVCS = ADVCS * conversion * (self.GAMMA / (self.kinematics.Q2 * (1 - self.eps)))

            self.cross_section['dvcs'] = (A_DVCS)

            # BH Term
            ABH, BBH, _ = self._calculate_bethe_heitler(p)
            A_BH = -ABH * conversion * self.GAMMA * (math.pow(self.F1, 2) + self.tau * math.pow(self.F2, 2))
            B_BH = -BBH * conversion * self.GAMMA * self.tau * math.pow(self.Gm, 2)

            self.cross_section['bh'] = (A_BH + B_BH)

            # Interference Term
            A, B, C = self._calculate_interference(p)
            A_term = -A * conversion * (self.GAMMA / (-self.kinematics.t * self.kinematics.Q2)) * (
                self.F1 * self.kinematics.ReH + self.tau * self.F2 * self.kinematics.ReE)
            B_term = -B * conversion * (self.GAMMA / (-self.kinematics.t * self.kinematics.Q2)) * (
                self.F1 + self.F2) * (
                         self.kinematics.ReH + self.kinematics.ReE)
            C_term = -C * conversion * (self.GAMMA / (-self.kinematics.t * self.kinematics.Q2)) * (
                self.F1 + self.F2) * self.kinematics.ReHt

            self.cross_section['int'] = (A_term + B_term + C_term)

            self.cross_section['full'] = self.cross_section['bh'] + self.cross_section['dvcs'] + self.cross_section[
                'int']

            yield self.cross_section[type] + next(error)


    def error_generator(self, mean=0.0, stdev=1.0, systematic=0.0) -> 'iterator':
        """

        Iterable python generator to provide random error sampling for a normal distribution.

        :param mean: gaussian mean
        :param stdev: gaussian standard deviation
        :param systematic: systematic error
        :return: yield total error for given inputs.
        """

        while True:
            yield np.random.normal(mean, stdev) + systematic


    def read_data_file(self, file: str) -> 'list':
        """

        Read comma separated file containing kinematic information including estimates of compton form factors.

        :param file: comma seperated text file containing kinematic variable information
        :return: list of pystruct dataclasses containing different kinematic information.
        """

        kinematic_variable_list = []
        try:
            assert os.path.isfile(file) is True

            self.data_file = file

            for chunk in pd.read_csv(file, chunksize=1):
                kinematic_variable_list.append(self.set_kinematics(chunk['xbj'].iloc[0],
                                                                   chunk['t'].iloc[0],
                                                                   chunk['Q2'].iloc[0],
                                                                   chunk['k_0'].iloc[0],
                                                                   chunk['ReH'].iloc[0],
                                                                   chunk['ImH'].iloc[0],
                                                                   chunk['ReE'].iloc[0],
                                                                   chunk['ImE'].iloc[0],
                                                                   chunk['ReHt'].iloc[0],
                                                                   chunk['ImHt'].iloc[0],
                                                                   chunk['ReEt'].iloc[0],
                                                                   chunk['ImEt'].iloc[0]))
        except AssertionError:
            print('Unable to locate file: {}. Please check file path again.'.format(file))
            raise

        return kinematic_variable_list


    def set_kinematics(self, xbj=0., t=0., Q2=0., k_0=0., ReH=0., ReE=0., ImH=0., ImE=0., ReHt=0., ReEt=0., ImHt=0.,
                       ImEt=0., **kwargs) -> 'pystruct':
        """

        Worker function to build a pystruct dataclass for a list of kinematic variables.

        :param xbj: bjorken-x value as a float
        :param t: four momentum transfer of proton
        :param Q2: four momentum transfer of electron
        :param k_0: beam energy
        :param ReH: Real compton FF component
        :param ReE: Real compton FF component
        :param ImH: Imaginary compton FF component
        :param ImE: Imaginary compton FF component
        :param ReHt: Real transverse compton FF component
        :param ReEt: Real transverse compton FF component
        :param ImHt: Imaginary transverse compton FF component
        :param ImEt: Imaginary transverse compton FF component
        :param kawgs: This option allows for the input of an array containing the initialization values.

        :return: pystruct structure containing kinematic variables
        """

        if 'array' in kwargs:
            self.kinematics = PyStruct(*astuple(kwargs['array']))
            return self.kinematics

        self.kinematics = PyStruct(xbj, t, Q2, k_0, ReH, ReE, ImH, ImE, ReHt, ReEt, ImHt, ImEt)

        return self.kinematics


@dataclass
class PyStruct:
    xbj: float = 0.0
    t: float = 0.0
    Q2: float = 0.0
    k_0: float = 0.0
    ReH: float = 0.0
    ReE: float = 0.0
    ImH: float = 0.0
    ImE: float = 0.0
    ReHt: float = 0.0
    ReEt: float = 0.0
    ImHt: float = 0.0
    ImEt: float = 0.0
