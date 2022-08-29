from dataclasses import dataclass
from abc import ABC
from abc import abstractmethod


class FemtoGenAbstractClass(ABC):
    """
        Generator class for unpolarized deeply virtual compton scattering.

        DVCS unpolarized scattering models based on work by Kriesten, Liuti, et al:
            - Theory of Deeply Virtual Compton Scattering off Unpolarized Proton (2020), 2004.08890 [hep-ph]

            - Extraction of Generalized Parton Distribution Observables from Deeply Virtual Electron Proton
              Scattering Experiments, Phys.Rev.D 101 (2020) 5, 054021 1903.05742 [hep-ph]

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update_elastic_form_factors(self, t: float) -> 'float, float, float, float':
        """

        Calculated the elastic and magnetic form factors F1, F2, Ge, Gm

        :param t: four momentum transfer of proton
        :return: The Fermi, Dirc, Electric, Magnetic form factors
        """

    @abstractmethod
    def calculate_kinematics(self, phi: float) -> 'None':
        """
        Calculates the kinematic variables needed tp compute the unpolarized scattering cross-section for DVCS

        :param phi: scattering angle relative to the x-y plane (radians).
        :return: None
        """
        pass

    @abstractmethod
    def _calculate_interference(self, phi: float) -> 'float, float, float':
        """
        Calculation of coefficients for interference term in the DVCS scattering cross section.

        :param phi: scattering angle relative to the x-y plane (radians).
        :return: coefficients for the DVCS-BN interference term.
        """
        pass

    @abstractmethod
    def _calculate_bethe_heitler(self, phi: float) -> 'float, float, float':
        """

        Calculation of coefficients for bethe-heitler term in the DVCS scattering cross section.

        :param phi: scattering angle relative to the x-y plane (radians).
        :return: coefficients for the DVCS-bethe-heitler interference term.
        """
        pass

    @abstractmethod
    def _calculate_dvcs(self, phi: float) -> 'float, float, float':
        """
        Calculation of coefficients for DVCS term in the DVCS scattering cross section.

        :param phi: scattering angle relative to the x-y plane (radians).
        :return: coefficients for the DVCS interference term.
        """
        pass

    @abstractmethod
    def generate_cross_section(self, phi: 'np.array', error=None, type='full') -> 'iterator':
        """

        Generator for the DVCS cross section as a function of phi

        :param phi: scattering angle relative to the x-y plane (radians).
        :param type: string key describing type of cross section to be calculated: full, dvcs, bh, int.
        :param error: Iterable error distribution. Must be a python generator function. Should be used
                              with femtogen.error_generator() function.
        :return: yield cross section value for a given phi value.
        """
        pass

    @abstractmethod
    def read_data_file(self, file: str) -> 'list':
        """

        Read comma separated file containing kinematic information including estimates of compton form factors.

        :param file: comma seperated text file containing kinematic variable information
        :return: list of pystruct dataclasses containing different kinematic information.
        """
        pass

    @abstractmethod
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
        pass


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
