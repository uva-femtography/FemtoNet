import importlib
import os
import pandas as pd
import numpy as np

from tqdm import tqdm
from femtonet.generator.log import pylogger


class FemtoGen:
    """
    Generator class for unpolarized deeply virtual compton scattering.

    DVCS unpolarized scattering models based on work by Kriesten, Liuti, et al:
        - Theory of Deeply Virtual Compton Scattering off Unpolarized Proton (2020), 2004.08890 [hep-ph]

        - Extraction of Generalized Parton Distribution Observables from Deeply Virtual Electron Proton
          Scattering Experiments, Phys.Rev.D 101 (2020) 5, 054021 1903.05742 [hep-ph]

    """

    def __init__(self):
        self._module_list = {}
        self._generator = None

        self.register_generator(generator='UU', module='unpolarized', package='femtonet.generator.physics')
        print('Available physics modules:')
        for key, value in tqdm(self._module_list.items()):
            print('\t** Key = {name}  Module = {pkg}.{module}'.format(name=key, pkg=value[0], module=value[1]))

    @property
    def generator(self):
        """
        Returns instance of generator class. Intended to give user access to generator functionality
        :return: Instance of generator class
        """
        return self._generator

    @generator.setter
    def generator(self, generator=None):
        """
        Setter method to set custom generator class directly without loading external module first.
        :param generator:
        :return: None
        """
        self._generator = generator

    def initialize(self, name="Generator"):
        """
        Initialize class within module.
        :param name: Class name
        :return: None
        """
        self._generator = getattr(self._module, name)()

    @pylogger.logger
    def load_generator(self, generator='UU'):
        """
        Load scattering module from physics library.
        :param generator: Generator tag. This is user defined and must be set when registering a scattering module.
        :return: None
        """
        try:
            self._module = importlib.import_module(
#                "femtonet.generator.physics", "unpolarized"
                self._module_list[generator][0],
                package=self._module_list[generator][1]
            )
            self.initialize()
        except ImportError as err:
#            print('Error: {func}: {error}'.format(func=__name__, error=err))
            print(err)


    def register_generator(self, generator=None, module=None, package='physics'):
        """
        Register scattering generator in FemtoGen module list.
        :param generator: Generator tag.
        :param module: Generator module
        :param package: relative package name.
        :return: None
        """
        self._module_list[generator] = (package, module)
        print('New generator module added to list: ... {0}.{1}'.format(package, module))


    @staticmethod
    def write_cross_section_csv(data_frame: 'DataFrame') -> 'DataFrame':
        """
        Writes the generated total dvcs cross-section and angle data to a csv formatted file in the
        ouput directory. If directory does not exist, try and create the directory relative to current
        working directory. Once the pandas data frame is created, it is returned from the function.

        :param data_frame a PANDAS dataFrame to write to disk.
        :return: PANDAS dataFrame containing cross section and phi angle information
        """

        try:
            assert os.path.isdir(os.path.join(os.getcwd(), 'output')) is True

        except AssertionError:
            print('Output directory does not exist, trying to create in base directory: {}'.format(os.getcwd()))
            directory = os.path.join(os.getcwd(), 'output')
            os.mkdir(directory, mode=0o777)

        data_frame.to_csv('output/cross_section.csv')

        return data_frame

    @staticmethod
    def make_cross_section_date_frame(cs: '{key:(numpy.array,numpy.array)}') -> 'DataFrame':
        """
        Builds a pandas DataFrame containing phi and cross-section data. Cross-section data is read in as a
        dictionary containing a key with the column description and a numpy array containing the cross-section
        data.

        || Warning || It is on the user to make sure the cross-section data correlates with the phi angle. No
        checks are made to test this.

        :param phi: scattering angle relative to the x-y plane (radians).
        :param cs: a dictionary with keys describing the column names and numpy arrays
                   containing the generated cross-section versys phi data
        :return: PANDAS dataFrame containing cross section and phi angle information
        """

        df = pd.DataFrame({})
        for name, cross_section in cs.items():
            cs_type = np.chararray(shape=(cross_section[0].size,), itemsize=len(name))
            cs_type[:] = name

            if len(df) == 0:
                df = pd.DataFrame({'phi': cross_section[0],
                                   'cross-section': cross_section[1],
                                   'scattering-type': cs_type.decode()
                                   })
            else:
                df = pd.concat([df,
                                pd.DataFrame({'phi': cross_section[0],
                                              'cross-section': cross_section[1],
                                              'scattering-type': cs_type.decode()
                                              })
                                ])

        return df
