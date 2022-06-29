import pandas as pd
import numpy as np
import bisect
import os
import multiprocessing as mp

from tqdm import tqdm
from itertools import chain
from scipy import interpolate


class FemtoMesh:

    def __init__(self, name):

        self.data_frame = None
        self.data_frame_name = name
        self.chunksize = 1000000
        self.model_generated = False

        self._xbj = 0
        self._t = 0
        self._q2 = 0

    def open(self):

        df = pd.read_csv(self.data_frame_name).dropna()
        return df

    def build_data_frame(self, xbj: 'float', t: 'float') -> 'pandas.DataFrame':
        """
        Build PANDAS DataFrame from mesh csv. The mesh values are read in chucks keeping only
        the relevant data points to reduce the file size.

        xbj: x-bjorken
        t: proton momentum transfer

        """
        df_list = []
        try:
            for chunk in pd.read_csv(self.data_frame_name, dtype=float, chunksize=self.chunksize):
                df = chunk[(chunk.xbj == xbj) & (chunk.t == t)]
                if df.empty:
                    pass
                else:
                    df_list.append(df)
        except FileNotFoundError as ex:
            print('{0}:{1} >\tFile not found.'.format(ex, __name__))

        self.data_frame = pd.concat(df_list)
        self.data_frame.dropna()
        self.model_generated = True

        return self.data_frame

    def build_data_frame2D(self, t: 'float') -> 'pandas.DataFrame':
        """
        Build PANDAS DataFrame from mesh csv. The mesh values are read in chucks keeping only
        the relevant data points to reduce the file size.

        t: proton momentum transfer
        """

        df_list = []
        try:
            for chunk in pd.read_csv(self.data_frame_name, dtype=float, chunksize=self.chunksize):
                df = chunk[chunk.t == t]
                if df.empty:
                    pass
                else:
                    df_list.append(df)

            self.data_frame = pd.concat(df_list)
            self.data_frame.dropna()
            self.model_generated = True

        except FileNotFoundError as ex:
            print('{0}:{1} >\tFile not found.'.format(ex, __name__))

        return self.data_frame

    @property
    def xbj(self) -> 'float':
        return self._xbj

    @xbj.setter
    def xbj(self, xbj: 'float'):
        self._xbj = xbj

    @property
    def t(self) -> 'float':
        return self._t

    @t.setter
    def t(self, t: 'float'):
        self._t = t

    @property
    def q2(self) -> 'float':
        return self._q2

    @q2.setter
    def q2(self, q2: 'float'):
        self._q2 = q2

    def get_xbj_list(self):
        return self.data_frame.xbj.unique().to_numpy()

    def get_x_list(self):
        return self.data_frame.x.unique().to_numpy()

    def get_t_list(self):
        return self.data_frame.t.unique().to_numpy()

    @staticmethod
    def search(v: 'numpy.array', value: 'float') -> 'float, float':
        """
        Search for specific value in array v and return former and prior value in list.

        value: value to search for in list
        v: list of possible values
        """
        try:
            assert value < len(v)
            assert value > 0
            i = bisect.bisect_left(v, value)
        except AssertionError as assertion:
            print('{0}: The request values is outside of the range of available data.'.format(assertion))
            raise

        return v[i], v[i - 1]

    @staticmethod
    def extrapolate(value: 'float', u_gpd: 'float', l_gpd: 'float', u_value: 'float', l_value: 'float') -> 'float':
        """
        (Deprecated: Replaced by scipy variant)Determine local relationship between Q2 and quark GPD values in a kinematic range; assumes a linear
        relationship. The slope and intercept are then used to determine the GPD value(s) at a Q2 of value.

        value: Q2 value to determine GPD
        u_gpd: upper GPD value
        l_gpd: lower GPD value
        l_value: lower Q2 value
        u_value: upper Q2 value
        """

        m = (u_gpd - l_gpd) / (u_value - l_value)
        b = l_gpd - m * l_value
        return m * value + b

    def model_to_csv(self):
        """
        Convert GPD model DataFrame to csv file
        """
        try:
            assert self.model_generated is True
            self.data_frame.to_csv("gpd_model.csv", index=False, header=['x', 'u', 'd', 'xu', 'xd'])
        except AssertionError as ex:
            print('Model {0} not saved. Returned {1}'.format(self.data_frame_name, ex))

    @staticmethod
    def model_search() -> 'list':
        """
        Search model directory for all listed GPD models and return list for selection on frontend.
        """
        models = []

        try:
            models = os.listdir('./femtomesh/data/models/')
            assert len(models) > 0, 'No models returned.'

        except AssertionError as ex:
            print('{0}:{1} >\tNo model files found.'.format(ex, __name__))

        return models

    @staticmethod
    def gpd_search(model: 'str') -> 'list':
        """
        Search model directory for all listed GPD files and return list for selection on frontend.
        """
        models = []
        try:
            models = os.listdir('./femtomesh/data/models/{}/'.format(model))
            assert len(models) > 0, 'No GPD models returned.'

        except AssertionError as ex:
            print('{0}:{1} >\tNo GPD files found.'.format(ex, __name__))

        return models

    @staticmethod
    def parallelize(func, data_frame, cpu_count):
        assert cpu_count <= mp.cpu_count()

        n_splits = cpu_count
        split = np.array_split(data_frame.x.unique(), n_splits)
        pool = mp.Pool(n_splits)
        df = pd.concat(pool.map(func, split))
        pool.close()
        pool.join()

        return df

    def grid_search(self, x_vector: 'numpy.array') -> 'pandas.DataFrame':
        """
        The main worker function of Femtomesh for 1-dimensional plots. The method builds the model for a given
        kinematic region and then determines the UP and DOWN quark GPDs for every value of x at a given Q2. A
        new DataFrame built containing the new information and returned to the user for plotting and analysis.
        """
        gpd_value_u = np.array([])
        gpd_value_d = np.array([])

        dff = self.data_frame
        x_value = x_vector

        for x in x_value:
            sub_df = dff[dff.x == x][['Q2', 'gpd_u', 'gpd_d']]

            gpd_representaion_u = interpolate.splrep(sub_df.Q2.to_numpy(), sub_df.gpd_u.to_numpy(), s=0)
            gpd_representaion_d = interpolate.splrep(sub_df.Q2.to_numpy(), sub_df.gpd_d.to_numpy(), s=0)

            gpd_value_u = np.append(gpd_value_u, interpolate.splev(self._q2, gpd_representaion_u, der=0))
            gpd_value_d = np.append(gpd_value_d, interpolate.splev(self._q2, gpd_representaion_d, der=0))

        d_frame = pd.DataFrame({'x': x_value,
                                'u': gpd_value_u,
                                'd': gpd_value_d})

        d_frame['xu'] = d_frame['x'] * d_frame['u']
        d_frame['xd'] = d_frame['x'] * d_frame['d']

        self.data_frame = d_frame

        return d_frame

    def process(self, multiprocessing=False, cpu_count=4, dim=1):
        """
        Processes submitted job and determines which process should handle it.
        """
        try:
            assert self.model_generated is True

            if dim == 1:
                if multiprocessing is True:
                    df = self.parallelize(self.grid_search, self.data_frame, cpu_count)
                else:
                    df = self.grid_search(self.data_frame.x.unique())

            else:
                if multiprocessing is True:
                    df = self.parallelize(self.grid_search_2D, self.data_frame, cpu_count)
                else:
                    df = self.grid_search_2D(self.data_frame.x.unique())

            return df
        except AssertionError as ex:
            print('{0}:{1} Must build dataframe model before processing mesh search.'.format(__name__, ex))
            raise

    def calculate_gpd_value(self, x):
        gpd_value_u = np.array([])
        gpd_value_d = np.array([])

        sub_df = self.data_frame[self.data_frame.x == x][['Q2', 'gpd_u', 'gpd_d']]

        gpd_representaion_u = interpolate.splrep(sub_df.Q2.to_numpy(), sub_df.gpd_u.to_numpy(), s=0)
        gpd_representaion_d = interpolate.splrep(sub_df.Q2.to_numpy(), sub_df.gpd_d.to_numpy(), s=0)

        gpd_value_u = np.append(gpd_value_u, interpolate.splev(self._q2, gpd_representaion_u, der=0))
        gpd_value_d = np.append(gpd_value_d, interpolate.splev(self._q2, gpd_representaion_d, der=0))

        return gpd_value_u, gpd_value_d

    def grid_search_2D(self, vector: 'numpy.array') -> 'pandas.DataFrame':
        """
        The main worker function of Femtomesh for 2-dimensional plots. The method builds the model for a given
        kinematic region and then determines the UP and DOWN quark GPDs for every value of x at a given Q2. A
        new DataFrame built containing the new information and returned to the user for plotting and analysis.
        """
        x_value = np.array([])
        xbj_value = np.array([])
        q2 = np.array([])
        gpd_value_u = np.array([])
        gpd_value_d = np.array([])

        _df = self.data_frame

        iters = [(i, j) for i in vector for j in _df['xbj'].unique()]
        total_elements = 0.5 * len(list(chain.from_iterable(iters)))
        progress_bar = tqdm(total=total_elements)

        for x, xbj in iters:
            x_value = np.append(x_value, x)
            xbj_value = np.append(xbj_value, xbj)
            q2 = np.append(q2, self._q2)

            sub_df = _df[(_df.x == x) & (_df.xbj == xbj)][['Q2', 'gpd_u', 'gpd_d']]

            gpd_representaion_u = interpolate.splrep(sub_df.Q2.to_numpy(), sub_df.gpd_u.to_numpy(), s=0)
            gpd_representaion_d = interpolate.splrep(sub_df.Q2.to_numpy(), sub_df.gpd_d.to_numpy(), s=0)

            gpd_value_u = np.append(gpd_value_u, interpolate.splev(self._q2, gpd_representaion_u, der=0))
            gpd_value_d = np.append(gpd_value_d, interpolate.splev(self._q2, gpd_representaion_d, der=0))
            progress_bar.update(1)
        progress_bar.close()

        d_frame = pd.DataFrame({'x': x_value,
                                'xbj': xbj_value,
                                'q2': q2,
                                'u': gpd_value_u,
                                'd': gpd_value_d})

        d_frame['xu'] = d_frame['x'] * d_frame['u']
        d_frame['xd'] = d_frame['x'] * d_frame['d']

        self.model_generated = True
        self.data_frame = d_frame

        return d_frame
