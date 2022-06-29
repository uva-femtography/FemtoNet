import sys
import asyncio
import aiohttp
import nest_asyncio

import pandas as pd
import numpy as np

from femtonet.femtonet_api.cubedata import CubeData

class FemtoNetAPI:
    def __init__(self):
        self.api_base_url = 'http://femtography.uvadcos.io/api/'
        self.cube_data = CubeData()

    def _api_async(func):
        """ Decorator used to generate an asynchronous request

            Args:
                func (async function): An asynchronous API request function.

            Returns:
                CubeData: A custom data class containing a list of columns and a Pandas DataFrame with the requested GPD information.
        
        """
        def request_wrapper(self, *args, **kwargs):
            if 'ipykernel' in sys.modules:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self._to_task(func(self, *args, **kwargs), True, loop))
                else:
                    nest_asyncio.apply(loop)
                    return asyncio.run(self._to_task(func(self, *args, **kwargs), True, loop))

            else:
                return asyncio.run(func(self, *args, **kwargs))
                
        return request_wrapper

    @_api_async
    async def data_query(self, model:str, gpd:str, xbj:float, t:float, qs:float):
        async with aiohttp.ClientSession() as session:
            url = self.api_base_url = 'http://femtography.uvadcos.io/api/{model}/{gpd}/{xbj}/{t}/{qs}'.format(model=model, gpd=gpd.upper(), xbj=xbj, t=t, qs=qs)
            
            async with session.get(url) as resp:
                print('<Query::submitted - {model}, {gpd}, {xbj}, {t}, {qs}>'.format(model=model, gpd=gpd.upper(), xbj=_xbj, t=_t, qs=qs))
                response = await resp.json(content_type='text/html')
                return self._set_data_cube(response)


    def _to_task(self, future, as_task, loop):
        return loop.create_task(future)

    def _set_data_cube(self, response):
        self.cube_data.data = pd.DataFrame.from_dict(response)
        self.cube_data.columns = self.cube_data.data.columns

        return self.cube_data

    def _find_nearest_value(self, data:np.array, value:float)->int:
        index = np.abs(data - value).argmin()
        return data[index]
        
        
