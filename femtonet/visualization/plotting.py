from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.io import show

import numpy as np

def in_notebook():
    """ Utility function to determineif code is run in notebook or note """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def plot_compton_form_factor_ensemble(data, title='', x_label='x', y_label='y', plot_width=900, plot_height=400, line_width=2, line_color='#145deb'):
    """ Utility function to dispaly an interactive line plot of the Compton Form Factor ensemble
            Args:
                data (numpy array): (ensemble row) x (CFF) array containing data.
                title (str): Plot title
                x_label (str): X-axis label
                y_label (str): Y-axis label
                plot_width (int): displayed plot width in pixels
                plot_height (int): displayed plot width in pixels
                line_width (int): line width
                line_color (str): string of line color hex value, ex. '#145deb'

            Returns:
                None
    """

    if in_notebook() == True:
        from bokeh.plotting import output_notebook

        output_notebook()
        print('code running in notebook ...')
    else:
        print('code running in terminal ...')
    
    _x_axis = np.array(['ReH', 'ImH', 'ReE', 'ImE', 'ReHt', 'ImHt', 'ReEt', 'ImEt'])

    x_axis = np.tile(_x_axis, (data.shape[0], 1))

    source = ColumnDataSource(data=dict(
        xs=x_axis.tolist(),
        ys=data.tolist()
    ))

    plot = figure(plot_width=plot_width, plot_height=plot_height, tools=["hover", "box_zoom"], title=title, x_range=_x_axis, x_axis_label=x_label, y_axis_label=y_label)
    renderer = plot.multi_line(
        xs='xs', 
        ys='ys', 
        source=source, 
        line_width=line_width, 
        line_alpha=0.3,
        line_color="#eba214",
        hover_line_color=line_color,
        hover_line_alpha = 1.0
    )

    show(plot)


if __name__=='__main__':
    data = np.load('Cff_dvcs_pred.npy')
    plot_compton_form_factor_ensemble(data=data, x_label='CFF', y_label='Value')
