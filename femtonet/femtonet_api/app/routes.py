import itertools
import os
from femtomesh import femtomesh as fm
from femtodb import femtodb

from flask import Response, jsonify
from app import app


@app.route('/', methods=['GET', 'POST'])
def index():
    return "FemtoNET-api"


@app.route('/api/<model>/<gpd>')
def params(model=None, gpd=None):
    df = fm.FemtoMesh('femtomesh/data/models/model_{0}/{1}.csv'.format(model, gpd)).open()

    kinematics_array = []
    info = [{'name': model,
             't': {
                 'max': df.t.max(),
                 'min': df.t.min()
             },
             'xbj': {
                 'max': df.xbj.max(),
                 'min': df.xbj.min()
             }
             }
            ]

    for (xbj, t, q2) in list(itertools.zip_longest(df.xbj.unique(), df.t.unique(), df.Q2.unique())):
        kinematics = {}
        if xbj is not None:
            kinematics['xbj'] = xbj
        if t is not None:
            kinematics['t'] = t
        if q2 is not None:
            kinematics['q2'] = q2

        kinematics_array.append(kinematics)

    return jsonify({'kinematics': kinematics_array, 'model': info})

@app.route('/api/<model>/<gpd>/t/<float(signed=True):t>')
def update_axis_t(model='uva', gpd='GPD_H', t=None):
    df = fm.FemtoMesh('femtomesh/data/models/model_{0}/{1}.csv'.format(model, gpd)).open()
    df = df[df.t==t]

    kinematics_array = []
    info = [
        {
            'name': model,
            'xbj': {
                 'max': df.xbj.max(),
                 'min': df.xbj.min()
                 },
            'q2': {
                 'max': df.Q2.max(),
                 'min': df.Q2.min()
                 }
        }
    ]

    for (_xbj, _q2) in list(itertools.zip_longest(df.xbj.unique(), df.Q2.unique())):
        kinematics = {}
        if _xbj is not None:
            kinematics['xbj'] = _xbj
        if _q2 is not None:
            kinematics['q2'] = _q2

        kinematics_array.append(kinematics)

    return jsonify({'kinematics': kinematics_array, 'model': info})

@app.route('/api/<model>/<gpd>/xbj/<float:xbj>')
def update_axis_xbj(model='uva', gpd='GPD_H', xbj=None):
    df = fm.FemtoMesh('femtomesh/data/models/model_{0}/{1}.csv'.format(model, gpd)).open()
    df = df[df.xbj==xbj]

    kinematics_array = []
    info = [
        {
            'name': model,
            't': {
                 'max': df.t.max(),
                 'min': df.t.min()
                 },
            'q2': {
                 'max': df.Q2.max(),
                 'min': df.Q2.min()
                 }
        }
    ]

    for (_t, _q2) in list(itertools.zip_longest(df.t.unique(), df.Q2.unique())):
        kinematics = {}
        if _t is not None:
            kinematics['t'] = _t
        if _q2 is not None:
            kinematics['q2'] = _q2

        kinematics_array.append(kinematics)

    return jsonify({'kinematics': kinematics_array, 'model': info})


@app.route('/api/<model>/<gpd>/<float:xbj>/<float(signed=True):t>/<float:q2>')
def search(model='uva', gpd='GPD_H', xbj=None, t=None, q2=None):
    """
        Search API: This is used to query the GPD mesh search from the femto mesh site.
    """

    mesh = fm.FemtoMesh('femtomesh/data/models/model_{0}/{1}.csv'.format(model, gpd))

    mesh.xbj = xbj
    mesh.t = t
    mesh.q2 = q2

    try:
        assert xbj is not None
        assert t is not None
        assert q2 is not None

        mesh.build_data_frame(xbj, t)
        df = mesh.process(multiprocessing=True, dim=1)

    except AssertionError:
        return "Assertion Error"

    df.reset_index(inplace=True)
    df = df.drop(columns=['index'])

    return df.to_json(orient='records', index=True, indent=4)


@app.route('/models')
def models():
    database = femtodb.FemtoDB()
    model_list = database.get_model_list()
    return jsonify({'models': model_list})


@app.route('/download/<filename>')
def download(filename):
    path = os.path.join('download', filename)

    def generate():
        with open(path) as f:
            yield from f
        os.remove(path)

    resp = Response(generate(), mimetype='text/csv')
    resp.headers.set('Content-Disposition', 'attachment', filename='gpd_model.csv')
    return resp
