# Nuclear Networks
#### v0.9.0
![siwif_logo](figures/siwif_logo.png)

`nuclearnets` is a python library for Cross Section prediction in Deeply Virtual Compton Scattering Experiments.

### Installation
```bash
git clone https://github.com/uva-femtography/nuclear-networks.git
cd nuclear-networks
make user
```
(If planning to contribute to `nuclearnets` development, use `make dev` instead)

### Getting Started
`nuclearnets` includes 2 pretrained models that can be used to make predictions right away. These are found in `nuclear-networks/saves/main_UU_sincos` and `nuclear-networks/saves/main_LU_sincos`.

**For all scripts, use the -h flag to get more information about command line arguments**
#### Plotting
##### Cross Section vs. Phi
`python -m nuclearnets.plotting.uu_vs_phi`

`python -m nuclearnets.plotting.lu_vs_phi`


![xsx_vs_phi_example](figures/xsx_predictions/lu_defurne34_1.png)

##### Cross Section vs Xi
`python -m nuclearnets.plotting.xsx_vs_xi`

![xsx_vs_xi_example](figures/xsx_predictions/lu_hallB_last_bin.png)

#### Bulk Predictions
##### Predict on Dataset
Use a pretrained model to make predictions on the built-in datasets.

`python -m nuclearnets.predict.predict_on_dataset`

##### Predict on CSV
Use a pretrained model to make predictions on your own dataset.

`python -m nuclearnets.predict.predict_on_csv`

The column format of the input dataset is expected to be:

xbj | t | Q2 | k0 | phi | xsx_type
--- | --- | --- | --- | --- | --- |

where xsx_type is an integer representation of the Cross Section type. Currently, we support:
1. Unpolarized (UU)
2. Polarized (LU)

### Training new Models
You can customize/retrain the model architecture using `python -m nuclearnets.train`. The model is small enough to be trained on a cpu, although a GPU would be helpeful.