## FemtoNET Software
Software tools developed in conjunction with the FemtoNET group, based at the University of Virginia, for research and analysis of Generalized Parton Distributions and Compton Form Factors.

### Installation
The FemtoNET environment is managed by conda. To install a development version first clone the repository,
```
git clone --recursive https://github.com/uva-femtography/femtonet.git
```
Change into the repo directory and install the conda environment.
```
conda create --name femtonet python=3.8 --no-default-packages
conda activate femtonet
```
The package requirements can then be installed using the following.
```
conda install -n femtonet requirements.txt
```
You can test the install for the generator by running the example script as a module.
```
python -m femtonet.generator.csplot
```
If you get a cross section plot, then the installation was a success.
