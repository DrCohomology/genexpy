# Generalizability of experimental studies

<figure>
  <img alt="" src="demos/Categorical encoders/figures/encoders_nstar_alpha_delta.png" title="Nstar" align="center">
  <figcaption><em>
        Estimated number of datasets necessary to achieve generalizable results for the categorical encoders study, 
        conditioned on desired generalizability level (left) and similarity threshold between rankings (right). 
        The hue shows three different goals of a study.
  </em></figcaption>
</figure>


This repository is the official implementation of [Generalizability of Experimental Studies]().

The `genexpy` Python module contains several functions to efficiently work on rankings, probability distributions of 
rankings, and kernels for rankings.

The demos contain the generalizability analysis performed in the paper on categorical encoders and Large Language Models.

## Usage

As specified in `setup.py`, `genexpy` requires `Python 3.11.8`, 
which you can get [here](https://www.python.org/downloads/release/python-3118/).
After that, you can install `genexpy` from terminal in one of the following ways. 

### Installation via pip 
_To ensure anonymity, the module is yet to be published: this installation method is not working._
```bash
pip install genexpy
```

### Installation via git
(1) Clone the repo; (2) navigate to the cloned repo; (3) install the module.
```bash
git clone https://github.com/DrCohomology/genexpy.git     
cd genexpy                                                           
pip install .                                             
```

### Cloning without installation
(1) Clone the repo; (2) navigate to the cloned repo; (3) create a virtual environment `venv`; (4) activate `venv`; 
(5) install the requirements.
```bash
git clone https://github.com/DrCohomology/genexpy.git     
cd genexpy
python -m venv venv                                       
venv\Scripts\activate.bat                                 
pip install -r requirements.txt                          
```
