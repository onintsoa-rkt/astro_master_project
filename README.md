The title of this project is ``Environment and galaxy properties with Simba".
The objective is to analyse the galaxy clustering dependence on three galaxy properties (colour, specific star formation rate and age).

-----

## What this project is about ?

The ultimate goal is to analyse the relationship between galaxy properties and the clustering of these galaxies. This project includes:
- measurements of galaxy clustering as a function of colour, sSFR and age using the two-point correlation function.

Landy-Szalay estimator: $$\xi_{LS}(r) = \dfrac{DD-2DR+RR}{RR}$$

Projected correlation fucntion: $$w(r_p) = 2\int_{0}^{\infty} d\pi \xi(r_p, \pi)$$

- different data analysis including resolution analysis, interpolation and oversampling, statistical analysis, comparison of replicated data with real data, comparison of results with simulation boxes of different sizes, comparison of different uncertainty estimations.

-----

## What is the data ?

We use output data from the [SIMBA](http://simba.roe.ac.uk/) cosmological simulation.
[CAESAR](https://caesar.readthedocs.io/en/latest/) is the pythoon package that was developed to specifically analyse outputs from cosmological simulations. Taking a SIMBA snapshot as input, CAESAR outputs a compact and portable hdf5 catalogue that contains all galaxy and halo information with various pre-computed key properties.
Data are publicly available at [http://simba.roe.ac.uk/simdata/](http://simba.roe.ac.uk/simdata/).


## What this repository contains ?

- `functions.py`: reusable helper functions used across notebooks
- `notebooks/`: jupyter notebooks containing analyses
- `environment.yml` → environment setup

-----

## How to run this project ?

### Create environment
```bash
conda env create -f environment.yml
conda activate caesar_env
```

---

### Additional installation steps
#### Install pygadgetreader
```bash
git clone https://github.com/dnarayanan/pygadgetreader.git
cd pygadgetreader
python setup.py install
```

#### Install caesar
```bash
git clone https://github.com/dnarayanan/caesar.git
cd caesar
python setup.py install
```

---

## Requirements

- Python 3.10
- Jupyter Notebook
- Packages listed in `requirements.txt`
