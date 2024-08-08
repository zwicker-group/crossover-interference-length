# draft_interference_measures

## JavaScript version

The JavaScript version should run in any decently modern browser by simply opening the
file `measure_CO_interference.html`.
There is the option to load example data using the button "Load example data", but you can also paste your own CO
positions (as explained in the file).
The button "Calculate quantities" then calculates all quantities (and the respective standard deviations of the mean)
and displays the result below.

The calculation of the coefficient of coincidence curve is based on the implementation given in
https://github.com/mwhite4/MADpatterns/blob/master/interval_analysis.m

The positions need to be specified in fractions of the total chromosome length, so they are numbers between 0 and 1. The
chromosome length can be specified in units of micrometer (µm) for cytological data, units of megabases (Mb) for genetic
data, or left set to 1 to obtain results normalized to the chromosome length.

## Python version

### Installation

The python version of the code requires the `numpy` and `scipy` packages, which can be installed using the requirements
files.

```bash
pip install -r requirements.txt
```

### Usage

Exemplary data from A. thalania (genetic data) from [1, 2] for various genotypes (wild-type, HEI10oe, zyp1 and zyp1
HEI10oe), both male/female and all five chromosomes (index 1..5) is contained in the folder `data/`.
The associated genetic chromosome lengths are stored in `data/chromosome_lengths.csv`, whereas the CO positions for the
samples are stored in files following the scheme `data/A_thalania_[genotype]_[sex]_[chromosome_index].csv`.
Running the python script displays summary statistics for various genotypes and sexes:


```bash
python example.py
```

The functions are implemented in the python script

```
measure_CO_interference.py
```

The summary statistics (mean number of COs, interference length, normalized interference
length, gamma shape parameter and interference distance) can be used by calling the python function

```
get_interference_measures(positions, my_params)
```

and the coefficient of coincidence can be computed using

```
coefficient_of_coincidence(positions, my_params)
```

The positions need to be specified in fractions of the total chromosome length, so they are numbers between 0 and 1. The
chromosome length can be specified in units of micrometer (µm) for cytological data, units of megabases (Mb) for genetic
data, or left set to 1 to obtain results normalized to the chromosome length.

The parameter dictionary my_params has the following relevant parameters
`length` which gives the length of the chromosome (or SC) in the respective unit. `NUM_BOOTSTRAP_SAMPLES` gives the
number of bootstrap samples to compute the standard deviation of the mean. 
`NUM_INTERVALS` gives the number of bins of the coefficient of coincidence curve and 
`COC_IMPLEMENTATION` can be set to either **mwhite** to compute the curve according to
the implementation given in 
https://github.com/mwhite4/MADpatterns/blob/master/interval_analysis.m
while setting it to **mernst** uses an implementation that does not require pre-binning
and is described in the publication in SI-1B.

[1] Durand, S., Lian, Q., Jing, J. et al. Joint control of meiotic crossover patterning by the synaptonemal complex and
HEI10 dosage. Nat Commun 13, 5999 (2022). https://doi.org/10.1038/s41467-022-33472-w
[2] Singh, D. K., Lian, L., Durand, S. et al. Heip1 is required for efficient meiotic crossover implementation and is 
conserved from plants to humans, Proceedings of the National Academy of Sciences 120, e2221746120 (2023). 
https://doi.org/10.1073/pnas.2221746120