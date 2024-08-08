from numpy import genfromtxt
from measure_CO_interference import *

if __name__ == "__main__":
    my_params = {'COC_IMPLEMENTATION': 'mwhite'}
    my_params = init_parameters(my_params)
    # Load chromosome lengths for A. thalania (genetic data from Durand2022) from file
    chromosome_lengths = genfromtxt("data/chr_lengths.csv", delimiter=',')
    list_of_genotypes = ['wt', 'HEI10oe', 'zyp1', 'zyp1 HEI10oe']
    list_of_sex = ['male', 'female']
    for genotype_idx, genotype in enumerate(list_of_genotypes):
        for sex_idx, sex in enumerate(list_of_sex):
            for chr_idx in range(len(chromosome_lengths)):
                # Load data of CO position for A. thalania data of respective genotype, sex and chromosome number
                filename = 'A_thalania_' + genotype + '_' + sex + '_' + str(chr_idx) + '.csv'
                data_x = genfromtxt("data/" + filename, delimiter=',')
                # One needs to set the chromosome length, otherwise it is assumed to be 1.
                my_params['length'] = chromosome_lengths[chr_idx]/1e6
                # compute summary statistics measures, e.g. mean number of COs, (normalized), interference length, interference distance and gamma shape parameter
                results = get_interference_measures(data_x, parameters=my_params)
                # Calculating the coefficient of coincidence curve
                coc_x, coc_y = coefficient_of_coincidence(data_x, parameters=my_params)
                print(genotype, sex, chr_idx+1, results)
                print('CoC values', coc_x, coc_y)
