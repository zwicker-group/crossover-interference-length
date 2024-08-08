from itertools import product
from random import sample
from scipy.stats import gamma
import numpy as np

default_parameters = {'length': 1, 'NUM_BOOTSTRAP_SAMPLES': 100, 'MAX_SIZE': 1e7, 'NUM_INTERVALS': 15,
                      'COC_IMPLEMENTATION': 'mernst'}


def init_parameters(new_parameters):
    """
    Function to initialize the parameter dictionary.
    :param new_parameters: Dictionary with parameters to overwrite the default parameters.
    :return: Dictionary with all keys in default parameters, but overwriting the values explicitly given in param.
    """
    updated_parameters = default_parameters.copy()
    updated_parameters.update(new_parameters)
    return updated_parameters


def get_distance_distribution(data_positions: np.array, only_adjacent_crossover: bool = False):
    """
    Return a sorted 1D array with the distribution of distances between crossover.
    :param data_positions: 2D array of crossover position.
    :param only_adjacent_crossover: True if only compute distances of adjacent crossovers, False otherwise.
    :return: sorted 1D array with all distances between crossovers.
    """
    if only_adjacent_crossover:
        distances = data_positions[:, 1:] - data_positions[:, :-1]
        distances = distances.flatten()
        distances = distances[~np.isnan(distances)]
    else:
        distances = []
        number_of_samples = data_positions.shape[0]
        for i in range(number_of_samples):
            if ~np.isnan(data_positions[i, 1]):  # check whether the chromosome has at least two crossover
                positions_sample = np.sort(data_positions[i])
                positions_sample = positions_sample[~np.isnan(positions_sample)]
                for k1 in range(len(positions_sample)):
                    for k2 in range(k1 + 1, len(positions_sample)):
                        distances += [positions_sample[k2] - positions_sample[k1]]
    return np.sort(distances)


def get_expected_distances(data_positions: np.array, parameters: dict, only_adjacent_crossover: bool = False):
    """
    Return a sorted 1D array with the expected distribution of distances between observed crossover positions.
    The maximal length of the array is given by the parameter parameters['maximal_number_of_expected_pairs'].
    :param data_positions: 2D array of crossover position.
    :param only_adjacent_crossover: True if only compute distances of adjacent crossovers, False otherwise.
    :param parameters: Dictionary with parameters.
    :return: sorted 1D array with expected distances between crossovers.
    """
    data_flat = data_positions.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]
    number_of_samples = np.shape(data_positions)[0]
    if len(data_flat) == 0:  # There are no crossovers
        distances_expected = np.nan
    elif only_adjacent_crossover:
        mean_crossover_per_chromosome = len(data_flat) / number_of_samples
        sample_size = parameters['MAX_SIZE']
        crossover_count_distribution = np.random.poisson(mean_crossover_per_chromosome, sample_size)
        data_choice = np.zeros((sample_size, np.max(crossover_count_distribution))) * np.nan
        for i in range(sample_size):
            crossover_count = crossover_count_distribution[i]
            data_choice[i, :crossover_count] = np.random.choice(data_flat, crossover_count)
        distances_expected = get_distance_distribution(data_choice, only_adjacent_crossover=True)
    else:
        if len(data_flat) < np.sqrt(parameters['MAX_SIZE']):
            possible_pairs = np.array(list(product(data_flat, data_flat)))  # get all possible pairs of crossovers
        else:
            # If the number of possible pairs of crossover positions is larger than a certain threshold
            # we randomly choose expected pairs from the possible pairs.
            possible_pairs = np.sort(np.random.choice(data_flat, (parameters['MAX_SIZE'], 2)), axis=1)
        distances_expected = np.abs(possible_pairs[:, 1] - possible_pairs[:, 0])
        distances_expected = distances_expected[distances_expected > 0]  # Delete cases with 0 distance
    return np.sort(distances_expected)


def get_data_sample(data_positions: np.array, sample_size: int = None):
    """
    Generate a random sub-set of samples with a certain sample size.
    :param data_positions: 2D array of crossover positions.
    :param sample_size: sample size which should be lower than number of given samples in data_positions.
    :return: 2D array of crossover positions with sample_size samples.
    """
    number_of_samples = np.shape(data_positions)[0]
    if sample_size is None or sample_size > number_of_samples:
        return data_positions
    else:
        return data_positions[sample(range(np.shape(data_positions)[0]), int(sample_size)), :]


def number_of_crossovers_per_chromosome(data_positions: np.array):
    """
    Compute the mean number of crossovers per chromosome, the standard deviation and the standard deviation of the mean.
    :param data_positions: 2D array of crossover position
    :return: tuple of mean, standard deviation, standard deviation of the mean
    """
    number_of_samples = np.shape(data_positions)[0]
    crossovers_per_chromosome = np.zeros(number_of_samples)
    for i in range(number_of_samples):
        crossovers_per_chromosome[i] = np.sum(~np.isnan(data_positions[i]))
    mean = np.mean(crossovers_per_chromosome)
    # std_deviation = np.std(crossovers_per_chromosome)
    std_deviation_mean = np.std(crossovers_per_chromosome) / np.sqrt(number_of_samples)
    return mean, std_deviation_mean


def interference_length(data_positions: np.array, parameters: dict):
    """
    Compute the interference length for a data set for one chromosome with a certain length.
    :param data_positions: 2D array of crossover position
    :param parameters: Dictionary with parameters.
    :return: interference length in units of the input length
    """
    data_flat = data_positions.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]
    number_of_samples = np.shape(data_positions)[0]
    mean_crossover_per_chromosome = len(data_flat) / number_of_samples
    distances_observed = get_distance_distribution(data_positions)
    distances_expected = get_expected_distances(data_positions, parameters)
    pairs_per_chromosome_observed = len(distances_observed) / number_of_samples
    pairs_per_chromosome_expected = mean_crossover_per_chromosome ** 2 / 2
    if len(data_flat) == 0:
        l_int = 1
    else:
        fraction_observed_pairs = pairs_per_chromosome_observed / pairs_per_chromosome_expected
        if len(distances_observed) > 0:
            l_int = (np.mean(distances_observed) - 1) * fraction_observed_pairs + 1 - np.mean(distances_expected)
        elif len(distances_expected) > 0:
            l_int =  1 - np.mean(distances_expected)
        else:
            l_int = 1
    return l_int * parameters['length']


def gamma_shape_parameter(data_positions: np.array, parameters: dict):
    """
    Compute the gamma shape parameter ny for a data net for one chromosome with a certain length.
    https://stackoverflow.com/questions/31359017/how-to-get-error-estimates-for-fit-parameters-in-scipy-stats-gamma-fit
    :param data_positions: 2D array of crossover position (does not influence the value of the gamma shape parameter)
    :param parameters: Dictionary with parameters.
    :return: gamma shape parameter alpha in units of the input length
    """
    distances_observed = get_distance_distribution(data_positions, only_adjacent_crossover=True)
    distances_observed = distances_observed[distances_observed > 0]
    if len(distances_observed) > 1:
        shape_parameter = gamma.fit(distances_observed, floc=0)[0]
    else:
        shape_parameter = np.nan
    return shape_parameter


def coefficient_of_coincidence_mwhite(data_positions: np.array, parameters: dict):
    """
    Compute the coefficient of coincidence curve.
    Implementation based on https://github.com/mwhite4/MADpatterns/blob/master/interval_analysis.m
    :param data_positions: 2D array of crossover position
    :param parameters: Dictionary with parameters.
    :return: coefficient_of_coincidence
    """
    number_of_samples = np.shape(data_positions)[0]
    num_of_intervals = parameters['NUM_INTERVALS']
    event_per_interval = np.zeros((number_of_samples, num_of_intervals))
    pattern_per_interval = np.zeros((number_of_samples, num_of_intervals))
    observed_pattern_frequency = np.zeros((num_of_intervals - 1, num_of_intervals))
    expected_pattern_frequency = np.zeros((num_of_intervals - 1, num_of_intervals))
    coc_x = parameters['length'] * np.linspace(0, 1 - 1 / num_of_intervals, num_of_intervals)
    for i in np.arange(number_of_samples):
        index_of_crossover = np.where(~np.isnan(data_positions[i, :]))[0]
        for x_pos in data_positions[i, index_of_crossover]:
            if x_pos * num_of_intervals < num_of_intervals:
                event_per_interval[
                    i, int(x_pos * num_of_intervals)] += 1.
                pattern_per_interval[
                    i, int(x_pos * num_of_intervals)] = 1.

    observed_pattern_frequency_per_interval = np.mean(pattern_per_interval, axis=0)
    for i in range(num_of_intervals - 1):
        for j in range(i + 1, num_of_intervals):
            observed_pattern_frequency[i, j - i] = np.sum(
                pattern_per_interval[:, i] * pattern_per_interval[:, j]) / number_of_samples
            expected_pattern_frequency[i, j - i] = \
                observed_pattern_frequency_per_interval[i] * observed_pattern_frequency_per_interval[j]
    np.seterr(invalid='ignore')
    coc_per_distance = np.divide(observed_pattern_frequency, expected_pattern_frequency)
    total_coc_per_distance = np.nansum(coc_per_distance, axis=0)
    total_pairs_per_distance = np.copy(coc_per_distance)
    total_pairs_per_distance[~np.isnan(total_pairs_per_distance)] = 1.
    total_pairs_per_distance = np.nansum(total_pairs_per_distance, axis=0)
    coc_y = total_coc_per_distance / total_pairs_per_distance
    return coc_x, coc_y


def coefficient_of_coincidence_mernst(data_positions: np.array, parameters: dict):
    """
    Compute the coefficient of coincidence curve.
    Implementation according to Marcel Ernst (2023)
    :param data_positions: 2D array of crossover position
    :param parameters: Dictionary with parameters.
    :return: coefficient_of_coincidence
    """
    number_of_samples = np.shape(data_positions)[0]
    number_of_intervals = parameters['NUM_INTERVALS']
    distances_observed = get_distance_distribution(data_positions)
    distances_expected = get_expected_distances(data_positions, parameters)
    histogram_observed = np.histogram(distances_observed, bins=np.linspace(0, 1, number_of_intervals + 1))
    histogram_expected = np.histogram(distances_expected, bins=np.linspace(0, 1, number_of_intervals + 1))

    coc_x = histogram_observed[1][:-1] + histogram_expected[1][1] / 2

    data_flat = data_positions.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]

    mean_crossover_per_chromosome = len(data_flat) / number_of_samples
    pairs_per_chromosome_observed = len(distances_observed) / number_of_samples
    pairs_per_chromosome_expected = mean_crossover_per_chromosome ** 2 / 2
    if len(data_flat) == 0:
        fraction_observed_pairs = np.nan
    else:
        fraction_observed_pairs = pairs_per_chromosome_observed / pairs_per_chromosome_expected
    histogram_expected = histogram_expected[0] / np.nansum(histogram_expected[0])
    histogram_observed = fraction_observed_pairs * histogram_observed[0] / np.nansum(histogram_observed[0])
    np.seterr(divide='ignore', invalid='ignore')
    # if histogram_expected has entries with 0 np.NaN is the expected output
    coc_y = histogram_observed / histogram_expected
    return coc_x, coc_y


def coefficient_of_coincidence(data_positions: np.array, parameters: dict):
    """
    Compute the coefficient of coincidence curve. Choosing implementation by
    setting parameters['COC_IMPLEMENTATION'] to either 'mwhite' or 'mernst'.
    :param data_positions: 2D array of crossover position
    :param parameters: Dictionary with parameters.
    :return: coefficient_of_coincidence
    """
    if parameters['COC_IMPLEMENTATION'] == 'mwhite':
        return coefficient_of_coincidence_mwhite(data_positions, parameters)
    elif parameters['COC_IMPLEMENTATION'] == 'mernst':
        return coefficient_of_coincidence_mernst(data_positions, parameters)


def get_x_from_linearization(x_1, x_2, y_1, y_2, y_th):
    """
    Linearize from (x_1, y_1) to (x_2, y_2) to get the x-value where the line crosses y_th.
    :return: x_th
    """
    m = (y_2 - y_1) / (x_2 - x_1)
    b = y_2 - m * x_2
    x_th = (y_th - b) / m
    return x_th


def interference_distance(data_positions: np.array, parameters: dict):
    """
    Compute the interference distance d_coc based on the point where the coefficient of coincidence firstly exceeds 0.5
    :param data_positions: 2D array of crossover position
    :param parameters: Dictionary with parameters.
    :return: interference distance d_coc
    """
    coc_x, coc_y = coefficient_of_coincidence(data_positions, parameters)

    if np.isnan(coc_y).all():
        d_coc = np.nan
    elif np.nanmax(coc_y) > 0.5:
        index_first_exceeds = np.where(coc_y > 0.5)[0][0]
        if index_first_exceeds == 0:
            d_coc = get_x_from_linearization(0, coc_x[0], 0, coc_y[0], 0.5)
        elif np.isnan(coc_y[index_first_exceeds - 1]):
            d_coc = coc_x[index_first_exceeds]
        else:
            d_coc = get_x_from_linearization(coc_x[index_first_exceeds - 1], coc_x[index_first_exceeds],
                                            coc_y[index_first_exceeds - 1], coc_y[index_first_exceeds], 0.5)
    else:
        d_coc = 1
    return d_coc * parameters['length']


def bootstrap_half_samples(measure_func, data_positions: np.array, parameters: dict):
    """
    Compute the mean and standard deviation of some interference measure
    :param measure_func: function to compute a specific interference measure. It needs to take the parameters:
        data_positions, parameters and length.
    :param data_positions: 2D array of crossover positions
    :param parameters: dictionary with parameters
    :return: tuple of mean and standard deviation of the mean of a certain interference measure
    """
    interference_measure_mean = measure_func(data_positions, parameters)
    interference_measure_values = np.zeros(parameters['NUM_BOOTSTRAP_SAMPLES'])
    number_of_samples = np.shape(data_positions)[0]
    sample_size = int(number_of_samples / 2 + 0.5)
    for i in range(parameters['NUM_BOOTSTRAP_SAMPLES']):
        data_sample = get_data_sample(data_positions, sample_size)
        interference_measure_values[i] = measure_func(data_sample, parameters)
    interference_measure_std_deviation_mean = np.nanstd(interference_measure_values) / np.sqrt(2)
    return interference_measure_mean, interference_measure_std_deviation_mean


def get_interference_measures(data_positions: np.array, parameters: dict):
    """
    Function to wrap the computation of all interference measures and return the results in a dictionary.
    :param data_positions: 2D numpy array with crossover positions for all samples of a specific chromosome of the
    relative positions along the respective chromosome (with float values from 0 to 1). First dimension is sample size,
    and second dimension is the positions of the crossovers (size at least the maximal number of crossovers).
    The array should use np.NaN for 'no crossover'.
    Example: [[0.3, 0.6, 0.8], [0.5, 0.7, np.nan], [0.4, np.nan, np.nan]]
    :param parameters: Dictionary with parameters overwriting default_parameters
    :return: Dictionary with results.
    """
    meanN, meanN_std = number_of_crossovers_per_chromosome(data_positions)
    Lint, Lint_std = number_of_crossovers_per_chromosome(data_positions)
    LintNorm = Lint * meanN / parameters['length']
    LintNorm_std = np.sqrt((Lint_std * meanN)**2 + (Lint * meanN_std)**2) / parameters['length']
    gamma, gamma_std = bootstrap_half_samples(gamma_shape_parameter, data_positions, parameters)
    dCoC, dCoC_std = bootstrap_half_samples(interference_distance, data_positions, parameters)
    return {r'Mean number of CO per chromosome $\langle N \rangle$': (meanN, meanN_std),
            r'Interference length $L_\mathrm{int}$': (Lint, Lint_std),
            r'Normalized Interference length $L_\mathrm{int}^\mathrm{norm}$': (LintNorm, LintNorm_std),
            r'Gamma shape parameter $\nu$': (gamma, gamma_std),
            r'Interference distance $d_\mathrm{CoC}$': (dCoC, dCoC_std)}


