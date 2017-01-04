import numpy as np

# Luce choice function
def luce(impression_array):
    """
    Luce choice function to obtain interaction likelihoods
    :param impression_array: array of observed impression values (numpy.ndarray)
    :return: probability_array: array of interaction probabilities (numpy.ndarray)
    """
    probability_array = np.exp(3 * impression_array) / (1 + np.exp(3 * impression_array))
    return probability_array

# Equally weighted average integration rule
def average_integration_rule(old_impressions, new_impressions):
    """
    qually weighted average integration rule to incorporate new impressions
    :param old_impressions: array of old impression values (numpy.ndarray)
    :param new_impressions: array of new impression values (numpy.ndarray)
    :return: new_impression: array of new impression values (numpy.ndarray)
    """
    new_impressions = (old_impressions + new_impressions) / 2
    return new_impressions