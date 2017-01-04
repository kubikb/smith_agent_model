from unittest import TestCase
import numpy as np
import smith_agent_model.helpers as helpers


class TestHelpers(TestCase):
    def test_luce_array(self):
        mu, sigma = 0, 1
        sample_array = np.random.normal(mu, sigma, 1000)
        probability_array = helpers.luce(sample_array)
        self.assertIsInstance(probability_array, np.ndarray)

    def test_average_integration_rule(self):
        mu, sigma = 0, 1
        random_old_impressions = np.random.normal(mu, sigma, 1000)
        random_new_impressions = np.random.normal(mu, sigma, 1000)
        new_impressions = helpers.average_integration_rule(old_impressions=random_old_impressions,
                                                           new_impressions=random_new_impressions)
        self.assertIsInstance(new_impressions, np.ndarray)
