from unittest import TestCase
from smith_agent_model.model import MaliciousGossipModel, MaliciousGossipModelException

class TestModel(TestCase):

    def test_target_non_negative_behavior_mean(self):
        model = MaliciousGossipModel(evil_target_negative_act_prob=0.05,
                                     evil_target_negative_act_mean=-4.5)
        self.assertEqual(round(model.evil_target_nonnegative_act_mean, 2),
                         0.76)

    def test_target_indices(self):
        model = MaliciousGossipModel(n_targets=20,
                                     ratio_evil_targets=0.2)
        evil_target_indices = model.evil_target_indices
        self.assertEqual(evil_target_indices.shape[0],
                         4)

    def test_invalid_integration_rule(self):
        with self.assertRaises(MaliciousGossipModelException):
            model = MaliciousGossipModel(integration_rule=str)

    def test_invalid_interaction_rule(self):
        with self.assertRaises(MaliciousGossipModelException):
            model = MaliciousGossipModel(interaction_likelihood_rule=int)

    def test_generate_behaviors(self):
        num_targets = 100
        model = MaliciousGossipModel(n_targets = 100)
        self.assertEqual(model.generate_target_behaviors().shape[0],
                         num_targets)

    def test_finding_targets_for_observers(self):
        num_observers = 1000
        num_targets = 100
        model = MaliciousGossipModel(n_targets=num_targets,
                                     n_observers=num_observers)
        target_indices = model.find_targets_for_observers()
        self.assertEqual(target_indices.shape[0],
                         num_observers)

    def test_obtain_interaction_decisions(self):
        num_observers = 1000
        num_targets = 100
        model = MaliciousGossipModel(n_targets=num_targets,
                                     n_observers=num_observers)
        target_indices = model.obtain_interaction_decisions()
        self.assertEqual(target_indices.shape,
                         (num_observers, num_targets))