# Eliot R. Smith's agent based model in Python

![Build Status](https://api.travis-ci.org/kubikb/smith_agent_model.svg?branch=master)

This Python package is able to simulate Eliot R. Smith's agent-based model from the following article:

> Smith, Eliot R. "Evil Acts and Malicious Gossip A Multiagent Model of the Effects of Gossip in Socially Distributed Person Perception." Personality and Social Psychology Review 18.4 (2014): 311-325.

You can provide arbitrary input values and functions so that the model can be tested under highly varying conditions (such as impression asymmetry)

## Installing it
Clone it and execute `python setup.py install`.

## Running it
The main component of this package is a Python class named `MaliciousGossipModel`. You can initialize it with the following arguments (default values after the argument names):
```
n_targets=20,                           # Number of targets
n_observers=20,                         # Number of observers
ratio_evil_targets=0.2,                 # Ratio of evil targets
normal_target_behavior_mean=0.5,        # Mean of normal target behavior distribution
evil_target_negative_act_prob=0.05,     # Probability of negative acts by evil targets
evil_target_negative_act_mean=-4.5,     # Mean of evil target behavior distribution when committing a negative act
evil_target_nonnegative_act_mean=0.75,  # Mean of evil target behavior distribution when committing a non-negative act
behavior_stddev=1,                      # Standard deviation of all behavior distributions
initial_impressions=None,               # Matrix of initial impressions. If not provided, a matrix of zeroes is used
directed_gossip=True,                   # Indicate whether directed or interesting gossip should be used
disregard_threshold=None,               # Disregard differing impressions above this threshold
integration_rule=None,                  # Function to use to integrate new impressions. If not provided, equally weighted averaging is used
interaction_likelihood_rule=None        # Function to determine the probability of interactions between observers and targets. If not provided, Luce choice function is used.
```

You can run the model by executing the `run()` function.

An example of running the model for 40 time periods with default input values would look like this:
```
from smith_agent_model.model import MaliciousGossipModel
import logging
import numpy as np

# Turn on basic logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s %(name)s %(filename)s:%(lineno)s - %(message)s',
                    level=logging.DEBUG)

if __name__ == "__main__":
    model = MaliciousGossipModel()
    for i in range(0,40):
        model.run()
        print np.mean(model.observer_target_impressions)
```