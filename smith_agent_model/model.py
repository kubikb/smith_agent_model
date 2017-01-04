import numpy as np
import logging
import helpers


# Custom Exception
class MaliciousGossipModelException(Exception):
    pass


class MaliciousGossipModel:

    n_targets = None
    n_observers = None
    num_evil_targets = None
    normal_target_behavior_mean = None
    evil_target_indices = None
    evil_target_negative_act_mean = None
    evil_target_negative_act_prob = None
    evil_target_nonnegative_act_mean = None
    behavior_stddev = None
    observer_target_impressions = None
    integration_rule = None
    interaction_likelihood_rule = None
    directed = None
    disregard_threshold = None
    tick = 1

    def __init__(self,
                 n_targets=20,
                 n_observers=20,
                 ratio_evil_targets=0.2,
                 normal_target_behavior_mean=0.5,
                 evil_target_negative_act_prob=0.05,
                 evil_target_negative_act_mean=-4.5,
                 evil_target_nonnegative_act_mean=None,
                 behavior_stddev=1,
                 initial_impressions=None,
                 directed_gossip=True,
                 disregard_threshold=None,
                 integration_rule=None,
                 interaction_likelihood_rule=None):

        logging.info("Initializing model object...")
        self.n_targets = n_targets
        logging.info("Number of targets set to %s." % n_targets)

        self.n_observers = n_observers
        logging.info("Number of observers set to %s." % n_observers)

        self.num_evil_targets = int(round(n_targets * ratio_evil_targets))
        logging.info("Number of evil targets set to %s." % self.num_evil_targets)

        evil_target_indices = np.random.choice(n_targets, self.num_evil_targets, replace=False)
        self.evil_target_indices = evil_target_indices
        logging.info("Successfully generated evil targets!")

        self.normal_target_behavior_mean = normal_target_behavior_mean
        logging.info("Behavior mean for normal targets set to %s." % normal_target_behavior_mean)

        self.evil_target_negative_act_prob = evil_target_negative_act_prob
        logging.info("Probability of negative acts from evil targets set to %s." % evil_target_negative_act_prob)

        self.evil_target_negative_act_mean = evil_target_negative_act_mean
        logging.info("Behavior mean for evil targets' negative acts set to %s." % evil_target_negative_act_mean)

        if evil_target_nonnegative_act_mean is None:
            if evil_target_negative_act_prob != 1:
                logging.warn("You did not provide a behavioral mean for evil target nonnegative acts! "
                             "Setting it so the mean valence is equal to the behavior average of normal targets.")
                evil_target_nonnegative_act_mean = (normal_target_behavior_mean -
                                                    (evil_target_negative_act_prob * evil_target_negative_act_mean)) / \
                                                   (1 - evil_target_negative_act_prob)
            else:
                # Handling cases where the probability of negative acts is 1
                evil_target_nonnegative_act_mean = 0

        self.evil_target_nonnegative_act_mean = evil_target_nonnegative_act_mean
        logging.info("Behavior mean for evil targets' non-negative acts set to %s." % evil_target_nonnegative_act_mean)

        self.behavior_stddev = behavior_stddev
        logging.info("Standard deviation of behaviors set to %s." % behavior_stddev)

        if initial_impressions is None:
            logging.warn("Initial impressions were not provided! Generating matrix with zeroes...")
            initial_impressions = np.zeros((n_observers,
                                            n_targets))
        else:
            nrows, ncols = initial_impressions.shape
            if nrows != n_observers:
                raise MaliciousGossipModelException("The number of rows in the initial impressions matrix (%s) "
                                                    "does not equal the number of observers (%s)"
                                                    % (nrows, n_observers))
            if ncols != n_targets:
                raise MaliciousGossipModelException("The number of rows in the initial impressions matrix (%s) "
                                                    "does not equal the number of observers (%s)" % (ncols, n_targets))
        self.observer_target_impressions = initial_impressions

        if integration_rule is None:
            logging.info("Integration rule function is not provided! "
                         "Using equally weighted average integration rule...")
            integration_rule = helpers.average_integration_rule
        self.validate_integration_rule(integration_rule)
        self.integration_rule = integration_rule

        if interaction_likelihood_rule is None:
            logging.info("Interaction likelihood rule is not provided! Using Luce choice function...")
            interaction_likelihood_rule = helpers.luce
        self.validate_interaction_likelihood_rule(interaction_likelihood_rule)
        self.interaction_likelihood_rule = interaction_likelihood_rule

        if directed_gossip:
            logging.info("Directed gossip is used!")
        else:
            logging.info("Interesting gossip is used!")
        self.directed = directed_gossip

        self.disregard_threshold = disregard_threshold
        if disregard_threshold is not None:
            logging.info("Disregard threshold set to %s." % disregard_threshold)

        logging.info("Model object has been successfully initialized!")

    ############################### Public methods ###############################
    def run(self):
        logging.info("Running model step for time %s is in progress..." % self.tick)
        new_impressions = self.observer_target_impressions.copy()
        interaction_decisions = self.obtain_interaction_decisions()
        targets_for_observers = self.find_targets_for_observers()
        target_behaviors = self.generate_target_behaviors()
        for observer_index in range(0, self.n_observers):
            target_index = targets_for_observers[observer_index, ]
            logging.debug("Target for observer %s is %s." % (observer_index, target_index))
            interaction_decision = interaction_decisions[observer_index, target_index]
            if interaction_decision == 1:
                target_current_behavior = target_behaviors[target_index, ]
                new_impressions[observer_index, target_index] = target_current_behavior
                logging.debug("Observer %s chose to OBSERVE target %s with behavior %s."
                              % (observer_index, target_index, target_current_behavior))
            else:
                observer_indices = np.arange(self.n_observers)
                # Ensure that observer doesn't gossip with itself
                observer_indices = observer_indices[observer_indices != observer_index]
                observer2_index = np.random.choice(observer_indices, 1)[0]

                if not self.directed:
                    observer2_impressions = self.observer_target_impressions[observer2_index,]
                    target_index = np.argmin(observer2_impressions)
                    logging.debug("Observer %s holds the most negative impression about target %s, therefore "
                                  "gossip is gonna take place about that target" % (observer2_index,
                                                                                    target_index))

                observer2_target_impression = self.observer_target_impressions[observer2_index, target_index]
                logging.debug("Observer %s chose to GOSSIP about target %s with observer %s (second observer's "
                              "impression about target: %s)"
                              % (observer_index, target_index, observer2_index, observer2_target_impression))
                if self.disregard_threshold is not None:
                    observer_target_impression = self.observer_target_impressions[observer_index, target_index]
                    if abs(observer2_target_impression - observer_target_impression) >= self.disregard_threshold:
                        logging.debug("Observer %s chose to DISREGARD the observer %s's impression about target %s!"
                                      % (observer_index, observer2_index, target_index))
                        new_impressions[observer_index, target_index] = observer_target_impression
                    else:
                        new_impressions[observer_index, target_index] = observer2_target_impression
                else:
                    new_impressions[observer_index, target_index] = observer2_target_impression

        logging.info("Successfully obtained new impressions! Updating impression matrix...")
        old_impressions = self.observer_target_impressions.copy()
        updated_impressions = self.integration_rule(old_impressions,
                                                    new_impressions)
        self.observer_target_impressions = updated_impressions
        logging.info("Running model step for time %s was successful!" % self.tick)

        # Increment counter
        self.tick = self.tick + 1

    def generate_target_behaviors(self):
        logging.info("Generating behaviors for targets is in progress...")
        # Obtain initial behavior means
        logging.info("Obtaining behaviors for normal targets...")
        target_behaviors = np.random.normal(self.normal_target_behavior_mean,
                                            self.behavior_stddev,
                                            self.n_targets)

        # Obtain evil means
        num_evil_targets = self.evil_target_indices.shape[0]
        logging.info("Generating means for %s evil targets..." % num_evil_targets)
        evil_target_behavior_probs = np.random.uniform(low=0.0,
                                                       high=1.0,
                                                       size=num_evil_targets)
        evil_target_behaviors = evil_target_behavior_probs.copy()
        evil_target_negative_acts = evil_target_behavior_probs <= self.evil_target_negative_act_prob
        num_evil_acts = evil_target_negative_acts[evil_target_negative_acts == True].size
        logging.info("Number of evil acts: %s" % num_evil_acts)
        evil_target_behaviors[evil_target_negative_acts == True] = np.random.normal(self.evil_target_negative_act_mean,
                                                                                    self.behavior_stddev,
                                                                                    num_evil_acts)
        evil_target_behaviors[evil_target_negative_acts == False] = np.random.normal(
            self.evil_target_nonnegative_act_mean,
            self.behavior_stddev,
            num_evil_targets - num_evil_acts)
        # Update behavior means
        target_behaviors[self.evil_target_indices] = evil_target_behaviors

        logging.info("Successfully generated target behaviors!")
        return target_behaviors

    def obtain_interaction_decisions(self):
        logging.info("Finding interaction probabilities...")
        interaction_probabilities = np.apply_along_axis(self.interaction_likelihood_rule,
                                                        axis=1,
                                                        arr=self.observer_target_impressions)
        interaction_decisions = interaction_probabilities - np.random.uniform(size=(self.n_observers,
                                                                                    self.n_targets))
        interaction_decisions = interaction_decisions >= 0

        logging.info("Interaction decisions were successfully obtained!")
        return interaction_decisions

    def find_targets_for_observers(self):
        logging.info("Finding targets for observers is in progress...")
        target_indices = np.random.choice(self.n_targets,
                                          self.n_observers)
        logging.info("Targets for observers were successfully found!")
        return target_indices

    def validate_integration_rule(self,
                                  integration_rule_func,
                                  num_impressions=1000,
                                  mu = 0,
                                  sigma = 1):
        logging.debug("Validating integration rule is in progress...")
        random_old_impressions = np.random.normal(mu, sigma, num_impressions)
        random_new_impressions = np.random.normal(mu, sigma, num_impressions)
        result_vector = self.__test_func(func_to_validate=integration_rule_func,
                                         args=(random_old_impressions,
                                               random_new_impressions),
                                         required_size=num_impressions,
                                         func_name="integration rule")
        logging.info("Integration rule function is valid!")

    def validate_interaction_likelihood_rule(self,
                                             interaction_likelihood_rule,
                                             num_impressions=1000,
                                             mu=0,
                                             sigma=1):
        logging.debug("Validating interaction likelihood rule is in progress...")
        sample_array = np.random.normal(mu, sigma, num_impressions)
        result_vector = self.__test_func(func_to_validate=interaction_likelihood_rule,
                                         args=(sample_array, ),
                                         required_size=num_impressions,
                                         func_name="interaction likelihood rule",
                                         is_vector=False)
        logging.info("Interaction likelihood rule function is valid!")

    ############################### Private methods ###############################
    def __test_func(self, func_to_validate, args, required_size, func_name, is_vector=True):

        # Validate that it is in fact a function
        if not callable(func_to_validate):
            raise MaliciousGossipModelException("The provided function (%s) is not a function!" % func_name)

        try:
            result_vector = func_to_validate(*args)
            if result_vector.shape[0] != required_size:
                raise MaliciousGossipModelException(
                    "%s function is not valid! Make sure it's a function that "
                    "takes two numpy arrays as arguments and returns "
                    "a numpy array of the same size!" % func_name.capitalize()
                )
            elif result_vector.ndim != 1 and is_vector:
                raise MaliciousGossipModelException(
                    "%s function is not valid! Make sure it's a function that "
                    "takes two numpy arrays as arguments and returns "
                    "a numpy array of the same size!" % func_name.capitalize()
                )
            else:
                logging.info("Integration rule function is valid!")
                return result_vector

        except Exception, e:
            logging.error("Encountered the following error while validating function named %s: %s" % (func_name, e))
            raise MaliciousGossipModelException(
                "%s function is not valid! Make sure it's a function that "
                "takes two column vectors (numpy arrays) as arguments and returns "
                "a column vector (numpy array) of the same size!" % func_name.capitalize())
