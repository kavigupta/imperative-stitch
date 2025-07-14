import numpy as np


def sample_hyperparameters(seed):
    rng = np.random.RandomState(seed)
    return {
        "max_arity": rng.choice(5) + 1,
        "minimum_number_matches": rng.geometric(0.1) + 2,
        "application_utility_metavar": -1 * rng.exponential(),
        "application_utility_symvar": -0.5 * rng.exponential(),
        "application_utility_fixed": -3 * rng.exponential(),
    }
