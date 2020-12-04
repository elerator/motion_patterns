import inspect
from collections import defaultdict
import numpy as np

def print_module_methods(module):
    """ Prints signature and formal parameters for each method in the module.
    Args:
        module: A module that contains methods.
    """
    for fun in [x[1] for x in inspect.getmembers(module)][:]:
        try:
            print(fun.__name__+str(inspect.signature(fun)), end="\n\n")
        except:
            pass

class NestedDict(defaultdict):
    def __init__(self):
        super().__init__(self.__class__)
    def __reduce__(self):
        return (type(self), (), None, None, iter(self.items()))


def slow_wave_features(dataset, features, where = None):
    """ Adapter method for dataset. Retrieve slow-wave features for all slow waves.
    Args:
         dataset: Dataset of type NedtedDict
         features: List of features (keys of dataset["sws"][sws_id])
         where: Boolean array that is used to select a subset of each vector of retrieved features. 
                Applicable if vectors are retrived.
    Returns:
         List of data for each feature.
    """
    slow_wave_ids = list(dataset["sws"].keys())

    features_out = []
    for name in features:
        feature = np.array([dataset["sws"][k][name] for k in slow_wave_ids])
        if type(feature[0]) == type(NestedDict()):
            raise KeyError(name)
        if type(where) != type(None):
            feature = feature[where]
        features_out.append(feature)
    return features_out

def set_random_state(seed_value = 42):
    """ Makes keras results reproducible. See stackoverflow.com/questions/50659482.
    """
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
