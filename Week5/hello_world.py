import math
import random
from pprint import pprint


def batches(batch_size, features=[], labels=[]):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching

    idx_batch = random.sample([x for x in range(len(features))], batch_size)
    idx_left = [x for x in range(len(features)) if x not in idx_batch]
    batch_features = [features[x] for x in idx_batch]
    batch_labels = [labels[x] for x in idx_batch]
    batch_left_features = [features[x] for x in idx_left]
    batch_left_labels = [labels[x] for x in idx_left]
    return [[batch_features,batch_labels],[batch_left_features,batch_left_labels]]


# 4 Samples of features
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]

#print(random.sample([x for x in range( len(example_features))],2))

# PPrint prints data structures like 2d arrays, so they are easier to read
pprint(batches(3, example_features, example_labels))
