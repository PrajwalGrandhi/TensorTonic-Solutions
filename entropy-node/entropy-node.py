import numpy as np

def entropy_node(y):
    y = np.array(y)
    unique_values, counts = np.unique(y, return_counts=True)
    non_zero_index = counts != 0
    counts = counts[non_zero_index]
    probs = counts / np.sum(counts)

    return -np.sum(probs*np.log2(probs))