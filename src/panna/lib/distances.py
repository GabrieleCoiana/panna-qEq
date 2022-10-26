import numpy as np


def cos_distance(v_1, v_2):
    """ compute cosine distances between 2 vectors

    Parameters
    ----------
    v_1: np array space_size
         first vector
    v_2: np array ?, space_size
         second vector

    Return
    ------
    distances, np array ?
    """
    distances = np.dot(v_1[np.newaxis, :], v_2.T) / (
        np.linalg.norm(v_1) * np.linalg.norm(v_2, axis=-1))
    # fix numerical errors
    distances[distances > 1] = 1
    return distances
