"""
yjia67
"""
import numpy as np

def prob_1_1(N):
    """
    Args: N: the number of trials.
    Returns: arr: array of rolls.
    """

    ### START CODE HERE ###
    arr = np.round(np.random.rand(6)*6, 0)
    ### END CODE HERE ###

    return arr

def prob_1_2(y):
    """
    Args: y: numpy array. 
    Returns: z: numpy array of shape (new_size,2).
    """

    ### START CODE HERE ###
    z = y.reshape((3,2))
    ### END CODE HERE ###

    return z

def prob_1_3(z):
    """
    Args: z: numpy array of shape (3,2).
    Returns: x: max value in z.
    r: row index of x.
    c: column index of x.
    """

    ### START CODE HERE ###
    x = np.max(z)
    ind = np.where(z == x)
    r, c = ind[0], ind[1]
    ### END CODE HERE ###

    return (x, r, c)


def prob_1_4(v):
    """
    Args: v: numpy array. 
    Returns: x: number of 1’s in v.
    """

    ### START CODE HERE ###
    x = np.sum(v == 1)
    ### END CODE HERE ###

    return x




if __name__ == '__main__':

    # # 1.1
    # N = 5
    # arr = prob_1_1(N)
    # print(arr)

    # # 1.2
    # y = np.array([11, 22, 33, 44, 55, 66])
    # z = prob_1_2(y)
    # print(z)

    # # 1.3
    # x, r, c = prob_1_3(z)
    # print('x: ' + str(x))
    # print('r: ' + str(r))
    # print('c: ' + str(c))

    # # 1.4
    # v = np.array([1, 4, 7, 1, 2, 6, 8, 1, 9])
    # x = prob_1_4(v)
    # print('x: ' + str(x))