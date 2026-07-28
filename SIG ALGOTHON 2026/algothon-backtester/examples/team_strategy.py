import numpy as np


currentPos = None


def getMyPosition(prcSoFar):
    global currentPos

    n_instruments, n_days = prcSoFar.shape

    if currentPos is None or currentPos.shape != (n_instruments,):
        currentPos = np.zeros(n_instruments, dtype=int)

    if n_days < 2:
        return np.zeros(n_instruments, dtype=int)

    last_ret = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    norm = np.sqrt(last_ret.dot(last_ret))

    if norm < 1e-12:
        return currentPos.astype(int)

    scaled_returns = last_ret / norm
    position_delta = (5000 * scaled_returns / prcSoFar[:, -1]).astype(int)
    currentPos = (currentPos + position_delta).astype(int)

    return currentPos
