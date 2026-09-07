import numpy as np




def is_unitary(U):
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]))