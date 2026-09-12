import numpy as np




def is_unitary(U):
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]))



'''
    def __mul__(self, other):
        return State(self.state * other)

    def __rmul__(self, other):
        return State(other * self.state)

    def __matmul__(self, other):
        other = other.state if isinstance(other, State) else other
        return self.state @ other

    def __rmatmul__(self, other):
        return other @ self.state'''