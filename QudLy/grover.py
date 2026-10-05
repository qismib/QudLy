import numpy as np


from . import ausiliary as Au
from . import config
from . import lib




class  Gate_Oracle(lib.Gate):

    """  """

    def __init__(self, q: int, target: int):
        super().__init__()
        self.name = 'Oracle'
        self.unitary = True
        self.clifford = False 
        self.target_qudits = (q,)
        self.parameters = (target, )
        self.Matrix()
        


    def Matrix(self):
        self.matrix=np.identity(self.dim)
        self.matrix[self.parameters[0]][self.parameters[0]]=-1
        return self.matrix



class Gate_Reflection(lib.Gate):
    """  """

    def __init__(self, q: int):
        super().__init__()
        self.name = 'Reflect'
        self.unitary = True
        self.clifford = False 
        self.target_qudits = (q,)
        self.Matrix()
        


    def Matrix(self):
        self.matrix=np.ones((self.dim, self.dim))*(2/self.dim)
        self.matrix=(self.matrix-np.identity(self.dim))*np.exp(1j*np.pi/self.dim)
        return self.matrix    