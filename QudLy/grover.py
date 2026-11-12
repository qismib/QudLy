import numpy as np


from . import lib




class  Gate_Oracle(lib.Gate):

    """A gate that, given a state's index, marks that state. It acts as O|j> = |j> if j  it's not the marked state, 
      O|j> = |-j> if j is the marked state"""

    def __init__(self, q: int, target):
        super().__init__()
        self.name = 'Oracle'
        self.unitary = True
        self.clifford = False 
        self.target_qudits = (q,)
        self.parameters = (target, )
        self.Matrix()
        


    def Matrix(self):
        for i in range(self.dim):
            self.matrix[i][i]=(-1)**(self.parameters[0](i))
 


class Gate_Reflection(lib.Gate):

    """This gate it's used for amplify the population ofthe marked state """

    def __init__(self, q: int):
        super().__init__()
        self.name = 'Reflect'
        self.unitary = True
        self.clifford = False 
        self.target_qudits = (q,)
        self.Matrix()
        


    def Matrix(self):
        self.matrix=np.ones((self.dim, self.dim))*(2/self.dim)
        self.matrix=(self.matrix-np.identity(self.dim))
        if self.dim%2==0:
            self.matrix=self.matrix*np.exp(1j*np.pi/self.dim)
   