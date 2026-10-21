import numpy as np


from . import config
from . import lib



class Gate_R(lib.Gate):

    """A sigle qudit gate that rapresent the product of a two level phase gate Pz gate and a two-level rotation RX gate,  
    there are three possible R gates, depending of the levels at which they act. """

    def __init__(self, q: int, theta: float, index:int):
            if config.DIM != 3:
                 raise ValueError('This gate is defined only in dimension 3')
            super().__init__()   
            self.indexes = ((index+1)%3 , (index-1)%3)
            self.name = 'R^{'f'{self.indexes[0]}{self.indexes[1]}''}'
            self.unitary=True
            self.clifford = False
            self.target_qudits = (q,)
            self.parameters = (theta, index)
            self.is_controlled = False
            self.Matrix()



    def Matrix(self):
        self.matrix[self.parameters[1]][self.parameters[1]]=np.exp(-self.parameters[0]*1j)
        self.matrix[self.indexes[1]][self.indexes[1]]=np.cos(self.parameters[0])
        self.matrix[self.indexes[0]][self.indexes[0]]=np.cos(self.parameters[0])
        self.matrix[self.indexes[1]][self.indexes[0]]=-np.sin(self.parameters[0])*1j
        self.matrix[self.indexes[0]][self.indexes[1]]=-np.sin(self.parameters[0])*1j
        return self.matrix






class R_tot(lib.Gate):

    """a controlled gate that is the combination of the three R gates. It applies R0 if the control qudit is |0>, R1 if it's in |1> and R2 if  it's  in |2>"""
    
    def __init__(self, q: int, p:int, theta:float):
        if config.DIM != 3:
                raise ValueError('This gate is defined only in dimension 3')
        super().__init__()   
        self.name = 'R'
        self.unitary=True
        self.clifford = False
        self.control_qudits = (p,)
        self.target_qudits = (q,)
        self.parameters = (theta,)
        self.is_controlled = True 
        self.base_matrices =[]


    def Matrix(self):
        import sympy as sp
        self.get_base_matrices()
        a, b, c = sp.symbols('a b c')
        self.matrix = a*self.base_matrices[0]+b*self.base_matrices[1]+c*self.base_matrices[2]
        return self.matrix



    def get_base_matrices(self):
            for i in range(self.dim):
                self.base_matrices.append(Gate_R(self.target_qudits[0], self.parameters[0], i).matrix)
            return self.base_matrices



