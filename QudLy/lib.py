import numpy as np
from abc import ABC, abstractmethod
import  copy

from . import ausiliary as Au
from . import config
#import ausiliary as Au
#import config


class Gate:

    def __init__(self):
        self.dim = config.DIM
        self.name = None
        self.is_controlled_by = None 
        self.unitary = False
        self.clifford = False
        self.matrix = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.target_qudits = ()
        self.control_qudits = ()
        self.parameters = ()
        self.Matrix()


    def dagger(self):
        new_gate=copy.copy(self)
        new_gate.matrix = self.matrix.conj().T
        new_gate.is_controlled_by = self.is_controlled_by
        new_gate.control_qudits = self.control_qudits
        return new_gate

    
    def is_unitary(self):
        uni=np.allclose(self.matrix.conj().T @ self.matrix, np.eye(self.matrix.shape[0]))
        self.unitary=uni
        return uni


    @abstractmethod
    def Matrix(self):
        raise(
            NotImplementedError
        )


    def __str__(self):
        self.draw=np.array2string(self.matrix, precision=2,separator='  ', formatter={'complex_kind': lambda z:  
                f"{z.real:g}" if np.isclose(z.imag, 0) else (f"{z.imag:g}j" if np.isclose(z.real, 0) else f"{z:g}")
        } )
        return self.draw




class  Gate_X(Gate):

    def __init__(self, q: int):
        super().__init__()
        self.name = 'X'
        self.unitary = True
        self.clifford = True
        self.target_qudits = (q,)



    def Matrix(self):
        self.matrix[0][self.dim-1]= 1
        i=1
        while i<self.dim:
            self.matrix[i][i-1]= 1
            i+=1
        return self.matrix




class Gate_Z(Gate):
    def __init__(self, q: int):
            super().__init__()
            self.name = 'Z'
            self.unitary=True
            self.clifford = True
            self.target_qudits = (q,)



    def Matrix(self):
        omega = np.exp(1j*2*np.pi/self.dim)
        i=0
        while i<self.dim:
            self.matrix[i][i]=omega**i
            i+=1
        return self.matrix




class Gate_H(Gate):
    def __init__(self, q: int):
                super().__init__()
                self.name = 'H'
                self.unitary=True
                self.clifford = True
                self.target_qudits = (q,)

    
    
    def Matrix(self):
        omega = np.exp(1j*2*np.pi/self.dim)
        i=0 #righe   
        while i<self.dim:
            k=0 #colonne 
            while k<self.dim:
                self.matrix[i][k]=omega**((self.dim-i)*k)
                k+=1
            i+=1
        self.matrix=self.matrix/np.sqrt(self.dim)
        return self.matrix

    


class Gate_P(Gate):
     
    def __init__(self, q: int, theta: float):
            super().__init__()
            self.name = 'P'
            self.unitary=False
            self.clifford = False
            self.target_qudits = (q,)
            self.parameters=(theta)

    def Matrix(self):

        pass





