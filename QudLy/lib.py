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
        self.target_qudits = ()
        self.control_qudits = ()
        self.parameters = ()
        self.matrix = np.zeros((self.dim, self.dim), dtype=np.complex128)

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
        self.draw=np.array2string(self.matrix, precision=2, separator='  ', formatter={'complex_kind': lambda z:  
                f"{z.real:g}" if np.isclose(z.imag, 0) else (f"{z.imag:g}j" if np.isclose(z.real, 0) else f"{z:.3f}")
        } )
        return self.draw




class  Gate_X(Gate):

    def __init__(self, q: int):
        super().__init__()
        self.name = 'X'
        self.unitary = True
        self.clifford = True
        self.target_qudits = (q,)
        self.Matrix()



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
            self.Matrix()



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
                self.Matrix()

    
    
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
            self.unitary=True
            self.clifford = False
            self.target_qudits = (q,)
            self.parameters=(theta)
            self.Matrix()



    def Matrix(self):
        omega = np.exp(1j*2*np.pi/self.dim)
        i=0
        while i<self.dim:
            self.matrix[i][i]=omega**(i*(self.parameters/np.pi))
            i+=1
        return self.matrix




class Gate_SUMX(Gate):
    def __init__(self, q: int, p: int):
        super().__init__()
        self.name = 'SUMX'
        self.unitary=True
        self.clifford = True
        self.target_qudits = (q,)
        self.control_qudits =(p,)
        self.matrix = np.zeros((self.dim**2, self.dim**2), dtype=np.complex128)
        self.Matrix()

    def Matrix(self):
        X=Gate_X(self.target_qudits)
        i=0
        while i<self.dim:
            self.matrix[self.dim+((i-1)*self.dim):self.dim+i*self.dim, self.dim+((i-1)*self.dim):self.dim+i*self.dim] = np.linalg.matrix_power(X.matrix, i)
            i+=1
        return self.matrix




class Gate_SUMP(Gate):
    def __init__(self, q: int, p: int, theta: float):
        super().__init__()
        if theta==np.pi:
            self.name='CZ'
        else:
            self.name = 'SUMP'
        self.unitary=True
        self.clifford = False
        self.target_qudits = (q,)
        self.control_qudits =(p,)
        self.parameters=(theta)
        self.matrix = np.zeros((self.dim**2, self.dim**2), dtype=np.complex128)
        self.Matrix()


    def Matrix(self):
        P=Gate_P(self.target_qudits, self.parameters)
        i=0
        while i<self.dim:
            self.matrix[self.dim+((i-1)*self.dim):self.dim+i*self.dim, self.dim+((i-1)*self.dim):self.dim+i*self.dim] = np.linalg.matrix_power(P.matrix, i)
            i+=1
        return self.matrix




class Gate_CZ(Gate_SUMP):
    def __init__(self, q: int, p: int):
        super().__init__(q, p, np.pi)
        self.name = 'CZ'


         


def apply_CX(q: int, p: int):
    Z=Gate_CZ(q, p)
    apply_QFT(q)
    apply_gate(Z)
    apply_QFT(q)



def apply_QFT(q: int):
    num=len(q)
    theta=float(0)
    i=num-1 
    while i>=0:
        k=i-1
        H=Gate_H.dagger(q[i])
        apply_gate(H)
        while k>=0:
            theta=np.pi*2**(config.DIM*(k-i))
            apply_gate(Gate_SUMP(q[i], q[k], theta))
            k=k-1
        i=i-1



#------------------------------WORK IN PROGRESS-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def apply_gate(State1, Gate):   #lista di stati 
    
    #return state
    pass


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



class State():

    states=[]

    def __init__(self, vett: complex, N:int):     #N da sistemare 

        self.dim=config.DIM

        if vett is None:
            self.state=np.asarray(np.zeros(self.dim), dtype=np.complex128)
            self.state[0]=1
        else:
            self.state=np.asarray(vett, dtype=np.complex128)
            if self.state.shape!=(config.DIM,):
                raise ValueError(f'State must have dimension {self.dim}')
            if np.allclose(self.state, 0):
                raise ValueError('State does not exists')
        
        if N is None:                                                      #da togliere con Circuit 
            raise ValueError('State must have a indentification number')
        self.num = N 

        norm = np.linalg.norm(self.state)
        if not np.isclose(norm, 0):
            self.state /= norm
        
    def __array__(self, dtype=np.complex128):
        return np.asarray(self.state, dtype=dtype)

    def __getitem__(self, index):
        return self.state[index]

    def __str__(self):
            return str(self.state)


    def decompose(self):
        base = np.eye(self.dim)
        terms = []

        for i, amplitude in enumerate(self.state):
            if np.isclose(amplitude, 0):
                continue
            basis = base[i]
            if np.allclose(basis, 0):
                continue
            terms.append(f"{amplitude} * |{i}>")
        return " + ".join(terms)




















