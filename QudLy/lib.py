import numpy as np
from abc import ABC, abstractmethod
import  copy
import math


from . import ausiliary as Au
from . import config


class Gate:

    def __init__(self):
        self.dim = config.DIM
        self.name = None
        self.is_controlled_by = None 
        self.unitary = False
        self.clifford = False
        self.is_controlled =False
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
        i=0    
        while i<self.dim:
            k=0 
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
        self.is_controlled = True 
        self.target_qudits = (q,)
        self.control_qudits =(p,)
        self.base_matrix = Gate_X(self.target_qudits[0]).matrix
        self.matrix = np.zeros((self.dim**2, self.dim**2), dtype=np.complex128)
        self.Matrix()


    def Matrix(self):                 
        i=0
        while i<self.dim:
            self.matrix[self.dim+((i-1)*self.dim):self.dim+i*self.dim, self.dim+((i-1)*self.dim):self.dim+i*self.dim] = np.linalg.matrix_power(self.base_matrix, i)    #np.block?
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
        self.is_controlled= True
        self.target_qudits = (q,)
        self.control_qudits =(p,)
        self.parameters=(theta,)
        self.base_matrix=Gate_P(self.target_qudits[0], self.parameters[0]).matrix
        self.matrix = np.zeros((self.dim**2, self.dim**2), dtype=np.complex128)
        self.Matrix()


    def Matrix(self):
        i=0
        while i<self.dim:
            self.matrix[self.dim+((i-1)*self.dim):self.dim+i*self.dim, self.dim+((i-1)*self.dim):self.dim+i*self.dim] = np.linalg.matrix_power(self.base_matrix, i)
            i+=1
        return self.matrix




class Gate_CZ(Gate_SUMP):
    def __init__(self, q: int, p: int):
        super().__init__(q, p, np.pi)
        self.name = 'CZ'


         
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------




def apply_CX(state, q: int, p: int):      
    Z=Gate_CZ(q, p)
    ris=state
    ris=apply_QFT(ris, q, q)
    ris=apply_gate(ris, Z)
    ris=apply_QFT(ris, q, q)
    return ris



def apply_QFT(state, ini:int=0, fi:int=None):        
    num=int(math.log(len(state.state), config.DIM))
    if fi==None:
        fi=num-1
    theta=float(0)
    i=fi
    ris=state
    while i>=ini:
        k=i-1
        H=Gate_H(i)
        H=H.dagger()
        ris=apply_gate(ris, H)
        while k>=ini:
            theta=np.pi*2**(config.DIM*(k-i))
            P=Gate_SUMP(i, k, theta)
            ris=apply_gate(ris, P)
            k=k-1
        i=i-1
    return ris



def apply_SWAP(state, q: int, p: int):
    res=state
    res=apply_CX(res, q, p)
    res=apply_CX(res, p, q)
    res=apply_CX(res, q, p)
    return res






def apply_gate(state, gate): 
    dim=config.DIM
    num=int(math.log(len(state.state), dim))
    st = state.state.reshape([dim] * num).copy()
    if gate.is_controlled:                       #CASINO
        control=gate.control_qudits[0]
        target=gate.target_qudits[0]
        if control<gate.target_qudits[0]:
            target-=1
        for i in range(dim):
            index=[slice(None)]*num
            index[control]=i
            Res=np.tensordot(np.linalg.matrix_power(gate.base_matrix, i), st[tuple(index)], axes=([1], [target]))
            Res=np.moveaxis(Res, 0, target)
            st[tuple(index)]=Res
        res=Total_state(st.reshape(-1))
    else:
        st=np.tensordot(gate.matrix, st, axes=([1], [gate.target_qudits[0]]))
        st=np.moveaxis(st, 0, gate.target_qudits[0])
        res=Total_state(st.reshape(-1))
    return res 

















#----------------------------------------------------------------------------------------------------------------------------------------------------------------



class abs_State():

    def __init__(self):
        self.dim = None 
        self.state = None 

    def __array__(self, dtype=np.complex128):
        return np.asarray(self.state, dtype=dtype)
    
    def __getitem__(self, index):
        return self.state[index]

    def __str__(self):
        self.draw=np.array2string(self.state, precision=2, separator='  ', formatter={'complex_kind': lambda z: f"{z.real:g}" if np.isclose(z.imag, 0) else (f"{z.imag:g}j" if np.isclose(z.real, 0) else f"{z:.3f}")} )
        return self.draw

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




class State(abs_State):

    def __init__(self, vett: complex, N:int):   

        super().__init__()

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
        
        if N is None:                                                      
            raise ValueError('State must have a indentification number')
        
        self.num = N 

        self.normalize()


    def normalize(self):
        if not np.isclose(np.linalg.norm(self.state), 0):
            self.state /= np.linalg.norm(self.state)
    
        
   

class Total_state(abs_State):            #total state    example: |000>   

    def __init__(self, vett):   

        super().__init__()

        self.dim=len(vett)

        if vett is None:
            raise ValueError ('Insert your state')

        self.state=vett
        


def create_state(lista):                     #Build the total state.  States must be given in order  

    S=lista[0].state

    for i in range(len(lista)-1):
        S=np.kron(S, lista[i+1].state)
    
    new_state=Total_state(S)

    return new_state




#----------------------------------------------------------------------------------------------------------------------------------------------------------------




def measure(state):
    prob=np.abs(state)**2
    m=np.random.choice(len(state.state), p=prob)
    st=indexes(m, len(state.state))
    return st



def measure_prob(state):
    prob=np.abs(state)**2
    result=[]
    for l in range(len(state.state)):
        if prob[l]!=0:
            result.append([indexes(l, len(state.state)), round(float(prob[l]), 4)])
    return result



def measure_single(state, q: int, collapse:bool = False):
    dim=config.DIM
    num=int(math.log(len(state.state), dim))
    st = state.state.reshape([dim] * num)
    ax=tuple(i for i in range(num) if i != q)         
    prob = np.sum(np.abs(st)**2, axis=ax)
    m=np.random.choice(dim, p=prob)
    if collapse:                                    
        new_vett=np.zeros_like(st)
        slices= [slice(None)]*num
        slices[q]=m
        new_vett[tuple(slices)]=st[tuple(slices)]
        norm = np.linalg.norm(new_vett)
        new_vett /= norm
        new_state=Total_state(new_vett.reshape(-1))
        return m, new_state
    else:
        return m 



def measure_prob_single(state, q:int ):
    dim=config.DIM
    num=int(math.log(len(state.state), dim))
    st = state.state.reshape([dim] * num)
    ax=tuple(i for i in range(num) if i != q)
    prob = np.sum(np.abs(st)**2, axis=ax)
    result=[]
    for i in range(len(prob)):
        if prob[i]!=0:
            result.append([i, round(float(prob[i]), 4)])
    return result



def indexes(pos:int, l:int):    # position in the array and lenght of the array
    dim=config.DIM
    num=int(math.log(l, dim))
    digits=[]
    i=num-1
    while i>=0:
        digits.append(pos//dim**i)
        pos=pos%(dim**i)
        i=i-1
    return digits 



