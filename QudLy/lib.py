import numpy as np
from abc import ABC, abstractmethod
import  copy
import math


from . import ausiliary as Au
from . import config



class Gate(ABC):

    """ The abstract class inherited by all gates, every gate has a matrix associated with
    when a gate is created you must specify the qudit on which the gate will act.
    if a gate is controlled, you also must give the qudit that controls the gate.
    """

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


    """These are some operations that could be useful for gate manipulation"""

    def dagger(self):
        new_gate=copy.copy(self)
        new_gate.matrix = self.matrix.conj().T
        new_gate.is_controlled_by = self.is_controlled_by
        new_gate.control_qudits = self.control_qudits
        return new_gate


    @abstractmethod
    def Matrix(self):
        raise(
            NotImplementedError
        )

    """ A method that controls the printing of the gate matrix"""

    def __str__(self):
        self.draw=np.array2string(self.matrix, precision=2, separator='  ', formatter={'complex_kind': lambda z:  
                f"{z.real:g}" if np.isclose(z.imag, 0) else (f"{z.imag:g}j" if np.isclose(z.real, 0) else f"{z:.3f}")
        } )
        return self.draw





"""These are the most important single-qudit gates, when a gate is created, its matrix is generated """


class  Gate_X(Gate):

    """A gate that acts as X|j> = |(j+1) mod dim>    """

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

    """A gate that acts as Z|j> = ω^j|j>   where ω = exp(2πi/dim) """

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

    """A gate that acts as H|j> = 1/sqrt(d)sum(ω**(j*k)|k>)  where ω = exp(2πi/dim) and sum is between 0 and dim-1 in k"""

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
                self.matrix[i][k]=omega**(i*k)
                k+=1
            i+=1
        self.matrix=self.matrix/np.sqrt(self.dim)
        return self.matrix

    


class Gate_P(Gate):

    """A gate that represent the generalized rotation around the Z-axis, the gate requires a float parameter which is the angle.
    It acts as P(θ)|j> = ω^(j*θ)/π|j>   where ω = exp(2πi/dim) """

    def __init__(self, q: int, theta: float):
            super().__init__()
            self.name = 'P'
            self.unitary=True
            self.clifford = False
            self.target_qudits = (q,)
            self.parameters=(theta, )
            self.Matrix()



    def Matrix(self):
        omega = np.exp(1j*2*np.pi/self.dim)
        i=0
        while i<self.dim:
            self.matrix[i][i]=omega**(i*(self.parameters[0]/np.pi))
            i+=1
        return self.matrix



"""Some controlled gates. They are made up of a base_matrix, which set the gate that is applied n times depending on the state of the control qudit """


class Gate_SUMX(Gate):

    """the controlled version of the gate X, it acts as SUMX|j>|k> = |j>|(j+k>) mod dim>        """

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

    """the controlled version of the gate P, it acts as SUMP(θ)|j>|k> = ω^(k*j*θ)/π|j>|k>   where ω = exp(2πi/dim)         """

    def __init__(self, q: int, p: int, theta: float):
        super().__init__()
        if np.isclose(theta, np.pi):
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

    """A specific version of SUMP with θ=π """

    def __init__(self, q: int, p: int):
        super().__init__(q, p, np.pi)
        self.name = 'CZ'


         
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------




def apply_CX(state, p: int, q: int):   

    """A function that receive a state and two int that rapresents qudits. CX rapresent a operator that act  as 
    CX|x>|y> = |x>|-x-y>   """

    Z=Gate_CZ(q, p)
    H=Gate_H(q)
    ris=state
    ris=apply_gate(ris, H)
    ris=apply_gate(ris, Z)
    ris=apply_gate(ris, H)
    return ris



def apply_QFT(state, ini:int=0, fi:int=None):        

    """ 
    Applies the Quantum Fourier Transform to a range of qudits.
    The QFT is the quantum analogue of the discrete Fourier transform.
    For a system of n qudits with dimension dim, it transforms a computational
    basis state |x> into  QFT|x> = (1 / sqrt(dim^n)) * sum_{y=0}^{dim^n-1}ω^(x*y) |y>,
    where ω = exp(2*pi*i/dim^n) is the primitive dim^n-th root of unity. """

    num=round(math.log(len(state.state), config.DIM))
    if fi==None:
        fi=num-1
    theta=float(0)
    i=ini
    ris=state
    while i<=fi:
        H=Gate_H(i)
        ris=apply_gate(ris, H)
        k=i+1
        while k<=fi:
            theta=np.pi*config.DIM**(i-k)
            P=Gate_SUMP(i, k, theta)
            ris=apply_gate(ris, P)
            k=k+1
        i=i+1

    In=ini
    Fin=fi 
    while In<=Fin:
        ris=apply_SWAP(ris, In, Fin)
        In=In+1
        Fin=Fin-1
    return ris  




def apply_SWAP(state, q: int, p: int):

    """A function that receives a state and two ints representing qudits. SWAP represents an operator that acts as
    SWAP|j>|k> = |k>|j>."""

    res=state
    res=apply_CX(res, q, p)
    res=apply_CX(res, p, q)
    res=apply_CX(res, q, p)
    return res




def apply_gate(state, gate): 

    """
    The function applies a gate to a total quantum state. It uses the tensor representation of the total state so that the gate acts only on the corresponding qudit.
    For now, the function only supports gates controlled by a single qudit.
    The inputs are a Total_state and a gate."""

    dim=config.DIM
    num=round(math.log(len(state.state), dim))
    st = state.state.reshape([dim] * num).copy()
    if gate.is_controlled:                      
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



class abs_State(ABC):
 
    """The abstract class inherited by all states. Every state has a dimension and an array associated with it.
    The methods are useful for printing the array and formatting it.
    """  

    def __init__(self):
        self.dim = None 
        self.state = None 

    def normalize(self):
        if not np.isclose(np.linalg.norm(self.state), 1):
            self.state /= np.linalg.norm(self.state)

    def __array__(self, dtype=np.complex128):
        return np.asarray(self.state, dtype=dtype)
    
    def __getitem__(self, index):
        return self.state[index]

    def __str__(self):
        self.draw=np.array2string(self.state, precision=2, separator='  ', formatter={'complex_kind': lambda z: f"{z.real:g}" if np.isclose(z.imag, 0) else (f"{z.imag:g}j" if np.isclose(z.real, 0) else f"{z:.3f}")} )
        return self.draw

    


class State(abs_State):

    """ The class represent a single qudit state. For initialization you must give an array and a index, which represents the wire number. 
    A single qudit state must have dimension = Dim. 
    Every state is automatically normalized. """

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
            raise ValueError('State must have an identification number')
        
        self.num = N 

        self.normalize()


    
        
   

class Total_state(abs_State):     

    """The class represent a total state. It was decided to adopt a linear representation 
    with the most significant qudit on the left. The state is automatically normalized.              """     #total state    example: |000>   

    def __init__(self, vett):   

        super().__init__()
        if vett is None:
                    raise ValueError ('Insert your state')

        vett = np.asarray(vett)

        num_qudits = round(math.log(len(vett), config.DIM))
        
        if num_qudits < 1 or config.DIM ** num_qudits != len(vett):
            raise ValueError('Total state length must be a power of the qudit dimension')
        
        self.dim=len(vett)

        self.state=vett

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, vett):
        if vett is None:
            self._state = None
            return

        vett = np.asarray(vett, dtype=np.complex128)
        norm = np.linalg.norm(vett)
        if np.isclose(norm, 0):
            raise ValueError('State does not exists')
        if not np.isclose(norm, 1):
            vett = vett / norm

        self._state = vett
        


def create_state(lista): 
    
    """A function that accept a list of states and compute the total state,
    which is the product tensor of the single-state qudit. It has dimension = dim^n.
    The states must be given in order from the most sigificant to the less significant """

    S=lista[0].state
    for i in range(len(lista)-1):
        S=np.kron(S, lista[i+1].state)
    
    new_state=Total_state(S)

    return new_state




#----------------------------------------------------------------------------------------------------------------------------------------------------------------




def measure(state):

    """A function that simulate a state's measure. It returns the index of the state """

    prob=np.abs(state)**2
    m=np.random.choice(len(state.state), p=prob)
    st=indexes(m, len(state.state))
    return st



def measure_prob(state):

    """A function that compute all the probabilities and print them associated with their index """

    prob=np.abs(state)**2
    result=[]
    for l in range(len(state.state)):
        if not np.isclose(prob[l], 0):
            result.append([indexes(l, len(state.state)), round(float(prob[l]), 4)])
    return result



def measure_single(state, q: int, collapse:bool = False):

    """A function that simulate a qudit's measure. The qudit's number must be given 
    It returns the index of the state. If collapse is true, the function return also the collaplsed state """

    dim=config.DIM
    num=round(math.log(len(state.state), dim))
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
    """A function that compute all the probabilities for a single qudit and print
     them associated with their index """
    dim=config.DIM
    num=round(math.log(len(state.state), dim))
    st = state.state.reshape([dim] * num)
    ax=tuple(i for i in range(num) if i != q)
    prob = np.sum(np.abs(st)**2, axis=ax)
    result=[]
    for i in range(len(prob)):
        if not np.isclose(0, prob[i]):
            result.append([i, round(float(prob[i]), 4)])
    return result



def indexes(pos:int, l:int):    

    """given the array's lenght and the position of an array's entrance, it gives the associated index"""

    dim=config.DIM
    num=round(math.log(l, dim))
    digits=[]
    i=num-1
    while i>=0:
        digits.append(pos//dim**i)
        pos=pos%(dim**i)
        i=i-1
    return digits 



