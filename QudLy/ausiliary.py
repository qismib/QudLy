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


'''
def apply_gate_2(st, gate):   #total state,  gate array

    dim=config.DIM     
    num=int(math.log(len(st.state), dim))    
    result=1   
    lis=[np.eye(dim) for _ in range(num)]
    i=0

    for g in gate:
        if g.is_controlled:
            lis[g.target_qudits[0]]=[g.base_matrix]
            lis[g.control_qudits[0]]=g.target_qudits
            
        else:
            lis[g.target_qudits[0]]=g.matrix

          
    while i<len(lis):
        if isinstance(lis[i], np.ndarray):
            result=np.kron(result, lis[i])
            i+=1
        else:
            pos=i
            U= None 
            for y in range(dim):
                if isinstance(lis[i], tuple):
                    M=np.zeros((dim, dim), dtype=np.complex128)
                    M[y, y]=1
                    i+=1
                    while not isinstance(lis[i], list):
                        M=np.kron(M, lis[i])
                        i+=1
                    M=np.kron(M, np.linalg.matrix_power(lis[i][0], y))
                    
                else:
                    M=np.linalg.matrix_power(lis[i][0], y)
                    i+=1
                    while not isinstance(lis[i], tuple):
                        M=np.kron(M, lis[i])
                        i+=1
                    Y=np.zeros((dim, dim), dtype=np.complex128)
                    Y[y, y]=1
                    M=np.kron(M, Y)
                    
                if U is None:
                    U=M.copy()
                else:
                    U=U+M
                tot=i
                i=pos
            result=np.kron(result, U)
            if i==0:
                i=i+tot+1
            else:
                i=i+tot

    Result=Total_state(result@st.state)
    return Result

'''