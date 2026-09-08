import QudLy as q
import numpy as np


q.set_dim(3)

X=q.Gate_X(1)
#X = X.dagger()
#K=Gate_pr(Dim)

#print(X)

Z=q.Gate_Z(2)
H=q.Gate_H(1)
P=q.Gate_P(1, np.pi/2)
X=q.Gate_X(1)
S=q.Gate_SUMX(2, 4)
Q=q.Gate_SUMP(1, 2, np.pi)
print(q.is_unitary(Q.matrix))
print(Q.name)

print(Q)


