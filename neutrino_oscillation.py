import QudLy as q
import numpy as np
import matplotlib.pyplot as plt


q.set_dim(3)

class result():
    def __init__(self, vett, theta):   
        self.res=vett
        self.angle=theta
        self.prob=[]
        self.theorical=()
        

def get_prob(result):

    count=np.zeros(9)
    for i in result.res:
        count[0]=count[0]+(i==[0, 0])
        count[1]=count[1]+(i==[0, 1])
        count[2]=count[2]+(i==[0, 2])
        count[3]=count[3]+(i==[1, 0])
        count[4]=count[4]+(i==[1, 1])
        count[5]=count[5]+(i==[1, 2])
        count[6]=count[6]+(i==[2, 0])
        count[7]=count[7]+(i==[2, 1])
        count[8]=count[8]+(i==[2, 2])
    count=count/1000
    result.prob=count
    return result.prob


def get_points(results, num):

    points=[]
    thetas=[]
    proba=[]


    for i in range(len(angles)):
        get_prob(results[num][1][i])
        temp=results[num][1][i].prob
        for j in range(9):
            if temp[j]!=0:
                points.append(j)
                thetas.append(results[num][1][i].angle)
                proba.append(temp[j])

    return thetas, points, proba
#----------------------------------------------------------------------------------------------------------------------------------------------------



angles = np.arange(0, 2*np.pi, np.pi/12)

results=[]


for i in range(9):
    state_tot=np.zeros(9)
    state_tot[i]=1
    state_tot=q.Total_state(state_tot)
    results.append([i, []])

    for t in angles:
        r=[]
        Gate_R=q.Gate_CR(1, 0, 2*t)
        result_state=q.apply_CNOT(state_tot, 0, 1)
        result_state=q.apply_gate(result_state, Gate_R)
        result_state=q.apply_CNOT(result_state, 0, 1)
        for j in range(1000):
            r.append(q.measure(result_state))
        temp=result(r, t)
        temp.theorical=q.measure_prob(result_state)
        results[i][1].append(temp)



state_tot=q.Total_state([0, 0, 1, 0, 0, 0, 0, 1, 0])
results.append(['|02>+|21>', []])
for t in angles:
    r=[]
    Gate_R=q.Gate_CR(1, 0, 2*t)
    result_state=q.apply_CNOT(state_tot, 0, 1)
    result_state=q.apply_gate(result_state, Gate_R)
    result_state=q.apply_CNOT(result_state, 0, 1)
    for j in range(1000):
        r.append(q.measure(result_state))
    temp=result(r, t)
    results[9][1].append(temp)




state_tot=q.Total_state([0, 0, 0, 0, 0, 2, 0, 0, 3])
results.append(['|12>+|22>', []])
for t in angles:
    r=[]
    Gate_R=q.Gate_CR(1, 0, 2*t)
    result_state=q.apply_CNOT(state_tot, 0, 1)
    result_state=q.apply_gate(result_state, Gate_R)
    result_state=q.apply_CNOT(result_state, 0, 1)
    for j in range(1000):
        r.append(q.measure(result_state))
    temp=result(r, t)
    results[10][1].append(temp)



state_tot=q.Total_state([0, 1, 0, 0, 1, 0, 1, 0, 0])
results.append(['|01>+|11>+|20>', []])
for t in angles:
    r=[]
    Gate_R=q.Gate_CR(1, 0, 2*t)
    result_state=q.apply_CNOT(state_tot, 0, 1)
    result_state=q.apply_gate(result_state, Gate_R)
    result_state=q.apply_CNOT(result_state, 0, 1)
    for j in range(1000):
        r.append(q.measure(result_state))
    temp=result(r, t)
    results[11][1].append(temp)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------



thetas, points, proba=get_points(results, 4)
lis=np.array(results[4][1][14].res)


#for i in range(24):
#    print(results[6][1][i].theorical)



labels = [
    r'$0$',
    r'$\frac{\pi}{12}$',
    r'$\frac{\pi}{6}$',
    r'$\frac{\pi}{4}$',
    r'$\frac{\pi}{3}$',
    r'$\frac{5\pi}{12}$',
    r'$\frac{\pi}{2}$',
    r'$\frac{7\pi}{12}$',
    r'$\frac{2\pi}{3}$',
    r'$\frac{3\pi}{4}$',
    r'$\frac{5\pi}{6}$',
    r'$\frac{11\pi}{12}$',
    r'$\pi$',
    r'$\frac{13\pi}{12}$',
    r'$\frac{7\pi}{6}$',
    r'$\frac{5\pi}{4}$',
    r'$\frac{4\pi}{3}$',
    r'$\frac{17\pi}{12}$',
    r'$\frac{3\pi}{2}$',
    r'$\frac{19\pi}{12}$',
    r'$\frac{5\pi}{3}$',
    r'$\frac{7\pi}{4}$',
    r'$\frac{11\pi}{6}$',
    r'$\frac{23\pi}{12}$'
]



his= lis[:, 0] * 2 + lis[:, 1]
w = 0.5
Bins = np.arange(-0.5, max(his) + w + 0.5, w)
plt.hist(his, edgecolor="#1ea38ff0", color="#32e0aff1",  bins=Bins)
plt.ylim(0, 1000)
plt.xlim(0, 9)
plt.xlabel('Stato finale')
plt.xticks(range(9), [r'$|00\rangle$', r'$|01\rangle$', r'$|02\rangle$', r'$|10\rangle$', r'$|11\rangle$', r'$|12\rangle$', r'$|20\rangle$', r'$|21\rangle$', r'$|22\rangle$'])
plt.ylabel('Frequenza')
plt.title('Misura di uno stato x')
plt.grid(axis='y', color="#bcbcbcf0", alpha=0.3)



fig, ax = plt.subplots(figsize=(10, 5))
fig.gca().set_axisbelow(True)
ax.grid(True, color="#e3e3e3f2")
ax.set(xlabel=('Theta(rad)'), ylabel='Final state', 
       )
fig.tight_layout()
plt.xlabel("Theta(rad)", fontsize=12)
plt.ylabel("Stato finale", fontsize=12)
graph = ax.scatter(thetas, points, c=proba, s=150, cmap='YlGnBu', vmin=0, vmax=1)
plt.xticks(angles, labels, fontsize=12)
plt.yticks(range(9), [r'$|00\rangle$', r'$|01\rangle$', r'$|02\rangle$', r'$|10\rangle$', r'$|11\rangle$', r'$|12\rangle$', r'$|20\rangle$', r'$|21\rangle$', r'$|22\rangle$'])
plt.ylim(-1, 9)
cbar=fig.colorbar(graph)
cbar.set_label("Frequenza relativa", fontsize=12)


counters = {4: 0, 4: 0}

for prob, x, y in zip(proba, thetas, points):

    offset = 12 if counters[y] % 2 == 0 else -15
    counters[y] += 1

    plt.annotate(
        prob,
        (x, y),
        xytext=(0, offset),
        textcoords='offset points',
        ha='center',
        va='bottom' if offset > 0 else 'top',
        fontsize=8
    )
#for x, y, prob in enumerate(zip(thetas, points , proba)):
 #   offset = 12 if i % 2 == 0 else -18


 #   plt.annotate(prob, (x, y), xytext=(-7, offset),
  #               textcoords='offset points', fontsize=8)


plt.show()



#print(results[0][1][12].angle)          #[num stato, [result1, result2, result3,....]]        dove result.res è un array