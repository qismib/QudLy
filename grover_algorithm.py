import QudLy as q
import numpy as np
import matplotlib.pyplot as plt


q.set_dim(8)


array=np.zeros(q.config.DIM)
array[0]=1
initial_state=q.Total_state(array)
H=q.Gate_H(0)
R=q.Gate_Reflection(0)



result_states=[]


for i in range(q.config.DIM):

    O=q.Gate_Oracle(0, i)

    state=q.apply_gate(initial_state, H)

    for i in range(round((np.pi*np.sqrt(q.config.DIM))/4)):
        state=q.apply_gate(state, O)
        state=q.apply_gate(state, R)

    result_states.append(state)
    




result_prob=np.zeros((q.config.DIM, q.config.DIM))

for i in range(q.config.DIM):
    r=[]
    for j in range(1000):
        r.append(q.measure(result_states[i]))

    for k in r:
        result_prob[k[0]][i]+=1


result_prob=result_prob/1000

print(result_prob)  







fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(
    result_prob,
    cmap="Greens",
    vmin=0,
    vmax=1
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Success Probability", fontsize=14)

states=[]
for i in range(q.config.DIM):
    states.append(fr"$|{i}\rangle$")

ax.set_xticks(np.arange(q.config.DIM))
ax.set_yticks(np.arange(q.config.DIM))

ax.set_xticklabels(states, fontsize=14)
ax.set_yticklabels(states, fontsize=14)

ax.set_xlabel("Marked State", fontsize=16)
ax.set_ylabel("Measured State", fontsize=16)

# Testo dentro le celle
for i in range(result_prob.shape[0]):
    for j in range(result_prob.shape[1]):

        # Testo bianco nelle celle scure
        color = "white" if result_prob[i, j] > 0.45 else "black"

        ax.text(
            j, i,
            f"{result_prob[i, j]:.3f}",
            ha="center",
            va="center",
            color=color,
            fontsize=10
        )

plt.tight_layout()

plt.show()