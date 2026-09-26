from .config  import set_dim
from .lib import Gate, Gate_X, Gate_Z, Gate_H, Gate_P, Gate_SUMX, Gate_SUMP, Gate_CZ, State, apply_gate, Total_state, create_state, apply_QFT, apply_CX, apply_SWAP
from .ausiliary import is_unitary
from .lib import measure, indexes, measure_prob, measure_single, measure_prob_single, apply_gate_2