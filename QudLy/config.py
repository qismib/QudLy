
DIM = None

def set_dim(dim):
    global DIM

    if DIM is not None:
        raise RuntimeError("DIM è già stata inizializzata")

    DIM = dim

