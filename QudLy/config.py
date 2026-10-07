
DIM = None

def set_dim(dim):
    global DIM

    if DIM is not None:
        raise RuntimeError("DIM has already been initialized")

    if not isinstance(dim, int) or dim < 2:
        raise ValueError("The dimension must be an integer >1")
    

    DIM = dim

