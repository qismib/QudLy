
DIM = None

def set_dim(dim):
    global DIM

    if dim is None:
        DIM = None
        return

    if not isinstance(dim, int) or dim < 2:
        raise ValueError("The dimension must be an integer >1")

    DIM = dim

