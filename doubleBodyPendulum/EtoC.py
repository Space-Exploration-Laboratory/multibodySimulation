import numpy as np

def EtoC(E):
    # Euler parameters
    E = E.reshape(4,)
    E1 = E[0]
    E2 = E[1]
    E3 = E[2]
    E4 = E[3]
    # Direction Cosine Matrix (DCM)
    C = np.array([[E2**2 - E3**2 - E4**2 + E1**2, 2 * (E2 * E3 - E4 * E1), 2 * (E4 * E2 + E3 * E1)],
                  [2 * (E2 * E3 + E4 * E1), E3**2 - E4**2 - E2**2 + E1**2, 2 * (E3 * E4 - E2 * E1)],
                  [2 * (E4 * E2 - E3 * E1), 2 * (E3 * E4 + E2 * E1), E4**2 - E2**2 - E3**2 + E1**2]])
    C = C.reshape(3,3)
    return C



