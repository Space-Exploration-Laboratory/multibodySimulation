import numpy as np

# 外積計算ようのチルダ演算子
def TILDE(A):
    # Tilde operator
    A = A.reshape(3,)
    TildeA = np.array([[0, -A[2], A[1]],
                      [A[2], 0, -A[0]],
                      [-A[1], A[0], 0]])
    TildeA = TildeA.reshape(3,3)
    return TildeA
