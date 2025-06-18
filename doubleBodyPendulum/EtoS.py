import numpy as np

# オイラーパラメタの時間微分の計算に用いる中間変数Sの作成用の関数
def EtoS(E):
    # Euler parameters
    E0 = E[0]
    E1 = E[1]
    E2 = E[2]
    E3 = E[3]
    # Skew-symmetric matrix (S)
    S = np.array([[-E1, E0, E3, -E2],
                  [-E2, -E3, E0, E1],
                  [-E3, E2, -E1, E0]])
    S = S.reshape(3,4)
    return S

