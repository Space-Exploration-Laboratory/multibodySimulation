import numpy as np

# 計算に必要な関数の定義ファイル
# main関数側で必要なものをインポートして使用する

# Euler angle (ZXZ) から Euler Palameter を生成
def Ang2E(a):
    c1 = np.cos(a[0])
    c2 = np.cos(a[1])
    c3 = np.cos(a[2])
    s1 = np.sin(a[0])
    s2 = np.sin(a[1])
    s3 = np.sin(a[2])

    C = np.array([[c1*c3 - s1*c2*s3, -c1*s3-s1*c2*c3, s1*s2],
                  [s1*c3+c1*c2*s3, -s1*s3+c1*c2*c3, -c1*s2],
                  [s2*s3, s2*c3, c2]])

    dcm = C.T
    q = np.array([np.sqrt(1 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])*0.5,
                  np.sqrt(1 - dcm[0, 0] + dcm[1, 1] - dcm[2, 2])*0.5,
                  np.sqrt(1 - dcm[0, 0] - dcm[1, 1] + dcm[2, 2])*0.5,
                  np.sqrt(1 + dcm[0, 0] + dcm[1, 1] + dcm[2, 2])*0.5])

    x, ix = np.max(q), np.argmax(q)
    if ix == 0:
        q[1:4] = 0.25/q[0] * np.array([dcm[0, 1] + dcm[1, 0], dcm[0, 2] + dcm[2, 0], dcm[1, 2] - dcm[2, 1]])
    elif ix == 1:
        q[0], q[2], q[3] = 0.25/q[1] * np.array([dcm[0, 1] + dcm[1, 0], dcm[2, 1] + dcm[1, 2], dcm[2, 0] - dcm[0, 2]])
    elif ix == 2:
        q[0], q[1], q[3] = 0.25/q[2] * np.array([dcm[2, 0] + dcm[0, 2], dcm[2, 1] + dcm[1, 2], dcm[0, 1] - dcm[1, 0]])
    elif ix == 3:
        q[0:3] = 0.25/q[3] * np.array([dcm[1, 2] - dcm[2, 1], dcm[2, 0] - dcm[0, 2], dcm[0, 1] - dcm[1, 0]])

    E = np.array([q[3], q[0], q[1], q[2]]).reshape(4,1)

    return E
# 参考文献　三菱スペースソフトウェア社 www.mss.co.jp/technology/report/pdf/19-08.pdf
# ソースコード編集の一部には、Chat GPT を使用


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


# 外積計算ようのチルダ演算子
def TILDE(A):
    # Tilde operator
    A = A.reshape(3,)
    TildeA = np.array([[0, -A[2], A[1]],
                      [A[2], 0, -A[0]],
                      [-A[1], A[0], 0]])
    TildeA = TildeA.reshape(3,3)
    return TildeA



# Euler parameters to Z-X-Z Euler angles
def E2Ang_ZXZ(E):
    # Normalize quaternion
    E = E.flatten()
    q0, q1, q2, q3 = E[0], E[1], E[2], E[3]
    norm = np.linalg.norm(E)
    q0, q1, q2, q3 = q0/norm, q1/norm, q2/norm, q3/norm

    # Convert quaternion to DCM
    dcm = np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),       2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),         1 - 2*(q1**2 + q3**2),   2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),         2*(q2*q3 + q0*q1),       1 - 2*(q1**2 + q2**2)]
    ])

    # Extract ZXZ Euler angles from DCM
    if abs(dcm[2,2]) < 1.0:
        theta_1 = np.arccos(dcm[2,2])
        theta_2 = np.arctan2(dcm[0,2], -dcm[1,2])
        theta_3 = np.arctan2(dcm[2,0], dcm[2,1])
    else:
        # Gimbal lock case (beta = 0 or pi)
        theta_1 = 0 if dcm[2,2] > 0 else np.pi
        theta_2 = 0
        theta_3 = np.arctan2(-dcm[0,1], dcm[0,0])

    return np.array([theta_1, theta_2, theta_3])
