import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
from scipy.integrate import solve_ivp
import pickle
#from scipy.linalg import block_diag


# 剛体Aの慣性行列
J_A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]) * 0.5
J_B = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]) * 0.1
# 剛体の質量
m_A = 1.0
m_B = 0.30
# 重力加速度
g = 0.01
# 拘束点での摩擦を表現するための、剛体Aの角速度ベクトルに対してかける減衰係数(ここではゼロ)
C_friction = 0.05

# Baumgarteの拘束安定化法の係数(ゼロでも計算可能だが、長時間の計算ではあった方が良い。α＝β=0との違いを見られたい)
alpha = 100
beta = 1000

# 剛体上の拘束点P1を示す、剛体上のベクトル
#r_AP1 = np.array([[0.0], [0.0], [0.0]])
r_AP2 = np.array([[0.0], [0.0], [0.0]])
r_BP1 = np.array([[0.0], [0.0], [0.0]])

#ヒンジの軸の方向ベクトル
#n_O = np.array([1, 0, 0]).reshape(3,1)
n_A = np.array([1, 0, 0]).reshape(3,1)
n_B = np.array([1, 0, 0]).reshape(3,1)
n_G = np.array([0,0,1]).reshape(3,1)

# contact points
# ロボットの地面との接触を表現するための点を4つ用意している。（底面側のみ４点を表現）
# 地面との接触は、車のように離れないことが前提なら拘束条件を使うが、
# 宇宙ロボットでは表面とロボットが離れることもあるので、ばねダンパを使い、沈下量に応じた反力を返すモデルを用いる。
# ペナルティ法とも呼ばれる。
# ベッカーモデルや、RFTなど、よりリアルなモデルを用いることもある。
xA = 0.2
yA = 0.2
zA = 0.1 # このxA ~ zAをアニメーションのプログラムと合わせることで、接触点と同じ形のCubeを描写できる。
R_contactpoint_1 = np.array([-xA/2, -yA/2, -zA/2]).reshape(3,1)
R_contactpoint_2 = np.array([-xA/2, yA/2, -zA/2]).reshape(3,1)
R_contactpoint_3 = np.array([xA/2, -yA/2, -zA/2]).reshape(3,1)
R_contactpoint_4 = np.array([xA/2, yA/2, -zA/2]).reshape(3,1)
k = 1000 #地面との接触のばね定数
c = 100 # 地面との接触のダンパ係数
mu = 0.6 # 地面との接触の動摩擦係数（静止摩擦は複雑なので無視している)

# あとで使うゼロ行列
zero_matrix = np.zeros((3, 3))

# 剛体の質量行列の逆行列(3x3)、慣性行列の逆行列(3x3)、それらの複合行列(6x6)
inv_M_A = np.diag([1 / m_A, 1 / m_A, 1 / m_A])
inv_M_B = np.diag([1 / m_B, 1 / m_B, 1 / m_B])
inv_J_A = np.linalg.inv(J_A)
inv_J_B = np.linalg.inv(J_B)
inv_M_matrix = linalg.block_diag(inv_M_A, inv_J_A, inv_M_B, inv_J_B)
print(inv_M_matrix)

# オイラーパラメタから、座標変換行列(方向余弦行列)への変換関数
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

# ある任意の点の位置と速度と、その点が横切る平面の法線ベクトルを与えることで、平面との接触をばねダンパモデルで与える関数
def TerrainForce(n, Rc, Vc):
    Rn = (Rc.T @ n) * n
    Vn = (Vc.T @ n) * n
    FTC = -Vn * c
    FTK = -Rn * k
    Rt = Rc - Rn
    Vt = Vc - Vn
    E_Vt = Vt/(np.sqrt(Vt.T @ Vt + 0.0001)) # 0.0001は0割り防止のおまじない
    FTMu = -mu * np.linalg.norm(FTC + FTK) * E_Vt
    FT = FTK + FTC + FTMu
    return FT


# 運動方程式、一階の常微分方程式の形、すなわち状態方程式形式で表現する。左辺が返り値
def func_eom(t, X):
    # 引数から状態変数の取り出し
    R_OA     = X[0               :3].reshape(3,1)
    E_OA     = X[0+3             :3+4].reshape(4,1)
    R_OB     = X[3+4             :3+4+3].reshape(3,1)
    E_OB     = X[3+4+3           :3+4+3+4].reshape(4,1)
    V_OA     = X[3+4+3+4         :3+4+3+4+3].reshape(3,1)
    Omega_OA = X[3+4+3+4+3       :3+4+3+4+3+3].reshape(3,1).reshape(3,1)
    V_OB     = X[3+4+3+4+3+3     :3+4+3+4+3+3+3].reshape(3,1)
    Omega_OB = X[3+4+3+4+3+3+3   :3+4+3+4+3+3+3+3].reshape(3,1)

    # 座標変換行列の作成
    C_OA = EtoC(E_OA)
    C_OB = EtoC(E_OB)

    # 拘束条件式 (Ψ：位置レベル、Φ：速度レベル)
    # PSI1 =  R_OA + C_OA @ r_AP1        # ------------------> ホッピングロボットでは天井とr_AP1とは拘束されていない。そのため、コメントアウトしている。
    PSI2 = (R_OA + C_OA @ r_AP2) - (R_OB + C_OB @ r_BP1) 
    # PSI3 = TILDE(C_OA @ n_A) @ (n_O)    # -----------------> ここも上におなじ
    PSI4 = TILDE(C_OA @ n_A) @ (C_OB @ n_B)
    PSI = np.vstack((PSI2, PSI4))

    # PHI1 =  V_OA - C_OA @ TILDE(r_AP1) @ Omega_OA
    PHI2 = (V_OA - C_OA @ TILDE(r_AP2) @ Omega_OA) - (V_OB - C_OB @ TILDE(r_BP1) @ Omega_OB)
    # PHI3 = TILDE(-C_OA @ TILDE(n_A) @ Omega_OA) @ (n_O) 
    PHI4 = TILDE(-C_OA @ TILDE(n_A) @ Omega_OA) @ (C_OB @ n_B) + TILDE(C_OA @ n_A) @ (-C_OB @ TILDE(n_B) @ Omega_OB)
    PHI = np.vstack((PHI2, PHI4))
    
    # 速度拘束条件の速度ベクトルV_OAによる変微分（V_OBが登場する場合、PHI_V_OBも用意する必要がある）
    # PHI1_V_OA = np.diag([1, 1, 1])
    PHI2_V_OA = np.diag([1, 1, 1])
    # PHI3_V_OA = np.diag([0,0,0]) # PHI3をV_OAで変微分すると、何ものこらない
    PHI4_V_OA = np.diag([0,0,0]) # PHI4をV_OAで変微分すると、何ものこらない
    PHI_V_OA = np.vstack((PHI2_V_OA, PHI4_V_OA)) # 縦に並べる
    
    #PHI1_V_OB = np.diag([0, 0, 0]) # PHI1の式には、V_OBは含まれないため
    PHI2_V_OB = (-1) * np.diag([1, 1, 1]) # 
    #PHI3_V_OB = np.diag([0,0,0])
    PHI4_V_OB = np.diag([0,0,0])
    PHI_V_OB = np.vstack((PHI2_V_OB, PHI4_V_OB)) # 縦に並べる
    
    # 速度拘束条件の角速度ベクトルOmega_OAによる変微分
    #PHI1_Omega_OA = -C_OA @ TILDE(r_AP1)
    PHI2_Omega_OA = -C_OA @ TILDE(r_AP2) 
    #PHI3_Omega_OA = -TILDE(n_O) @ (-C_OA) @ TILDE(n_A)
    PHI4_Omega_OA = -C_OB @ TILDE(n_B) @ C_OB.T @ (-C_OA) @ TILDE(n_A)
    PHI_Omega_OA = np.vstack((PHI2_Omega_OA, PHI4_Omega_OA))

    #PHI1_Omega_OB = np.diag([0, 0, 0])
    PHI2_Omega_OB = - (-C_OB @ TILDE(r_BP1))
    #PHI3_Omega_OB = np.diag([0, 0, 0])
    PHI4_Omega_OB = C_OA @ TILDE(n_A) @ C_OA.T @ (-C_OB) @ TILDE(n_B)
    PHI_Omega_OB = np.vstack((PHI2_Omega_OB,  PHI4_Omega_OB))

    # 速度レベルの拘束条件の、全ての速度ベクトルによる変微分結果を横に並べた行列
    PHI_V = np.hstack((PHI_V_OA, PHI_Omega_OA, PHI_V_OB, PHI_Omega_OB ))

    # 速度ベクトルの拘束条件式の、時間微分の式におけるd/dt V, d/dt Omega に関わらない項
    # Φの時間微分(Ψの時間での2階微分)を作ると、右辺= 〇〇 * dV/dt + △△ * dOmega/dt + □□　の形で表現できる。この □□ が "dPHI_R"
    dC_OA = C_OA @ TILDE(Omega_OA)
    dC_OB = C_OB @ TILDE(Omega_OB)
    tilde_n_A = TILDE(n_A)
    tilde_n_B = TILDE(n_B)
    #tilde_n_O = TILDE(n_O)
    
    #dPHI_R_row1 = -C_OA @ TILDE(Omega_OA) @ TILDE(r_AP1) @ Omega_OA
    dPHI_R_row2 = -C_OA @ TILDE(Omega_OA) @ TILDE(r_AP2) @ Omega_OA  - (-C_OB @ TILDE(Omega_OB) @ TILDE(r_BP1) @ Omega_OB)
    #dPHI_R_row3 = ( -tilde_n_O @ (-dC_OA) @  tilde_n_O ) @ Omega_OA
    dPHI_R_row4 = (-dC_OB @ tilde_n_B @ C_OB.T @ (-C_OA) @ tilde_n_A  +  (-C_OB) @ tilde_n_B @ dC_OB.T @ (-C_OA) @ tilde_n_A  +  (-C_OB) @ tilde_n_B @ C_OB.T @ (-dC_OA.T) @ tilde_n_A ) @ Omega_OA \
                 +( dC_OA @ tilde_n_A @ C_OA.T @ (-C_OB) @ tilde_n_B  +  ( C_OA) @ tilde_n_A @ dC_OA.T @ (-C_OB) @ tilde_n_B  +  ( C_OA) @ tilde_n_A @ C_OA.T @ (-dC_OB.T) @ tilde_n_B ) @ Omega_OB
    dPHI_R = np.vstack((dPHI_R_row2, dPHI_R_row4))
    
    # z_contactpoint_1 = (R_OA + C_OA @ R_contactpoint_1)[2]
    # z_contactpoint_2 = (R_OA + C_OA @ R_contactpoint_2)[2]
    # z_contactpoint_3 = (R_OA + C_OA @ R_contactpoint_3)[2]
    # z_contactpoint_4 = (R_OA + C_OA @ R_contactpoint_4)[2]
    # v_contactpoint_1 = (V_OA + dC_OA @ R_contactpoint_1)[2]
    # v_contactpoint_2 = (V_OA + dC_OA @ R_contactpoint_2)[2]
    # v_contactpoint_3 = (V_OA + dC_OA @ R_contactpoint_3)[2]
    # v_contactpoint_4 = (V_OA + dC_OA @ R_contactpoint_4)[2]
    Rc_1 = (R_OA + C_OA @ R_contactpoint_1)
    Rc_2 = (R_OA + C_OA @ R_contactpoint_2)
    Rc_3 = (R_OA + C_OA @ R_contactpoint_3)
    Rc_4 = (R_OA + C_OA @ R_contactpoint_4)
    Vc_1 = (V_OA + dC_OA @ R_contactpoint_1)
    Vc_2 = (V_OA + dC_OA @ R_contactpoint_2)
    Vc_3 = (V_OA + dC_OA @ R_contactpoint_3)
    Vc_4 = (V_OA + dC_OA @ R_contactpoint_4)
   
    if Rc_1[2] < 0: # Z軸が地面より下なら
        #F_contactpoint_1 = np.array([[0],[0],  -k * z_contactpoint_1 - c * v_contactpoint_1]).reshape(3,1)
        F_contactpoint_1 = TerrainForce(n_G, Rc_1, Vc_1)
    else:
        F_contactpoint_1 = np.array([0, 0, 0]).reshape(3,1)

    #print(F_contactpoint_1)
    if Rc_2[2] < 0:
        #F_contactpoint_2 = np.array([[0],[0],  -k * z_contactpoint_2 - c * v_contactpoint_2]).reshape(3,1)
        F_contactpoint_2 = TerrainForce(n_G, Rc_2, Vc_2)
    else:
        F_contactpoint_2 = np.array([0, 0, 0]).reshape(3,1)
    
    if Rc_3[2] < 0:
        #F_contactpoint_3 = np.array([[0],[0],  -k * z_contactpoint_3 - c * v_contactpoint_3]).reshape(3,1)
        F_contactpoint_3 = TerrainForce(n_G, Rc_3, Vc_3)
    else:
        F_contactpoint_3 = np.array([0, 0, 0]).reshape(3,1)
    
    if Rc_4[2] < 0:
        #F_contactpoint_4 = np.array([[0],[0],  -k * z_contactpoint_4 - c * v_contactpoint_4]).reshape(3,1)
        F_contactpoint_4 = TerrainForce(n_G, Rc_4, Vc_4)
    else:
        F_contactpoint_4 = np.array([0, 0, 0]).reshape(3,1)
        
    F_External_A = F_contactpoint_1 + F_contactpoint_2 + F_contactpoint_3 + F_contactpoint_4
    N_External_A = TILDE(R_contactpoint_1) @ C_OA.T @ F_contactpoint_1 \
                  +TILDE(R_contactpoint_2) @ C_OA.T @ F_contactpoint_2 \
                  +TILDE(R_contactpoint_3) @ C_OA.T @ F_contactpoint_3 \
                  +TILDE(R_contactpoint_4) @ C_OA.T @ F_contactpoint_4
    # 接触点で作用する力は、剛体Aの重心に作用する力ではない。
    # そのため、力により生じる偶力（力のモーメント、トルク）を考慮する必要がある
    # ある点に作用する力が、重心周りに及ぼすトルクは、上のように
    # TILDE(取り付け点へのローカルベクトル) @ C_OA.T @ グローバル座標で見た作用力ベクトル
    # で求められる。

    F_OA = np.array([[0.0], [0.0], [-m_A * g]]) + F_External_A
    F_OB = np.array([[0.0], [0.0], [-m_B * g]])
    # 剛体Aに作用する "剛体座標系" で見たトルクベクトル（回転に関する項は、剛体固定座標系で見るため）。
    # 第1項：任意の作用トルク（ここではゼロ、重力は重心に作用するのでトルクは生じない）、第2項：減衰のトルク(ここではゼロ)、第3項：オイラーの方程式の右辺の項 　
    
    Motor_Torque = 0.0
    if t > 1 and t < 3: #一定時間だけ、トルクON
        Motor_Torque = 1.0

    N_OA = np.array([[Motor_Torque], [0.0 ], [0.0]]) + (-C_friction) * (Omega_OA - Omega_OB)
    Right_Part_of_Rotation_EOM_for_Body_A = N_OA  - TILDE(Omega_OA) @ J_A @ Omega_OA + N_External_A 
    # N_External_A は、反トルクをBに入れなくても良いのか？　ーー＞　Aに作用する外トルクなので、Aのみ
    N_OB = -N_OA
    Right_Part_of_Rotation_EOM_for_Body_B = N_OB  - TILDE(Omega_OB) @ J_B @ Omega_OB

    # 全ての外力、外トルクを合わせたベクトル。
    # OB, OCがいる時は、F=np.vstack((F_OA, N_OA, F_OC, N_OC, F_OC, N_OC)), あるいは、F=np.vstack((F_OA, F_OB, F_OC, N_OA, N_OB, N_OC))
    # 質量行列、慣性行列の並べ方に合わせて外力も並べること。（当然、力と質量行列、トルクと慣性行列が相当するように）
    F = np.vstack((F_OA, Right_Part_of_Rotation_EOM_for_Body_A, F_OB, Right_Part_of_Rotation_EOM_for_Body_B))

        
    # Lambdaの求め方その１
    B = -dPHI_R - alpha * PHI - beta * PSI
    #Lambda = np.linalg.solve((PHI_V @ inv_M_matrix @ PHI_V.T), PHI_V @ inv_M_matrix @ F - B)
    Lambda = np.linalg.pinv(PHI_V @ inv_M_matrix @ PHI_V.T) @ ( PHI_V @ inv_M_matrix @ F - B)
    
    #Lambdaの求め方その２（上ではPHIを１つのブロックにまとめている。まとめるかどうかの違いだけ）
    #Lambda = np.linalg.solve((PHI_V_OA @ inv_M_A @ PHI_V_OA.T + PHI_Omega_OA @ inv_J_A @ PHI_Omega_OA.T),\
    #                          PHI_V_OA @ inv_M_A @ F_OA + PHI_Omega_OA @ inv_J_A @ (-TILDE(Omega_OA) @ J_A @ Omega_OA) \
    #                            + dPHI_R + alpha * PSI + beta * PHI)
    
    # 状態変数の微分、運動方程式の加速度に相当する項。
    dV = inv_M_matrix @ (F - PHI_V.T @ Lambda)
    
    # オイラーパラメタの時間微分
    # 姿勢表現にオイラー角を用いる場合は、ここにオイラー角の時間微分の式が入る
    # オイラーパラメタ、オイラー角、どちらを用いる場合でも、姿勢は角速度ベクトルを単に積分するだけではもとまらないことが超重要である。
    damping = 10
    dE_OA = 0.5 * EtoS(E_OA).T @ Omega_OA - 1/2/damping *E_OA @ (1-1/(E_OA.T @ E_OA))
    dE_OB = 0.5 * EtoS(E_OB).T @ Omega_OB - 1/2/damping *E_OB @ (1-1/(E_OB.T @ E_OB))

    # return する変数の作成。前２つは、状態方程式の速度の微分の項。後ろのdVが、加速度の項（2階微分の項）
    DX = np.vstack((V_OA, dE_OA, V_OB, dE_OB, dV))

    return DX.flatten()


# 実行時に最初に回る部分
if (__name__ == '__main__'):

    # 剛体の位置ベクトル、姿勢(オイラーパラメタ)、速度、角速度ベクトルの、「初期値」を定義する
    R_A_init     = np.array([0.0, 0.0, zA/2]).reshape(3,1) 
    E_A_init     = np.array([1.0, 0.0, 0.0, 0.0]).reshape(4,1)
    V_A_init     = np.array([0.0, 0.0, 0.0]).reshape(3,1)
    Omega_A_init = np.array([0.0, 0.0, 0.0]).reshape(3,1)
    R_B_init     = np.array([0.0, 0.0, zA/2]).reshape(3,1) 
    E_B_init     = np.array([1.0, 0.0, 0.0, 0.0]).reshape(4,1)
    V_B_init     = np.array([0.0, 0.0, 0.0]).reshape(3,1)
    Omega_B_init = np.array([0.0, 0.0, 0.0]).reshape(3,1)
    
    # 初期値をまとめて１つのベクトルにする。（並べ方は、運動方程式の並べ方と合わせる）
    X_init = np.vstack((R_A_init, E_A_init, R_B_init, E_B_init, V_A_init, Omega_A_init, V_B_init, Omega_B_init))
    # 計算する時間の範囲を指定
    t_span = [0.0,20.0]
    # 微分方程式の計算。数値積分で、時間発展を計算する。RK45でルンゲ・クッタ法を指定、出力の刻み幅をt_evalで指定。
    result = solve_ivp(func_eom, t_span, X_init.flatten(), method='RK45',t_eval = np.linspace(*t_span,1001))

    print(result)


    # 結果のアウトプット
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax2 = fig.add_subplot(2, 4, 2)
    ax3 = fig.add_subplot(2, 4, 3)
    ax4 = fig.add_subplot(2, 4, 4)
    ax5 = fig.add_subplot(2, 4, 5)
    ax6 = fig.add_subplot(2, 4, 6)
    ax7 = fig.add_subplot(2, 4, 7)
    ax8 = fig.add_subplot(2, 4, 8)
    c1,c2,c3,c4 = "blue","green","red","black"
 
 
    ax1.plot3D(result.y[0,:], result.y[1,:], result.y[2,:], color=c1)
    ax1.plot3D(result.y[7,:], result.y[8,:], result.y[9,:], color=c2)
    ax1.set_xlabel(r"$x$") 
    ax1.set_ylabel(r"$y$") 
    ax1.set_zlabel(r"$z$")
    ax1.axis('equal')
    
    ax2.plot(result.t, result.y[0,:], color=c1)
    ax2.set_xlabel("$t$ [s]") 
    ax2.set_ylabel(r"$R_{Ax}$") 

    ax3.plot(result.t, result.y[1,:], color=c2)
    ax3.set_xlabel("$t$ [s]") 
    ax3.set_ylabel(r"$R_{Ay}$") 

    ax4.plot(result.t, result.y[2,:], color=c3)
    ax4.set_xlabel("$t$ [s]") 
    ax4.set_ylabel(r"$R_{Az}$") 

    ax5.plot(result.t, result.y[3,:], color=c1)
    ax5.plot(result.t, result.y[4,:], color=c2)
    ax5.plot(result.t, result.y[5,:], color=c3)
    ax5.plot(result.t, result.y[6,:], color=c4)
    ax5.plot(result.t, result.y[10,:], color=c1, linestyle='dashdot')
    ax5.plot(result.t, result.y[11,:], color=c2, linestyle='dashdot')
    ax5.plot(result.t, result.y[12,:], color=c3, linestyle='dashdot')
    ax5.plot(result.t, result.y[13,:], color=c4, linestyle='dashdot')
    ax5.set_xlabel("$t$ [s]") 
    ax5.set_ylabel(r"$E_{A, Bi}$") 

    ax6.plot(result.t, result.y[7,:], color=c1)
    ax6.set_xlabel("$t$ [s]") 
    ax6.set_ylabel(r"$R_{Bx}$") 
    
    ax7.plot(result.t, result.y[8,:], color=c2)
    ax7.set_xlabel("$t$ [s]") 
    ax7.set_ylabel(r"$R_{By}$")

    ax8.plot(result.t, result.y[9,:], color=c3)
    ax8.set_xlabel("$t$ [s]") 
    ax8.set_ylabel(r"$R_{Bz}$")

    fig.tight_layout()
    plt.show()
    #plt.savefig("result.png")

    # 結果の入った構造体の保存（別のファイルで使用する時用)
    filename = 'ode_result.pkl'
    # Save the `OdeResult` object to a file using pickle
    with open(filename, 'wb') as file:
        pickle.dump(result, file)