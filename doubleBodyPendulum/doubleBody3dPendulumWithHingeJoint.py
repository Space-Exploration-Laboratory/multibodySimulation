import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
from scipy.integrate import solve_ivp
import pickle
#from scipy.linalg import block_diag
from functions import EtoS
from functions import EtoC
from functions import TILDE
from functions import Ang2E

# 剛体Aの慣性行列
J_A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
J_B = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
# 剛体の質量
m_A = 10.0
m_B = 10.0
# 重力加速度
g = 9.8
# 拘束点での摩擦を表現するための、剛体Aの角速度ベクトルに対してかける減衰係数(ここではゼロ)
C_friction = 0.0

# Baumgarteの拘束安定化法の係数(ゼロでも計算可能だが、長時間の計算ではあった方が良い。α＝β=0との違いを見られたい)
alpha = 100
beta = 1000

# 剛体上の拘束点P1を示す、剛体上のベクトル
r_AP1 = np.array([[0.0], [0.0], [1.0]])
r_AP2 = np.array([[0.0], [0.0], [-1.0]])
r_BP1 = np.array([[0.0], [0.0], [1.0]])

#ヒンジの軸の方向ベクトル
n_O = np.array([1, 0, 0]).reshape(3,1) 
# 【 重 要 】
# 下で記した初期条件と一致させるなら、本来はここは[1; 0; 0]でなくてはならない
# しかし、ピンジョイントの動きを説明するため、あえて初期条件と整合しない拘束条件を設けている
# そのときは、徐々に拘束を満たすように計算される。
# 厳密に計算したいなら、初期条件をこの拘束条件と整合するようにすること
# n_0 = [1, 0, 1]なら、剛体A, BをY軸周りに事前に45度回転させておく必要あり。
# つまり、E_OA_init, E_OB_initを変更する必要がある。
# さらに位置R_OA_init, R_OB_initも変更する必要がある。
n_A = np.array([1, 0, 0]).reshape(3,1)
n_B = np.array([1, 0, 0]).reshape(3,1)

# あとで使うゼロ行列
zero_matrix = np.zeros((3, 3))

# 剛体の質量行列の逆行列(3x3)、慣性行列の逆行列(3x3)、それらの複合行列(6x6)
inv_M_A = np.diag([1 / m_A, 1 / m_A, 1 / m_A])
inv_M_B = np.diag([1 / m_B, 1 / m_B, 1 / m_B])
inv_J_A = np.linalg.inv(J_A)
inv_J_B = np.linalg.inv(J_B)
inv_M_matrix = linalg.block_diag(inv_M_A, inv_J_A, inv_M_B, inv_J_B)
print(inv_M_matrix)


# 運動方程式、一階の常微分方程式の形、すなわち状態方程式形式で表現する。左辺が返り値
def func_eom(t, X):
    # 引数から状態パラメタの取り出し
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
    # 天井と剛体Aの位置拘束（原点〜A上の点AP1）
    PSI1 =  R_OA + C_OA @ r_AP1
    # 剛体Aと剛体Bの位置拘束（A上の点AP2〜B上の点BP1）
    PSI2 = (R_OA + C_OA @ r_AP2) - (R_OB + C_OB @ r_BP1) 
    # 天井のヒンジベクトルn_0と、剛体Aのヒンジベクトルn_Aの向きが一致する拘束(内積＝0)
    PSI3 = TILDE(C_OA @ n_A) @ (n_O)
    # 剛体Aのヒンジベクトルn_Aと、剛体Bのヒンジベクトルn_Bの向きが一致する拘束(内積＝0)
    PSI4 = TILDE(C_OA @ n_A) @ (C_OB @ n_B)
    PSI = np.vstack((PSI1,PSI2,PSI3,PSI4))

    PHI1 =  V_OA - C_OA @ TILDE(r_AP1) @ Omega_OA
    PHI2 = (V_OA - C_OA @ TILDE(r_AP2) @ Omega_OA) - (V_OB - C_OB @ TILDE(r_BP1) @ Omega_OB)
    PHI3 = TILDE(-C_OA @ TILDE(n_A) @ Omega_OA) @ (n_O) 
    PHI4 = TILDE(-C_OA @ TILDE(n_A) @ Omega_OA) @ (C_OB @ n_B) + TILDE(C_OA @ n_A) @ (-C_OB @ TILDE(n_B) @ Omega_OB)
    PHI = np.vstack((PHI1, PHI2, PHI3, PHI4))
    
    # 速度拘束条件の速度ベクトルV_OAによる変微分（V_OBが登場する場合、PHI_V_OBも用意する必要がある）
    PHI1_V_OA = np.diag([1, 1, 1])
    PHI2_V_OA = np.diag([1, 1, 1])
    PHI3_V_OA = np.diag([0,0,0]) # PHI3をV_OAで変微分すると、何ものこらない
    PHI4_V_OA = np.diag([0,0,0]) # PHI4をV_OAで変微分すると、何ものこらない
    PHI_V_OA = np.vstack((PHI1_V_OA, PHI2_V_OA, PHI3_V_OA, PHI4_V_OA)) # 縦に並べる
    
    PHI1_V_OB = np.diag([0, 0, 0]) # PHI1の式には、V_OBは含まれないため
    PHI2_V_OB = (-1) * np.diag([1, 1, 1]) # 
    PHI3_V_OB = np.diag([0,0,0])
    PHI4_V_OB = np.diag([0,0,0])
    PHI_V_OB = np.vstack((PHI1_V_OB, PHI2_V_OB, PHI3_V_OB, PHI4_V_OB)) # 縦に並べる
    
    # 速度拘束条件の角速度ベクトルOmega_OAによる変微分
    PHI1_Omega_OA = -C_OA @ TILDE(r_AP1)
    PHI2_Omega_OA = -C_OA @ TILDE(r_AP2) 
    PHI3_Omega_OA = -TILDE(n_O) @ (-C_OA) @ TILDE(n_A)
    PHI4_Omega_OA = -C_OB @ TILDE(n_B) @ C_OB.T @ (-C_OA) @ TILDE(n_A)
    PHI_Omega_OA = np.vstack((PHI1_Omega_OA, PHI2_Omega_OA, PHI3_Omega_OA, PHI4_Omega_OA))

    PHI1_Omega_OB = np.diag([0, 0, 0])
    PHI2_Omega_OB = - (-C_OB @ TILDE(r_BP1))
    PHI3_Omega_OB = np.diag([0, 0, 0])
    PHI4_Omega_OB = C_OA @ TILDE(n_A) @ C_OA.T @ (-C_OB) @ TILDE(n_B)
    PHI_Omega_OB = np.vstack((PHI1_Omega_OB, PHI2_Omega_OB, PHI3_Omega_OB, PHI4_Omega_OB))

    # 速度レベルの拘束条件の、全ての速度ベクトルによる変微分結果を横に並べた行列
    PHI_V = np.hstack((PHI_V_OA, PHI_Omega_OA, PHI_V_OB, PHI_Omega_OB ))

    # 速度ベクトルの拘束条件式の、時間微分の式におけるd/dt V, d/dt Omega に関わらない項
    # Φの時間微分(Ψの時間での2階微分)を作ると、右辺= 〇〇 * dV/dt + △△ * dOmega/dt + □□　の形で表現できる。この □□ が "dPHI_R"
    dC_OA = C_OA @ TILDE(Omega_OA)
    dC_OB = C_OB @ TILDE(Omega_OB)
    tilde_n_A = TILDE(n_A)
    tilde_n_B = TILDE(n_B)
    tilde_n_O = TILDE(n_O)
    #print(tilde_n_O)
    #print(tilde_n_A)
    dPHI_R_row1 = -C_OA @ TILDE(Omega_OA) @ TILDE(r_AP1) @ Omega_OA
    dPHI_R_row2 = -C_OA @ TILDE(Omega_OA) @ TILDE(r_AP2) @ Omega_OA  - (-C_OB @ TILDE(Omega_OB) @ TILDE(r_BP1) @ Omega_OB)
    dPHI_R_row3 = ( -tilde_n_O @ (-dC_OA) @  tilde_n_O ) @ Omega_OA
    dPHI_R_row4 = (-dC_OB @ tilde_n_B @ C_OB.T @ (-C_OA) @ tilde_n_A  +  (-C_OB) @ tilde_n_B @ dC_OB.T @ (-C_OA) @ tilde_n_A  +  (-C_OB) @ tilde_n_B @ C_OB.T @ (-dC_OA) @ tilde_n_A ) @ Omega_OA \
                 +( dC_OA @ tilde_n_A @ C_OA.T @ (-C_OB) @ tilde_n_B  +  ( C_OA) @ tilde_n_A @ dC_OA.T @ (-C_OB) @ tilde_n_B  +  ( C_OA) @ tilde_n_A @ C_OA.T @ (-dC_OB) @ tilde_n_B ) @ Omega_OB
    dPHI_R = np.vstack((dPHI_R_row1, dPHI_R_row2, dPHI_R_row3, dPHI_R_row4))
    
    # 剛体Aに作用する、世界座標系で見た外力のベクトル(ここでは重力加速度による力)
    F_OA = np.array([[0.0], [0.0], [-m_A * g]])
    F_OB = np.array([[0.0], [0.0], [-m_B * g]])
    # 剛体Aに作用する "剛体座標系" で見たトルクベクトル（回転に関する項は、剛体固定座標系で見るため）。
    # 第1項：任意の作用トルク（ここではゼロ、重力は重心に作用するのでトルクは生じない）、第2項：減衰のトルク(ここではゼロ)、第3項：オイラーの方程式の右辺の項 　
    N_OA = np.array([[0.0], [0.0 ], [0.0]]) + (-C_friction) * (Omega_OA - Omega_OB)
    Right_Part_of_Rotation_EOM_for_Body_A = N_OA  - TILDE(Omega_OA) @ J_A @ Omega_OA
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
    # ランクの制約から、擬似逆行列を用いて解を求める
    
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
    R_A_init     = np.array([0.0, 0.0,-1.0]).reshape(3,1) 
    E_A_init     = np.array([1.0, 0.0, 0.0, 0.0]).reshape(4,1)
    V_A_init     = np.array([0.0, 0.0, 0.0]).reshape(3,1)
    Omega_A_init = np.array([0.0, 0.0, 0.0]).reshape(3,1)
    R_B_init     = np.array([0.0, 0.0, -3.0]).reshape(3,1) 
    E_B_init     = np.array([1.0, 0.0, 0.0, 0.0]).reshape(4,1)
    V_B_init     = np.array([0.0, 5.0, 0.0]).reshape(3,1)
    Omega_B_init = np.array([5.0, 0.0, 0.0]).reshape(3,1)
    
    # 初期値をまとめて１つのベクトルにする。（並べ方は、運動方程式の並べ方と合わせる）
    X_init = np.vstack((R_A_init, E_A_init, R_B_init, E_B_init, V_A_init, Omega_A_init, V_B_init, Omega_B_init))
    # 計算する時間の範囲を指定
    t_span = [0.0,10.0]
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