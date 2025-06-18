import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
from scipy.integrate import solve_ivp
import pickle

import EtoC
import EtoS
import TILDE


# 剛体Aの慣性行列
J_A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
# 剛体の質量
M_A = 10.0
# 重力加速度
g = 9.8
# 拘束点での摩擦を表現するための、剛体Aの角速度ベクトルに対してかける減衰係数(ここではゼロ)
C_friction = 0.0


# 剛体上の拘束点P1を示す、剛体上のベクトル
r_AP1 = np.array([[0.0], [0.0], [1.0]])
# あとで使うゼロ行列
zero_matrix = np.zeros((3, 3))

# 剛体の質量行列の逆行列(3x3)、慣性行列の逆行列(3x3)、それらの複合行列(6x6)
inv_M_A = np.diag([1 / M_A, 1 / M_A, 1 / M_A])
inv_J_A = np.linalg.inv(J_A)
inv_M_matrix = np.vstack((np.hstack((inv_M_A, zero_matrix)), np.hstack((zero_matrix, np.linalg.inv(J_A)))))



# 運動方程式、一階の常微分方程式の形、すなわち状態方程式形式で表現する。左辺が返り値
def func_eom(t, X):
    # 引数から固定パラメタの取り出し
    R_OA     = np.array([[X[0]],[X[1]],[X[2]]]).reshape(3,1)
    E_OA     = np.array([X[3],X[4],X[5],X[6]]).reshape(4,1)
    V_OA     = np.array([[X[7]],[X[8]],[X[9]]]).reshape(3,1)
    Omega_OA = np.array([[X[10]],[X[11]],[X[12]]]).reshape(3,1)

    # 座標変換行列の作成
    C_OA = EtoC.EtoC(E_OA)

    # 拘束条件式 (Ψ：位置レベル、Φ：速度レベル)
    PSI = R_OA + C_OA @ r_AP1
    PHI = V_OA - C_OA @ TILDE.TILDE(r_AP1) @ Omega_OA
 
    # 速度拘束条件の速度ベクトルV_OAによる変微分（V_OBが登場する場合、PHI_V_OBも用意する必要がある）
    PHI_V_OA = np.diag([1, 1, 1])
    # 速度拘束条件の角速度ベクトルOmega_OAによる変微分
    PHI_Omega_OA = -C_OA @ TILDE.TILDE(r_AP1)

    # 速度レベルの拘束条件の、全ての速度ベクトルによる変微分結果を横に並べた行列
    PHI_V = np.hstack((PHI_V_OA, PHI_Omega_OA))

    # 速度ベクトルの拘束条件式の、時間微分の式におけるd/dt V, d/dt Omega に関わらない項
    # Φの時間微分(Ψの時間での2階微分)を作ると、右辺= 〇〇 * dV/dt + △△ * dOmega/dt + □□　の形で表現できる。この □□ が "dPHI_R"
    dPHI_R = -C_OA @ TILDE.TILDE(Omega_OA) @ TILDE.TILDE(r_AP1) @ Omega_OA

    # 剛体Aに作用する、世界座標系で見た外力のベクトル(ここでは重力加速度による力)
    F_OA = np.array([[0.0], [0.0], [-M_A * g]])
    # 剛体Aに作用する "剛体座標系" で見たトルクベクトル（回転に関する項は、剛体固定座標系で見るため）。
    # 第1項：任意の作用トルク（ここではゼロ、重力は重心に作用するのでトルクは生じない）、第2項：減衰のトルク(ここではゼロ)、第3項：オイラーの方程式の右辺の項 　
    N_OA = np.array([[0.0], [0.0 ], [0.0]]) + (-C_friction) * Omega_OA - TILDE.TILDE(Omega_OA) @ J_A @ Omega_OA

    # 全ての外力、外トルクを合わせたベクトル。
    # OB, OCがいる時は、F=np.vstack((F_OA, N_OA, F_OC, N_OC, F_OC, N_OC)), あるいは、F=np.vstack((F_OA, F_OB, F_OC, N_OA, N_OB, N_OC))
    # 質量行列、慣性行列の並べ方に合わせて外力も並べること。（当然、力と質量行列、トルクと慣性行列が相当するように）
    F = np.vstack((F_OA, N_OA))

        
    # Lambdaの求め方その１
    #B = -dPHI_R - alpha * PHI - beta * PSI
    #Lambda = np.linalg.solve((PHI_V @ inv_M_matrix @ PHI_V.T), PHI_V @ inv_M_matrix @ F - B)
    
    #Lambdaの求め方その２（上ではPHIを１つのブロックにまとめている。まとめるかどうかの違いだけ）
    Lambda = np.linalg.solve((PHI_V_OA @ inv_M_A @ PHI_V_OA.T + PHI_Omega_OA @ inv_J_A @ PHI_Omega_OA.T),\
                              PHI_V_OA @ inv_M_A @ F_OA + PHI_Omega_OA @ inv_J_A @ (-TILDE.TILDE(Omega_OA) @ J_A @ Omega_OA) \
                                + dPHI_R )
    
    # 状態変数の微分、運動方程式の加速度に相当する項。
    dV = inv_M_matrix @ (F - PHI_V.T @ Lambda)
    
    # オイラーパラメタの時間微分
    # 姿勢表現にオイラー角を用いる場合は、ここにオイラー角の時間微分の式が入る
    # オイラーパラメタ、オイラー角、どちらを用いる場合でも、姿勢は角速度ベクトルを単に積分するだけではもとまらないことが超重要である。
    dE_OA = 0.5 * EtoS.EtoS(E_OA).T @ Omega_OA - 1/2/10 *E_OA @ (1-1/(E_OA.T @ E_OA))

    # return する変数の作成。前２つは、状態方程式の速度の微分の項。後ろのdVが、加速度の項（2階微分の項）
    DX = np.vstack((V_OA, dE_OA, dV))

    return DX.flatten()


# 実行時に最初に回る部分
if (__name__ == '__main__'):

    # 剛体の位置ベクトル、姿勢(オイラーパラメタ)、速度、角速度ベクトルの、「初期値」を定義する
    # R_init = (np.array([[0.0], [0.0], [-1.0]]).reshape(3,1)) 
    # E_init = np.array([[1.0], [0.0], [0.0], [0.0]])
    # V_init = np.array([[0.0], [0.0], [0.0]])
    # Omega_init = np.array([[0.0], [0.0], [1.0]])
    R_init = (np.array([[0.0], [0.50], [-0.8660]]).reshape(3,1)) 
    E_init = np.array([[0.9659], [0.2588], [0.0], [0.0]])# 7/13Uploadの資料で示した初期値np.array([[0.8776], [0.0], [0.0], [0.4794]])は誤りで、整合性の取れない初期値を入れたので、値の「飛び」が生じている。
    V_init = np.array([[0.0], [0.0], [0.0]])
    Omega_init = np.array([[0.0], [0.0], [0.0]])

    # 初期値をまとめて１つのベクトルにする。（並べ方は、運動方程式の並べ方と合わせる）
    X_init = np.vstack((R_init, E_init, V_init, Omega_init))
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
 
 
    ax1.plot3D(result.y[0,:], result.y[1,:], result.y[2,:], color=c2)
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
    ax5.set_xlabel("$t$ [s]") 
    ax5.set_ylabel(r"$E_1$") 

    ax6.plot(result.t, result.y[4,:], color=c2)
    ax6.set_xlabel("$t$ [s]") 
    ax6.set_ylabel(r"$E_2$") 
    
    ax7.plot(result.t, result.y[5,:], color=c3)
    ax7.set_xlabel("$t$ [s]") 
    ax7.set_ylabel(r"$E_3$")

    ax8.plot(result.t, result.y[6,:], color=c4)
    ax8.set_xlabel("$t$ [s]") 
    ax8.set_ylabel(r"$E_4$") 
  
    fig.tight_layout()
    plt.show()
    #plt.savefig("result.png")

    # 結果の入った構造体の保存（別のファイルで使用する時用)
    filename = 'ode_result.pkl'
    # Save the `OdeResult` object to a file using pickle
    with open(filename, 'wb') as file:
        pickle.dump(result, file)