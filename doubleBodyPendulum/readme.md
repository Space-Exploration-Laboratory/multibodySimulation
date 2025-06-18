# doubleBody3dPendulum.py
* 2剛体、2拘束の剛体振り子のシミュレーション
* 計算結果として、ode_result.pkl が出力される。

## doubleBody3dPendulumWithHingeJoint.py
* 関節が回転ジョイントの2重振り子

## doubleBody3dHoppingRover.py
* 回転ジョイント型の2重振り子を改変し、地面との接触を付加して微小重力天体でのHopping Rover Simulation として仕立てたもの。

# 実行方法
```
$ python3 doubleBody3dPendulum.py #実行
$ python3 plot3dDoubleRigidbody.py    #結果の描画
$ python3 plot3dHoppingRover.py       #hopping rover向けの描画
```
*アニメーションは描画するボックスサイズが違うだけで、描画のサイズは計算とは関係ない。計算に用いる慣性モーメントなどは別で定義するためである*


# read_ode_result.py
* 結果を ode_result.pkl から読み込み、グラフを描写


# plot3dRigidbody.py
* 結果を ode_result.pkl から読み込み、３次元のアニメーションを描写

# EtoC.py
* オイラーパラメタから方向余弦行列（座標変換行列）を作成

# Ang2E.py
* Z'X'Zのオイラー角から、相当するオイラーパラメタを生成
* 計算の初期値作成などにつかう