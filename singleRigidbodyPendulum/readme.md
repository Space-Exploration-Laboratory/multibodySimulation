# singleBody3dPendulum.py
* 1剛体、1拘束の剛体振り子のシミュレーション
* 使用方法： python3 Single_Body_3D_Pendulum.py
* 計算結果として、ode_result.pkl が出力される。
* 剛体振り子であるが、拘束点を変えればコマにもなる。

```
$ python3 singleBody3dPendulum.py #実行
$ python3 plot3dRigidbody.py    #結果の描画
```


# read_ode_result.py
* 結果を ode_result.pkl から読み込み、グラフを描写


# plot3dRigidbody.py
* 結果を ode_result.pkl から読み込み、３次元のアニメーションを描写


# functions.py
計算に必要な関数をまとめたもの。
mainの関数からインポートして使用する。

|Name|Function|
|---|---|
|EtoC|オイラーパラメタから方向余弦行列（座標変換行列）を作成|
|EtoS|EtoC の計算用のための中間変数|
|TILDE|外積計算用のオペレータ|
|Ang2E|$Z'X'Z$のオイラー角から、相当するオイラーパラメタ$E$を生成 <br> 計算の初期値作成などにつかう|


