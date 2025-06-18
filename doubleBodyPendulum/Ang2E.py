import numpy as np

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