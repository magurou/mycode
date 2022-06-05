import torch 
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import torch.optim as optim
from torchviz import make_dot
from IPython import display
from sklearn.datasets import load_boston

sampleData1 = np.array([[6000.0, 35000.0], [4500.0, 30000.0], [3500.0, 15000.0], [2000.0, 10000.0], [1500.0, 5000.0]])
#print(sampleData1)
##データの前処理
x = sampleData1[:,0] #身長だけ取る
y = sampleData1[:,1] #体重だけ取る
X = x - x.mean() #誤差を測るため全体の平均から離れている分を計算
Y = y - y.mean() #上と同様
#損失関数の計算 (平均２乗誤差)
def mse(Yp, Y):
    loss = ((Yp - Y) **2).mean()
    return loss
##予測計算
def predict(X):
    return W * X + B

#予測計算の準備
X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
#パラメータの定義
W = torch.tensor(1.0, requires_grad=True).float() #自動微分ができるようにする
B = torch.tensor(1.0, requires_grad=True).float()

#繰り返す数
num_epochs = 1000
#学習率の定義
lr = 0.001

#SGD(確率的勾配降下法)
optimizer = optim.SGD([W, B], lr = lr, momentum = 0.9) #momentumはチューニングを表す
#記録用の配列を定義する
history = np.zeros((0, 2))

#ループして学習をする
for epoch in range(num_epochs):

    ##予測計算
    Yp = predict(X)
    #print(Yp)
    ##損失の計算
    loss = mse(Yp, Y)
    #print(loss)
    #勾配計算
    loss.backward()
    #print(W.grad)
    #print(B.grad)
    ##勾配をもとにパラメータを修正する
    optimizer.step()

    #勾配値初期化
    #optimizer.zero_grad()
    ##勾配を使って計算が連続で行われるため、途中で変えると影響が出る
    ##よってパラメータを修正する際は一時的に計算グラフの生成機能を停止させる
    with torch.no_grad():
        W -= lr * W.grad
        B -= lr * B.grad
        #print(f'W.grad = {W.grad}')
        ##計算済みの勾配値をリセットする
        B.grad.zero_()
        W.grad.zero_()
        #print(f'W.grad = {W.grad}')
    #損失の計算
    if(epoch % 10 == 0):
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoch = {epoch} loss = {loss:.4f}')

#パラメータの最終値
print('W = ', W.data.numpy())
print('B = ', B.data.numpy())

#損失の確認
print(f'初期状態; 損失 : {history[0, 1]: .4f}')
print(f'最終状態 : 損失 : {history[-1, 1] : .4f}')
#print(history.shape)

x_ml = np.arange(166, 200, 0.1)
x_ml = torch.tensor(x_ml)
y_ml = x_ml * W + B
y_ml = y_ml.detach().numpy()
x_ml = x_ml.detach().numpy()
#plt.plot(W.data, B.data,label ='test')
plt.scatter(x.data, y.data)
#plt.plot(x_ml, y_ml)
plt.legend()
plt.show()
