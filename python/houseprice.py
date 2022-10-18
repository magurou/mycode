import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden_1, n_hidden_2):
        super().__init__()

        self.l1 = nn.Linear(n_input, n_hidden_1)
        self.l2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.l3 = nn.Linear(n_hidden_2, n_output)
        #ReLu関数の定義
        self.relu = nn.ReLU(inplace = True)
        

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        x4 = self.relu(x3)
        x5 = self.l3(x4)
        return x5


def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns

train = pd.read_csv(
    "C:/Users/hiroki/Desktop/Python_lesson/kaggle/houseprice/train.csv")
test = pd.read_csv(
    "C:/Users/hiroki/Desktop/Python_lesson/kaggle/houseprice/test.csv")
#print(test.shape, train.shape)
all = pd.concat([train.drop(columns = "SalePrice"), test])
#print(train.shape)
num2str_list = ['MSSubClass','YrSold','MoSold']
for column in num2str_list:
    all[column] = all[column].astype(str)
# 変数の型ごとに欠損値の扱いが異なるため、変数ごとに処理
for column in all.columns:
    # dtypeがobjectの場合、文字列の変数
    if all[column].dtype=='O':
        all[column] = all[column].fillna('None')
    # dtypeがint , floatの場合、数字の変数
    else:
        all[column] = all[column].fillna(0)

all = pd.get_dummies(all)

#print(train.index[0], train.index[-1])
# 学習データと予測データに分割して元のデータフレームに戻す。
train = pd.merge(all.iloc[train.index[0]:train.index[-1] + 1],train['SalePrice'],left_index=True,right_index=True)
test = all.iloc[train.index[-1] + 1:]
#print(test.shape, train.shape)
train = train[(train['LotArea']<20000) & (train['SalePrice']<400000)& (train['YearBuilt']>1920)]
train["SalePrice"] = np.log(train["SalePrice"])
x_train = train.drop(columns = ["SalePrice"])
y_train = train["SalePrice"]

n_input = x_train.shape[1]
#print(x_train.shape)
#print(test.shape)
#print(y_train.shape)
#print(n_input)
n_hidden_1 = 64
n_hidden_2 = 64

n_output = 1

net = Net(n_input, n_output, n_hidden_1, n_hidden_2).to(device)
lr = 0.01
#print(net)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = lr)
num_epochs = 1500


inputs = torch.tensor(x_train.values.tolist()).float()
labels = torch.tensor(y_train.values.tolist()).float().view((-1, 1))
#print(inputs)
history = np.zeros((0, 2))

for epoch in range(num_epochs):

    optimizer.zero_grad()

    outputs = net(inputs.to(device))
    #print(outputs)
    loss = criterion(outputs, labels.to(device))

    loss.backward()

    optimizer.step()
    #print(loss)
    if epoch % 1000 == 0:
        print(f"Epoch [{epoch} / {num_epochs}, loss: {loss.item():.5f}]")
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))

print(f"初期状態: 損失: {history[0, 1]:.5f}")
print(f"最終状態: 損失: {history[-1, 1]:.5f}")

pred = net(torch.tensor(test.values.tolist()).float().to(device))
pred = torch.exp(pred)
#print(pred)
id = np.array(test["Id"]).astype(int)
my_solution = pd.DataFrame(pred.cpu().view(-1).detach().numpy(), id, columns = ["SalePrice"])

my_solution.to_csv("titanic_prediction.csv", index_label = ["ID"])