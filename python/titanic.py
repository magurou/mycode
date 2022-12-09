import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn



class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.sigmoid(x1)
        return x2


def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = pd.read_csv(
    "C:/Users/USERNAME/Desktop/Python_lesson/kaggle/titanic/train.csv")
test = pd.read_csv(
    "C:/Users/USERNAME/Desktop/Python_lesson/kaggle/titanic/test.csv")

# 欠損値を直す fillna()はNaNの値を()内のものに置き換える
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
# 文字を数値化
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})

train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})
# テスト用データも同様に行う
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()


x_train = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]]
y_train = train["Survived"]

x_test = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]]

tmp1 = x_train.values.tolist()
tmp2 = y_train.values.tolist()

sns.heatmap(train.corr('spearman'),  cmap='coolwarm', vmin=-1, vmax=1, annot=True)
plt.show()

n_input = x_train.shape[1]

n_output = 1

net = Net(n_input, n_output)

criterion = nn.BCELoss()

lr = 0.001

optimizer = optim.SGD(net.parameters(), lr=lr)

inputs = torch.tensor(x_train.values.tolist()).float()
labels = torch.tensor(y_train.values.tolist()).float().view((-1, 1))

inputs_test = torch.tensor(x_test.values.tolist()).float()

num_epochs = 25000
history = np.zeros((0, 3))

for epoch in range(num_epochs):

    optimizer.zero_grad()

    outputs = net(inputs)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    train_loss = loss.item()

    predicted = torch.where(outputs < 0.5, 0, 1)
    train_acc = (predicted == labels).sum() / len(y_train)

    if epoch % 10 == 0:
        print(f"Epoch [{epoch} / {num_epochs}, loss: {train_loss:.5f} acc: {train_acc:.5f}")
        item = np.array([epoch, train_loss, train_acc])
        history = np.vstack((history, item))

print(f"初期状態: 損失: {history[0, 1]:.5f} 精度: {history[0, 2]: .5f}")
print(f"最終状態: 損失: {history[-1, 1]:.5f} 精度: {history[-1, 2]: .5f}")

outputs_test = net(inputs_test)
my_prediction = torch.where(outputs_test < 0.5, 0, 1)
print(type(my_prediction))
PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction.view(-1).numpy(), PassengerId, columns = ["Survived"])

my_solution.to_csv("my_ml_one.csv", index_label = ["PassengerId"])

