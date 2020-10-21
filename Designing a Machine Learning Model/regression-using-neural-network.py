import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import torch
from torch import optim

advertising_data = pd.read_csv('dataset/Advertising.csv', index_col=0)
advertising_data[['TV']] = preprocessing.scale(advertising_data[['TV']])
advertising_data[['radio']] = preprocessing.scale(advertising_data[['radio']])
advertising_data[['newspaper']] = preprocessing.scale(advertising_data[['newspaper']])


x = advertising_data.drop('sales', axis=1)
y = advertising_data['sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


#########convert to tensor format########
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)

#Features
inp = 3

#Output (Sales)
out = 1

hid = 100

loss_fn = torch.nn.MSELoss()
learning_rate = 0.0001

model = torch.nn.Sequential(torch.nn.Linear(inp, hid),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hid, out))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(10000):
    y_pred = model(x_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    if iter % 1000 == 0:
        print(iter, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


y_pred_tensor = model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()

print(r2_score(y_test, y_pred))

