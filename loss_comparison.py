"""
Code for comparing different cost function on the same network, namely quadratic cost,
softmax log loss and multiclass cross entropy. Evaluated on same model as in activation_comparison.py
with sigmoid activation function and using gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_blobs #dataset generator
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(2025)

N = 50 #number of training points
N_test = 25 #number of validation points
classes = 3
#create training data
X, y = make_blobs(n_samples= N+N_test, centers = classes,cluster_std=0.5,random_state=0)
train_X = X[:N]; train_y = y[:N] #first N points are training data
test_X = X[N:]; test_y = y[N:] #remaining points are validation data
#print(test_y) #see class labels

#make y into arrays suitable for training
train_y_arr = np.zeros((N,3)) #make into array
train_y_arr[train_y == 0] = [1,0,0]; train_y_arr[train_y == 1] = [0,1,0]; train_y_arr[train_y == 2] = [0,0,1]
test_y_arr = np.zeros((N_test,3))
test_y_arr[test_y == 0] = [1,0,0]; test_y_arr[test_y == 1] = [0,1,0]; test_y_arr[test_y == 2] = [0,0,1]

#colors for plot of data
train_c = np.array(len(train_y)*['#86A4F7'])
train_c[train_y==1] = '#6CE1AB'; train_c[train_y==2] = '#FB6666'

test_c = np.array(len(test_y)*['#86A4F7'])
test_c[test_y==1] = '#6CE1AB'; test_c[test_y==2] = '#FB6666'

#plot training and validation data
plt.scatter(*zip(*train_X),c=train_c)
plt.scatter(*zip(*test_X),c=test_c,marker='x')
plt.savefig('plot/test_blobs_validation.png',dpi=600)
plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#make data into tensors
x_data = torch.from_numpy(train_X).float()
y_data = torch.from_numpy(train_y_arr).float()

x_test = torch.from_numpy(test_X).float()
y_test = torch.from_numpy(test_y_arr).float()

class NN(nn.Module):
    def __init__(self, normalization = False):
        super().__init__()
        self.l2 = nn.Linear(2,4)
        self.l3 = nn.Linear(4,3)
        self.normalize = normalization

    def forward(self,x):
        x = F.sigmoid(self.l2(x))
        x = F.sigmoid(self.l3(x))
        if self.normalize: #Normalize if normalization is set to True
            x = F.normalize(x,1,dim=1) #L1 normalization
        return x

def train_loop(model, input,target, loss_fn, optimizer, cost):
    model.train()
    optimizer.zero_grad()
    pred = model(input)
    loss = loss_fn(pred,target)
    cost.append(loss.item())
    loss.backward()
    optimizer.step()

def test_loop(model, input, target, loss_fn, test_cost):
    """Evaluate network on test data"""
    model.eval()
    pred = model(input)
    loss = loss_fn(pred,target)
    test_cost.append(loss.item())

def model_accuracy(model):
    """Test predicrtion accuracy on validation data"""
    pred_values = model(x_test)
    prediction = torch.argmax(pred_values, dim = 1) #find index of largest value in each network output
    count = 0
    for (pred, target) in zip(prediction,test_y): #compare predicted and target category
        if pred.item() == target:
            count += 1 #add if prediction is correct
    print(count)
    return count/N_test #return ratio of accurate prediction

def cross_entropy(v,y):
    """Categorical Cross-Entropy Loss"""
    return -torch.mean(torch.sum(y*torch.log(v),1))

Niter = int(1e5)
iter = np.arange(Niter)
loss_funcs = {
    'Quadratic cost':  nn.MSELoss(),
    'Softmax Logloss': nn.CrossEntropyLoss(), #combines soft max with log loss
    'Cross Entropy':   cross_entropy
    }

#color palette
palette = {'blue':'#3E31D6', 'orange':'#FEA11B', 'green':'#2EC483', 'red':'#DD0000'}
fig, ax = plt.subplots()
for color, key in zip(palette, loss_funcs):
    print(f'Start {key}')
    #initiate model
    if key == 'Cross Entropy': #use normalization for CCE
        model = NN(normalization=True)
    else:
        model = NN()
    optimizer = optim.SGD(model.parameters(),lr=0.05)
    cost = []
    test_cost = []
    loss_fn = loss_funcs[key] #set loss function

    #train and test
    start_time = time.time()
    for t in tqdm(range(Niter)):
        train_loop(model, x_data, y_data, loss_fn, optimizer, cost) #train model for batch
        test_loop(model, x_test, y_test, loss_fn, test_cost) #test model on validation data
    time_used = time.time()-start_time
    it_s = round(Niter/time_used,2) #average iterations per second

    print(key,model(x_test)[:5]) #print some predictions of validation data
    acc = model_accuracy(model) #compute model accuracy on validation data
    print(f'| {key:15} | train_loss: {cost[-1]:.4e} | test_loss: {test_cost[-1]:.4e} | it/s: {it_s} | accuracy: {acc:.2%}|')

    ax.semilogy(iter,cost,color=palette[color],label=key)
    ax.semilogy(iter,test_cost,palette[color],alpha=0.5,linestyle='--')
    ax.set(xlabel='Iterations',ylabel='Loss')
plt.legend()
plt.tight_layout()
plt.savefig('plot/loss_comp.png',dpi=600)
plt.show()

"""
Results:
| Quadratic cost  | train_loss: 1.6833e-03 | test_loss: 1.6934e-03 | it/s: 1871.79 | accuracy: 100.00%|
| Softmax Logloss | train_loss: 5.5795e-01 | test_loss: 5.5857e-01 | it/s: 1495.16 | accuracy: 100.00%|
| Cross Entropy   | train_loss: 1.0791e-03 | test_loss: 3.4670e-03 | it/s: 1264.18 | accuracy: 100.00%|

Prediction of first five validation points
Quadratic cost:  [0.9680, 0.0299, 0.0216],
                 [0.0111, 0.0074, 0.9915],
                 [0.0184, 0.9815, 0.0058],
                 [0.0196, 0.9795, 0.0062],
                 [0.9703, 0.0327, 0.0183]

Softmax Logloss: [9.9207e-01, 5.7798e-03, 1.9800e-03],
                 [7.6507e-03, 1.8034e-05, 9.9955e-01],
                 [1.6964e-02, 9.9684e-01, 1.7574e-05],
                 [1.6576e-02, 9.9674e-01, 1.8550e-05],
                 [9.9283e-01, 6.2062e-03, 1.6889e-03]

Cross Entropy:   [9.9918e-01, 2.2552e-04, 5.8989e-04],
                 [2.3138e-04, 3.0047e-05, 9.9974e-01],
                 [1.3255e-04, 9.9981e-01, 6.2203e-05],
                 [1.6289e-04, 9.9977e-01, 6.7185e-05],
                 [9.9917e-01, 2.6657e-04, 5.6540e-04]
"""
