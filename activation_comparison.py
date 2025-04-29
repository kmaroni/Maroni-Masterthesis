"""
Code to compare different activation function, namely Sigmoid, ReLU, Leaky ReLU and Silu. Use a feed foreward network
with (2,4,3) structure and using quadratic cost function. Also compared for batch sizes 1, 10, 25 and 50, where the former
is gradient descent.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_blobs #dataset
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(2025)

N = 50 #number of training points
classes = 3
X, y = make_blobs(n_samples= N, centers = classes,cluster_std=0.5,random_state=0)

#colors for plot of data
c = np.array(len(y)*['#86A4F7'])
c[y==1] = '#6CE1AB'; c[y==2] = '#FB6666'

plt.scatter(*zip(*X),c=c)
plt.axis('equal')
plt.savefig('plot/test_blobs.png',dpi=600)
plt.show()

y_ = np.zeros((N,3))
y_[y == 0] = [1,0,0]; y_[y == 1] = [0,1,0]; y_[y == 2] = [0,0,1]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

x_data = torch.from_numpy(X).float() #make into tensors
y_data = torch.from_numpy(y_).float()

class NN(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.l2 = nn.Linear(2,4)
        self.l3 = nn.Linear(4,3)
        self.act = act #set activation function

    def forward(self,x):
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        return x

def train_loop(model, input,target, loss_fn, optimizer, cost):
    model.train()
    optimizer.zero_grad()
    pred = model(input)
    loss = loss_fn(pred,target)
    cost.append(loss.item())
    loss.backward()
    optimizer.step()

loss_fn = nn.MSELoss()
Niter = int(1e5)

batch_size = 1
#batch_size = 10
#batch_size = 25
#batch_size = N

network_costs = [] #list of all costs of networks
act = {'Sigmoid':F.sigmoid, 'ReLU':F.relu, 'Leaky ReLU':F.leaky_relu, 'Silu':F.silu}

iter = np.arange(Niter)


#colour palette:
palette = {'blue':'#3E31D6', 'orange':'#FEA11B', 'green':'#2EC483', 'red':'#DD0000'}
for (key, color) in zip(act,palette):
    print(f'Start {key}')
    #initiate model
    model = NN(act[key]) #set activation function
    optimizer = optim.SGD(model.parameters(),lr=0.03)
    cost = []

    #train
    start_time = time.time()
    for t in tqdm(range(Niter)):
        if batch_size == N: #comment this out to test GD with shuffling
            #GD
            input = x_data; target = y_data
        else:
            #SGD with mini batch
            indices = torch.randperm(N)[:batch_size] #pick random indices for minibatch
            input = x_data[indices]; target = y_data[indices]
        train_loop(model, input, target, loss_fn, optimizer, cost)
    time_used = time.time()-start_time
    it_s = round(Niter/time_used,2)  #average iterations per second
    print(f'|{key:10}| loss: {cost[-1]:.4e}| it/s: {it_s}|')

    #plot
    if batch_size == N:
        step = 1
    else:
        step = 1000 #for readability, plot every thousand loss value for SGD.
    plt.semilogy(iter[::step],cost[::step],color=palette[color],label=key)

    if key == 'ReLU':
        cost_r = cost #save cost values for ReLU to plot later
        #print(model(x_data)) #print network output to see why ReLU performs worse

plt.xlabel('Iterations')
plt.ylabel('Trainging loss')
plt.title(f'batch size = {batch_size}')
plt.legend()
plt.savefig(f'plot/activation_comp_{batch_size}.png',dpi=600)
#plt.savefig(f'plot/activation_comp_{batch_size}_shullfle.png',dpi=600) #uncomment if shuffle is on for GD
plt.show()

#plot first values of ReLU to see if it converges.
plt.semilogy(iter[:1000:10],cost_r[:1000:10],color=palette['blue'])
plt.xlabel('Iterations')
plt.ylabel('Trainging loss')
plt.savefig(f'plot/relu_{batch_size}.png',dpi=600)
plt.show()

loss_sigmoid = [1.5656e-02, 1.9701e-03, 1.7423e-03, 1.6978e-03, 1.6978e-03]
loss_lrelu =   [1.4192e-04, 1.3736e-03, 2.6709e-03, 3.3812e-04, 4.3281e-03]
loss_silu =    [5.4287e-03, 2.4922e-03, 4.6856e-03, 6.2068e-03, 6.0725e-03]

losses = {'Sigmoid':loss_sigmoid, 'Leaky ReLU':loss_lrelu, 'Silu':loss_silu}
batch_sizes = ['1', '10', '25', '50', '50*']
colors = ['#3E31D6','#2EC483','#DD0000']
list = [1,2,3,4,5]

fig, ax = plt.subplots()
for color, act in zip(colors,losses):
    ax.semilogy(list, losses[act], marker='s', label=act, color=color)
plt.legend()
plt.xlabel('Batch size')
plt.ylabel('Final loss')
plt.xticks(list, ['1', '10', '25', '50', '50*'])
plt.savefig('plot/change_of_loss.png',dpi=600)
plt.show()

"""
Results:
batch_size=1:
|Sigmoid   | loss: 1.5656e-02| it/s: 2483.0 |
|ReLU      | loss: 3.3333e-01| it/s: 2449.56|
|Leaky ReLU| loss: 1.4192e-04| it/s: 2461.65|
|Silu      | loss: 5.4287e-03| it/s: 2462.45|
---------------------------------------------
batch_size=10:
|Sigmoid   | loss: 1.9701e-03| it/s: 2399.24|
|ReLU      | loss: 3.3333e-01| it/s: 2368.74|
|Leaky ReLU| loss: 1.3736e-03| it/s: 2389.67|
|Silu      | loss: 2.4922e-03| it/s: 2384.06|
---------------------------------------------
batch_size=25:
|Sigmoid   | loss: 1.7423e-03| it/s: 2222.27|
|ReLU      | loss: 3.3333e-01| it/s: 2195.52|
|Leaky ReLU| loss: 2.6709e-03| it/s: 2206.54|
|Silu      | loss: 4.6856e-03| it/s: 2218.97|
---------------------------------------------
batch_size=50:
|Sigmoid   | loss: 1.6978e-03| it/s: 2725.19|
|ReLU      | loss: 2.2551e-01| it/s: 2684.94|
|Leaky ReLU| loss: 3.3812e-04| it/s: 2694.28|
|Silu      | loss: 6.2068e-03| it/s: 2692.92|
---------------------------------------------
batch_size=50 (with shuffling):
|Sigmoid   | loss: 1.6978e-03| it/s: 2396.67|
|ReLU      | loss: 3.3333e-01| it/s: 2373.06|
|Leaky ReLU| loss: 4.3281e-03| it/s: 2371.32|
|Silu      | loss: 6.0725e-03| it/s: 2370.41|


ReLU outputs(batch_size=25):
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
                        ⋮
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [5.7220e-06, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]

ReLU outputs(batch_size=50):
        [1.0029e+00, 0.0000e+00, 0.0000e+00],
                        ⋮
        [1.0086e+00, 0.0000e+00, 0.0000e+00],
        [8.2181e-01, 0.0000e+00, 0.0000e+00],
        [1.0035e-03, 0.0000e+00, 0.0000e+00],
        [5.0893e-01, 0.0000e+00, 0.0000e+00],
        [1.3393e-01, 0.0000e+00, 0.0000e+00],
                        ⋮
        [8.1268e-01, 0.0000e+00, 0.0000e+00],
                        ⋮
        [9.8230e-01, 0.0000e+00, 0.0000e+00],
        [6.7686e-01, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [3.3592e-01, 0.0000e+00, 0.0000e+00],
        [8.8342e-01, 0.0000e+00, 0.0000e+00],
                        ⋮
        [1.1315e+00, 0.0000e+00, 0.0000e+00],
        [2.2740e-02, 0.0000e+00, 0.0000e+00],
        [7.3965e-03, 0.0000e+00, 0.0000e+00],
                        ⋮
        [8.8376e-01, 0.0000e+00, 0.0000e+00],
        [1.9363e-02, 0.0000e+00, 0.0000e+00],
        [9.0734e-01, 0.0000e+00, 0.0000e+00],
                        ⋮
        [1.0637e+00, 0.0000e+00, 0.0000e+00],
                        ⋮
        [9.7859e-01, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [2.6464e-01, 0.0000e+00, 0.0000e+00],
        [1.0182e+00, 0.0000e+00, 0.0000e+00],
        [4.2805e-02, 0.0000e+00, 0.0000e+00],
                        ⋮
        [9.5071e-01, 0.0000e+00, 0.0000e+00],
                        ⋮
        [1.3087e+00, 0.0000e+00, 0.0000e+00],
        [1.2323e+00, 0.0000e+00, 0.0000e+00]

        For brevity, ⋮ means [0.0000e+00, 0.0000e+00, 0.0000e+00] is repeated multiple times
"""
