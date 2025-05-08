"""
Code for implementation of small network used in section 7.1
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
from tqdm import tqdm #progress bar

#colour palette:
blue = '#3E31D6'; orange = '#FEA11B'

rand.seed(2025) #for reproducibility
N = 10 #number of samples of samples

x_data=np.array([[0.4,0.2],[0.6,0.4],[0.7,0.2],[0.9,0.6],[0.8,0.9],
                 [0.1,0.5],[0.2,0.1],[0.2,0.8],[0.3,0.4],[0.5,0.7]])
y_data = np.zeros((N,2))
#first half is blue other half is orange
y_data[:5] = [1,0]; y_data[5:] = [0,1]

#Plot data
plt.scatter(*zip(*x_data[:5]), marker ='x', color=blue)
plt.scatter(*zip(*x_data[5:]), marker ='o', color=orange)
plt.savefig('plot/simple_example.png',dpi=600)
plt.show()

def cost_eval(W2,W3,b2,b3):
    """Compute empirical risk for parameters W2,b2,W3,b3"""
    loss = np.zeros(N)
    for i in range(N):
        x = x_data[i]; y = y_data[i]
        a2 = s(W2@x + b2)
        a3 = s(W3@a2 + b3)
        loss[i] = np.mean((a3-y)**2) #mean square error
    return np.mean(loss) #quadratic cost

#define activation function and its derivative
s = lambda x: 1/(1+np.exp(-x)) #sigmoid activation function
ds = lambda x: s(x)*(1-s(x)) #derivative of sigmoid

#initialize parameters according to Xavier initialization
W2 = rand.uniform(-np.sqrt(6/(2+4)),np.sqrt(6/(2+4)),(4,2)); b2 = [0,0,0,0]
W3 = rand.uniform(-np.sqrt(6/(4+2)),np.sqrt(6/(4+2)),(2,4)); b3 = [0,0]

eta = 0.05 #learning rate
Niter = int(1e5) #number of iterations
train_loss = np.zeros(Niter) #training loss throughout training iterations

print('Start training.')
for k in tqdm(range(Niter)): 
    i = rand.randint(N) #pick training point uniformly at random
    x = x_data[i]; y = y_data[i]

    #forward pass
    a2 = s(W2@x + b2)
    a3 = s(W3@a2 + b3)

    #backward pass
    delta3 = ds(a3)*(a3-y)
    delta2 = ds(a2)*(W3.T@delta3)

    #Compute parameter updates
    W2 = W2 - eta*np.outer(delta2,x)
    W3 = W3 - eta*np.outer(delta3,a2)
    b2 = b2 - eta*delta2
    b3 = b3 - eta*delta3

    #evaluate loss
    loss = cost_eval(W2,W3,b2,b3)
    train_loss[k] = loss

print('Done!')
print(f'Final train loss: {train_loss[-1]:.4e}') #print final result

#Plot training loss
iter = np.arange(Niter)
fig, ax =plt.subplots()
ax.semilogy(iter,train_loss, color=blue, alpha=0.35)
ax.semilogy(iter[::1000],train_loss[::1000], color=blue) #plot every 1000th element for readability
plt.savefig('plot/example_nn_trainloss.png',dpi=600)
plt.show()

"""
Result: 
Final train loss: 1.4292e-03
"""
