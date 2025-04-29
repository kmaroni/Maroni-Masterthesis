from kan import *
import numpy as np

#number of basis functions
m = 2; n = 2
r = m*(n-1)

H = lambda x: x-x**2 #given polynomial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#initiating model, creating dataset and training model:
model = KAN(width=[1,r+1,1], grid=m, k=1, seed=2024, device=device)
dataset = create_dataset(H,ranges=[0,1], n_var=1, train_num=4000, device=device)
model.fit(dataset, opt="LBFGS", steps=20)
#model.plot()
#plt.show()

#fixing model to symbolic formula:
poly = ['x','x^2','x^3']
model.auto_symbolic(lib=poly)

model.fit(dataset, opt="LBFGS", steps=10)

function = ex_round(model.symbolic_formula()[0][0],4)
print(ex_round(function,4))


#plot
blue = '#3E31D6'; orange = '#FEA11B'

x = np.linspace(0,1)
f = lambda x_1 :0.5427*x_1 - 1.0*(0.2287 - x_1)**2 + 0.0523 #result of auto symbolic
plt.plot(x,H(x),color=blue,label='Given polynomial')
plt.plot(x,f(x),linestyle='--',color=orange, label='Approx. polynomial')
plt.legend()
plt.savefig('plot/kan_poly.png',dpi=600)
plt.show()

"""
Results:
model:    | train_loss: 4.77e-04 | test_loss: 4.56e-04 |
symbolic: | train_loss: 2.06e-07 | test_loss: 1.95e-07 |
"""

