from kan import *

#number of basis functions
m = 9; n = 5
r = m*(n-1)

#given polinomial:
H = lambda x: 16*(2*x-1)**5 - 20*(2*x-1)**3 + 5*(2*x-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#initiating model, creating dataset and training model:
model = KAN(width=[1,r+1,1], grid=m, k=1, seed=2024, device=device)
dataset = create_dataset(H, ranges=[0,1], n_var=1, train_num=4000, device=device)
model.fit(dataset, opt="LBFGS", steps=2)

"""
Results:
steps = 2  | train_loss: 7.61e-03 | test_loss: 7.82e-03 |
steps = 20 | train_loss: 1.45e-03 | test_loss: 1.47e-03 |
"""


