from kan import *

#number of basis functions
#m = 3; n = 3
m = 5; n = 3
r = m*(n-1)

#given polinomial:
#H = lambda x: 32*(2*x-1)**3 - 48*(2*x-1)**2 + 18*x - 1
H = lambda x: 32*x**3 - 48*x**2 + 18*x - 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#initiating model, creating dataset and training model:
model = KAN(
    width=[1,r+1,1],
    #grid = 2*m,
    grid = m
    k=1,
    seed=2024,
    device=device)
dataset = create_dataset(H, ranges=[0,1], n_var=1, train_num=4000, device=device)
model.fit(dataset, opt="LBFGS", steps=100)

"""
Results:
m = 3: | train_loss: 4.52e-03 | test_loss: 4.39e-03 |
m = 5: | train_loss: 1.70e-03 | test_loss: 1.75e-03 |
m = 5 (grid = 10): | train_loss: 7.34e-04 | test_loss: 7.95e-04 |
"""
