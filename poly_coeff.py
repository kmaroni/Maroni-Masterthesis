import numpy as np
import numpy.linalg as LA
from prettytable import PrettyTable

class Coeffs:
    def __init__(self,n,m,p):
        """
        n,m: natural numbers, n>=2
        p: list of functions
        """
        self.one = np.ones((n-1,1)) #vector of ones to be used later
        self.n = n
        self.m = m

        #compute beta coefficients
        x = [j/m for j in range(m+1)] #uniform mesh

        y = [(k-1)/(2*(n-1)) for k in range(1,n)] #y_k evaluated in M
        M = np.zeros((n-1,n-1)) #Matrix M from proof of thm 6.4
        for k, y_k in enumerate(y):
            for r in range(n-1):
                M[r,k]=p[r](y_k)

        a = (2*(n-1))/(m*n)
        coeff = []

        for j in range (1,m+1):
            b = [a**(2+r)*p[r](x[j-1]) for r in range(n-1)] #vector b from proof of thm 6.4
            b = np.array(b).T #make colomn vector
            beta = LA.solve(M,b) #[\beta_{j,k}] for k=1,...,n-1
            coeff.append(beta.flatten())

        #matrix of coeff [\beta_{j,k}]_{j=1,...,m, k=1,...,n-1}
        self.coeff = np.array(coeff)

    def K(self):
        """Return the first contraction constant K"""
        #\sum_j\sum_k|\beta_{i,k}|
        return np.sum(np.abs(self.coeff.flatten()))

    def KT(self):
        """Return the second contraction constant K tilde"""
        #2*max_{j=1,...,m} \sum_k |\beta_{j,k}]
        return 2*np.max(np.abs(self.coeff)@self.one)

    def KB(self):
        """Return the third contraction constant K bar"""
        #tridiagonal matrix with (0,1,-1) to subtract consecutive indices
        S = np.eye(self.n-1,self.m) + np.eye(self.n-1,self.m,k=1)*(-1)

        #\max_{j=1,...,m-1}\sum_k|\beta_{j+1,k}-\beta-{j,k}|
        return np.max(np.abs(S@self.coeff)@self.one)

    def print_c(self):
        """Print out matrix of coefficients"""
        print(self.coeff)

#numerical errors from tests in [1]
err1=[0.0377, 0.00806, 0.00238, 0.000521, 0.000165, 3.66e-5, 9.97e-6]
err2=[0.459,  0.179,   0.0834,  0.0409,   0.0109,   0.00471]
err3=[0.201,  0.027,   0.0057,  0.00103]
err4=[0.379,  0.0625,  0.0318,  0.00947]

tests = [err1, err2, err3, err4]
Kn = []
for i, test in enumerate(tests):
    avg = 0
    N = len(test)-1
    for k in range(N):
        avg += test[k+1]/(N*test[k])
    Kn.append(avg) #K_num is avg of epsilon_k/epsilon_{k-1}

#test1:
#H(x)=x-x^2
p1 = lambda x:-2
n = 2; m = 2
test1 = Coeffs(n,m,[p1])

#test2:
#H(x)=T_3(2x-1)=4(2x-1)^3-3(2x-1)
p2 = lambda x: 96*(2*x-1)
p2d = lambda x: 192
n = 3; m = 3
poly2 = [p2,p2d]

test2 = Coeffs(n,m,poly2)

#test3:
#same function
m = 5
test3 = Coeffs(n,m,poly2)

#test4:
#H(x)=T_5(2x-1)=16(2x-1)^5-20(2x-1)^3-5(2x-1)
n = 5; m = 9
p4 = lambda x: 4*(320*(2*x-1)**3-120*(2*x-1))
p4d = lambda x: 8*(960*(2*x-1)**2-120)
p4d2 = lambda x: 16*1920*(2*x-1)
p4d3 = lambda x: 32*1920
poly4 = [p4, p4d, p4d2, p4d3]

test4 = Coeffs(n,m,poly4)


#print as nice table
table = PrettyTable(['test number','K_num', 'K','KT','KB','(KT+KT)/2'])
table.add_row(['test #1',f'{Kn[0]:.4f}',f'{test1.K():.4f}',f'{test1.KT():.4f}',f'{test1.KB():.4f}', f'{0.5*(test1.KT()+test1.KB()):.4f}'])
table.add_row(['test #2',f'{Kn[1]:.4f}',f'{test2.K():.4f}',f'{test2.KT():.4f}',f'{test2.KB():.4f}', f'{0.5*(test2.KT()+test2.KB()):.4f}'])
table.add_row(['test #3',f'{Kn[2]:.4f}',f'{test3.K():.4f}',f'{test3.KT():.4f}',f'{test3.KB():.4f}', f'{0.5*(test3.KT()+test3.KB()):.4f}'])
table.add_row(['test #4',f'{Kn[3]:.4f}',f'{test4.K():.4f}',f'{test4.KT():.4f}',f'{test4.KB():.4f}', f'{0.5*(test4.KT()+test4.KB()):.4f}'])

print(table)

"""
Result:
+-------------+---------+---------+---------+---------+-----------+
| test number |  K_num  |    K    |    KT   |    KB   | (KT+KT)/2 |
+-------------+---------+---------+---------+---------+-----------+
|   test #1   | 0.25648 | 0.50000 | 0.50000 | 0.00000 |  0.25000  |
|   test #2   | 0.40898 | 1.14129 | 1.05350 | 0.52675 |  0.79012  |
|   test #3   | 0.17538 | 0.70163 | 0.45511 | 0.11378 |  0.28444  |
|   test #4   | 0.32384 | 3.10716 | 2.03060 | 0.87534 |  1.45297  |
+-------------+---------+---------+---------+---------+-----------+

sources:
[1] Després, B. ‘A convergent Deep Learning algorithm for approximation of
    polynomials’. In: Comptes Rendus. Mathématique vol. 361 (G6 7th Sept. 2023),
    pp. 1029–1040.
"""
