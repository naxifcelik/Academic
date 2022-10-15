import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

#\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
#                          Nazif ÇELİK 090200712                        #
#\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

def main():
    sampledims = [10 ,20, 100, 1000, 10000]
    for smp in sampledims:
        fit(smp)
if __name__ == '__main__':
    main()
    
    
def fit(sampledims):
    x = np.sort(np.random.random(sampledims))
    y = J0_(x)
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)
    Xtrain = Xtrain.reshape(-1,1)
    xtest = np.linspace(0, 1 , num = 100).reshape(-1, 1)
    polyreg = make_pipeline(PolynomialFeatures(10), Ridge(alpha = 0.001))
    polyreg.fit(Xtrain, ytrain)
    plot = polyreg.predict(xtest)
    
    sns.set_theme()
    plt.plot(x, y, label="ground truth", color = "blue")
    plt.scatter(Xtrain, ytrain, label="data", s = 10)   
    plt.plot(xtest, plot, label="fit", color = "yellow")
    plt.legend()
    plt.show()
    return

n=10
def fact(n):
   N=np.ones(n,dtype=float)
   N[0]=1
   N[1]=1
    
   for i in range(len(N)-2):
       N[i+2]=N[i+1]*(i+2)
   return(N)
result=fact(n)


def J0(x,fact):
   ct = 0
   for n in range(len(fact)):
     ct += (-1)**n * ((x/2)**(2*n))/(fact[n]**2)
   return ct

fact=factorial(n)
def J0_(x):
   return J0(x,fact)