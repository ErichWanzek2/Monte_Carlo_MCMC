"""MARKOV CHAIN MONTE CARLO"""
"""ERICH WANZEK"""
################################################################################
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import random
import corner
################################################################################
################################################################################
"""Functions and plots"""
################################################################################
def linear(x,a,b):
    """This function is a linear funciton"""
    return a*x+b
#-------------------------------------------------------------------------------
def generate_linear(a,b,N):
    """This function generates the linear function"""
    x=np.linspace(0,10,N)
    y=[]
    for i in range(len(x)):
        y.append(linear(x[i],a,b))
    return x,y
#-------------------------------------------------------------------------------
def sine(x,a,b,c):
    """This is a sinuisoidal funciton"""
    return a*np.sin(b*x)+c*x
#-------------------------------------------------------------------------------
def generate_sine(a,b,c,N):
    """This funciton generates the sinusoidal funciton"""
    x=np.linspace(0,20,N)
    y=[]
    for i in range(len(x)):
        y.append(sine(x[i],a,b,c))
    return x,y
#-------------------------------------------------------------------------------
def random_noise_generator(level,data):
    """this funciton generates and adds random noise to data
       ar selcted noise level"""
    for i in range(len(data)):
        data[i]=data[i] + level*random.uniform(-1,1) 
    return data
#-------------------------------------------------------------------------------
def plot(x,y):
    """Plots Y vs X lists"""
    plt.plot(x,y,'r+')
    plt.xlabel('x')
    plt.ylabel('y')
##    x2,y2=generate_linear(2.02,3.18,100)
##    plt.plot(x2,y2)
    plt.ylim(0,30)                        
    plt.show()
#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#%%##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%##%#%
def DATA(a,b,c,model,N,level):
    """this function generates synthetic data"""
    if model=='linear':
       x,y=generate_linear(a,b,N)
       y=random_noise_generator(level,y)
       plot(x,y)
       return y
       
    if model =='sine':
       x,y=generate_sine(a,b,c,N)
       y=random_noise_generator(level,y)
       plot(x,y)
       return y
################################################################################   
################################################################################
"""MCMC ALGRITHM SECTION"""
################################################################################
def perturb(params):
    """This function perturbs the parametes of the markov chain"""
    perturbed_params=[]
    stepsigma=[.01,.01,.01]      #.1 standard  # .01 does the trick for sine
    for i in range(len(params)):
        perturbed_params.append(random.gauss(params[i], stepsigma[i]))
    return perturbed_params
#-------------------------------------------------------------------------------
def GM(params,model):
    """This fucntion genrates list data poits of selected model"""
    N=20      #or 20
    if model == 'linear':
       a=params[0]
       b=params[1]
       x,y=generate_linear(a,b,N)  #linear model
       return y
    if model =='sine':
       a=params[0]
       b=params[1]
       c=params[2] 
       x,y=generate_sine(a,b,c,N)  #sine model
       return y
#-------------------------------------------------------------------------------
def LH(data,g_model,sigma):
    """This function calculates the liklihood of each parameterized model"""
    l1=0
    l2=0
    for i in range(len(g_model)):
        l1=((np.log(np.pi*(sigma[i])**2)))+l1

    for i in range(len(g_model)):           
        l2=(((data[i]-g_model[i])/(sigma[i]))**2)+l2
    L=-l1-(1/2)*l2
    return L
#-------------------------------------------------------------------------------    
def MH(L_old,L_new,old_params,new_params):
    """This function performs the Metropolis Hastings algorithm"""
    if L_new > L_old:
       p=new_params
    else:
        if (random.random()) < np.exp(L_new-L_old):
           p=new_params
        else:
            p=old_params
    return p
#%#%#%#%%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
def MCMC(data,model,sigma,I):
    """This fucniton performs the MCMC algorithm"""
    if model=="linear":
       params=[0,0]
    if model=='sine':
       params=[4,1,-.3]
    L=[LH(data,GM(params,model),sigma)]
    chain=[params]
    for i in range(1,I):
        new_params=perturb(chain[i-1])
        L.append(LH(data,GM(new_params,model),sigma))
        chain.append(MH(L[i-1],L[i],chain[i-1],new_params))   
    return chain
################################################################################
################################################################################
"""DATA"""
x=[1.0815,2.0906,3.0127,4.0913,5.0632,6.0098,7.0278,8.0547,9.0958,10.0965,11.0158,
    12.0971,13.0957,14.0485,15.0800,16.0142,17.0422,18.0916,19.0792,20.0959]
xlength=len(x)
print(xlength)
y=[3.0951,3.4227,1.0783,-3.0820,-5.5794,-5.3297,-2.5325,0.6026,1.4255,-0.8572,
 -4.4055,-7.5680,-7.5651,-5.0845,-1.8224,-0.6096,-2.3531,-6.2875,-9.4180,-9.8996]
ylength=len(y)
print(ylength)


a=3.9
b=0.9
c=-0.31
N=200
x2,y2=generate_sine(a,b,c,N)
plt.plot(x,y,'r+')
plt.plot(x2,y2,'k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

################################################################################
################################################################################
def MASTER(a,b,c,model,N,level, I):
    """This is the master fucntion
       a, b, c, parameters
       model is the model type: 'linear' or 'sine'
       N is the number of data point for synthetic data generation
       level is the noise level to be injected into the synthetic data
       I is the number of MCMC iterations to be performed
    """
    sigma=[]    
    for i in range(0,1000000):
        sigma.append(1)  #0.0001

##    data=DATA(a,b,c,model,N,level)
    data=y
    MarkovChain=MCMC(data,model,sigma,I)

    MarkovChain=MarkovChain[30000:] ## CUTOFF BURN IN iterations 
    
    if model == 'linear':
       fig = corner.corner(MarkovChain, labels=[r"$m$", r"$b$"],
                       quantiles=[0.16,0.5,0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})
       fig.savefig("triangle.png")

    if model=='sine':
       fig = corner.corner(MarkovChain, labels=[r"$A$", r"$B$", r"$C$"],
                       quantiles=[0.16,0.5,0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})
       fig.savefig("triangle.png") 

    return 
################################################################################    
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
MASTER(2,3,2,'sine',100,4,1000000)   















