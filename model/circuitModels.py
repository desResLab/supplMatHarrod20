import sys
import numpy as np
import scipy as sp
# print('--- Numpy Version: ',np.__version__)
# print('--- Scipy Version: ',sp.__version__)
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define Constant
mmHgToBarye = 1333.22

class circuitModel():

  def __init__(self,numParam,numState,numAuxState,numOutputs,
               parName,limits,defParam,
               cycleTime,totalCycles,forcing=None):
    # Time integration parameters
    self.cycleTime = cycleTime
    self.totalCycles = totalCycles
    # Forcing
    self.forcing = forcing
    # Init parameters
    self.numParam    = numParam
    self.numState    = numState
    self.numAuxState = numAuxState
    self.numOutputs  = numOutputs
    self.parName     = parName
    self.limits      = limits
    self.defParam    = defParam
    self.auxMat      = []

  def evalDeriv(self,t,y,params):
    pass

  def postProcess(self,odeSol,start,stop):
    pass

  def genDataFile(self,dataSize,stdRatio,dataFileName):
    data = np.zeros((self.numOutputs,dataSize))
    # Get Standard Deviaitons using ratios
    stds = self.solve(self.defParam)*stdRatio
    for loopA in range(dataSize):
      # Get Default Paramters
      data[:,loopA] = self.solve(self.defParam) + np.random.randn(len(stds))*stds
    np.savetxt(dataFileName,data)

  def solve(self,params=None,y0=None):
    # Set Initial guess equal to the default parameter values
    self.params = params
    if(self.params is None):
      self.params = self.defParam
    # Homogeneous initial conditions with None
    self.y0 = y0
    if(self.y0 is None):
      self.y0 = np.zeros(self.numState)
    # Set initial and total time
    t0 = 0.0
    t_bound = self.totalCycles*self.cycleTime
    saveSteps = np.linspace(t0,t_bound,200,endpoint=True)
    # Perform the RK4 iterations
    # print(self.evalDeriv(0.0,np.array([0.0]),self.params))
    self.auxMat.clear()
    odeSol = solve_ivp(lambda t, y: self.evalDeriv(t,y,self.params), (t0,t_bound), self.y0, max_step=2.0e-3, t_eval=saveSteps)

    # Post Process
    start = len(saveSteps) - len(saveSteps[saveSteps > (self.totalCycles-1)*self.cycleTime])
    stop  = len(saveSteps)
    # Convert to numpy Array
    aux = np.array(self.auxMat)

    if(False):
      plt.subplot(3,1,1)
      plt.plot(self.forcing[:,0],self.forcing[:,1],'bo-')      
      plt.title('Forcing')
      plt.subplot(3,1,2)
      plt.plot(odeSol.t[start:stop],odeSol.y[0,start:stop]/mmHgToBarye,'r-',label='P1')
      plt.plot(aux[:,0],aux[:,1]/mmHgToBarye,'b-',label='Pd')      
      plt.legend()
      plt.subplot(3,1,3)
      plt.plot(aux[:,0],aux[:,2],'b-',label='Q1')
      plt.plot(aux[:,0],aux[:,3],'r-',label='Q2')
      plt.legend()
      plt.tight_layout()
      plt.show()

    return self.postProcess(odeSol,start,stop)

  def evalNegLL(self,data,stdRatio,params=None,y0=None):
    '''
    Important: the columns of the data file must be in the same 
    order as the outputs from the model
    '''
    # Get user or default parameters
    currParams = params
    if(params is None):
      currParams = self.defParam      
    # Assign initial conditions
    currIni = y0
    if(currIni is None):
      currIni = np.zeros(self.numState)
    # Get the absolute values of the standard deviations
    stds = self.solve(self.defParam,currIni)*stdRatio
    # Get Model Solution
    modelOut = self.solve(currParams,currIni)
    print(currParams)
    # Eval LL
    ll1 = -0.5*np.prod(data.shape)*np.log(2.0*np.pi)
    ll2 = -0.5*data.shape[1]*np.log(np.prod(stds))
    ll3 = -0.5*((modelOut.reshape(-1,1)-data)**2/(stds.reshape(-1,1)**2)).sum()
    negLL = -(ll1 + ll2 + ll3)
    return negLL

class rcModel(circuitModel):

  def __init__(self,cycleTime,totalCycles,forcing=None):    
    # Init parameters
    numParam    = 2
    numState    = 1
    numAuxState = 4
    numOutputs  = 3
    parName = ["R","C"]
    limits = np.array([[100.0, 1500.0],
                       [1.0e-5, 1.0e-2]])
    defParam = np.array([1000.0,0.00005])
    #  Invoke Superclass Constructor
    super().__init__(numParam,numState,numAuxState,numOutputs,
                     parName,limits,defParam,
                     cycleTime,totalCycles,forcing)

  def evalDeriv(self,t,y,params):
    # Use the fast C-compiled alternative
    return evalDerivRC(t,y,params)

  def postProcess(self,odeSol,start,stop):

    res = np.zeros(self.numOutputs)
    # Compute Min Pressure
    res[0] = np.min(odeSol.y[0,start:stop])/mmHgToBarye
    # Compute Max Pressure
    res[1] = np.max(odeSol.y[0,start:stop])/mmHgToBarye
    # Compute Average Pressure
    res[2] = (np.trapz(odeSol.y[0,start:stop],odeSol.t[start:stop])/float(self.cycleTime))/mmHgToBarye
    return res

class rcrModel(circuitModel):

  def __init__(self,cycleTime,totalCycles,forcing=None):    
    # Init parameters
    numParam    = 3
    numState    = 1
    numAuxState = 4
    numOutputs  = 3
    parName = ["R1","R2","C"]
    limits = np.array([[100.0, 1500.0],
                       [100.0, 1500.0],
                       [1.0e-5, 1.0e-2]])
    defParam = np.array([1000.0,1000.0,0.00005])
    #  Invoke Superclass Constructor
    super().__init__(numParam,numState,numAuxState,numOutputs,
                     parName,limits,defParam,
                     cycleTime,totalCycles,forcing)


  def evalDeriv(self,t,y,params):
    return evalDerivRCR(t,y,params)

  def postProcess(self,odeSol,start,stop):

    res = np.zeros(self.numOutputs)
    # Compute Min Pressure
    res[0] = np.min(odeSol.y[0,start:stop])/mmHgToBarye
    # Compute Max Pressure
    res[1] = np.max(odeSol.y[0,start:stop])/mmHgToBarye
    # Compute Average Pressure
    res[2] = (np.trapz(odeSol.y[0,start:stop],odeSol.t[start:stop])/float(self.cycleTime))/mmHgToBarye
    return res

def basicTests(modelType):

  # Create the model
  cycleTime = 1.07
  totalCycles = 10
  forcing = np.loadtxt('../assets/inlet.flow')
  if(modelType == 0):
    rc = rcModel(cycleTime,totalCycles,forcing)
  else:
    rc = rcrModel(cycleTime,totalCycles,forcing)
  y0 = 55.0*mmHgToBarye*np.ones(rc.numState)
  outs = rc.solve(y0=y0)
  print('Model outputs',outs)

  # Solve the model with default parameters
  if(True):
    # Generate the Data
    dataSize = 20 # number of observations
    stdRatio = 0.01 # 1% standard deviation in error
    rc.genDataFile(dataSize,stdRatio,'../assets/data.txt')

    # Evaluate the model log-likelihood
    data = np.loadtxt('../assets/data.txt')

    params = np.array([100.0,100.0,0.005])
    ll = rc.evalNegLL(data,stdRatio,params,y0)
    print('Model log-likelihood: ',ll)

    params = rc.defParam
    ll = rc.evalNegLL(data,stdRatio,params,y0)
    print('Model log-likelihood: ',ll)

def optLLTest(modelType):

  # Create the model
  cycleTime   = 1.07
  totalCycles = 10
  stdRatio    = 0.01 # 1% standard deviation in error
  forcing     = np.loadtxt('../assets/inlet.flow')
  if(modelType == 0):
    rc = rcModel(cycleTime,totalCycles,forcing)
  else:
    rc = rcrModel(cycleTime,totalCycles,forcing)
  
  # Get data from assets
  data = np.loadtxt('../assets/data.txt')

  # Set Initial Parameter Guess
  x0 = np.zeros(rc.numParam)
  if(modelType == 0):
    x0[0] = 20.0
    x0[1] = 0.05
  else:
    x0[0] = 20.0
    x0[1] = 20.0
    x0[2] = 0.05

  # Perform optimization
  bounds = tuple(map(tuple, rc.limits))
  for loopA in range(1):
    print('Nelder-Mead Restart Number: ',loopA)
    optSol = minimize(lambda x: rc.evalNegLL(data,stdRatio,x), x0, method='SLSQP', # method='Nelder-Mead', 
                      tol=1.0e-8, bounds=bounds, options={'maxiter': 100, 'disp': True})
    # Update initial guess in restarted Nelder Mean iterations
    x0 = optSol.x

  print(optSol.x)

def surfLL(modelType):
  # Init Model
  cycleTime   = 1.07
  totalCycles = 10
  stdRatio    = 0.01 # 1% standard deviation in error
  forcing     = np.loadtxt('../assets/inlet.flow')
  if(modelType == 0):
    rc = rcModel(cycleTime,totalCycles,forcing)
  else:
    rc = rcrModel(cycleTime,totalCycles,forcing)
  y0          = 55.0*mmHgToBarye*np.ones(rc.numState)
  # Get data from assets
  data = np.loadtxt('../assets/data.txt')

  # Construct Grid
  xVals = np.linspace(rc.limits[0,0],rc.limits[0,1],10)
  yVals = np.linspace(rc.limits[1,0],rc.limits[1,1],10)
  xx, yy = np.meshgrid(xVals, yVals)
  zz = np.zeros(xx.shape)

  # Compute LL at all grid points
  for loopA in range(xx.shape[0]):
    print('%d/%d' % (loopA,xx.shape[0]))
    for loopB in range(xx.shape[1]):
      print('%d/%d' % (loopB,xx.shape[1]))
      zz[loopA,loopB] = rc.evalNegLL(data,stdRatio,np.array([xx[loopA,loopB],yy[loopA,loopB]]),y0)

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot_trisurf(xx.flatten(), yy.flatten(), zz.flatten(), linewidth=0.2, antialiased=True)
  plt.show()

# TEST MODELS
if __name__ == "__main__":
  # modelType = 0
  modelType = 1

  # Perform basic test on the model
  basicTests(modelType)

  # Optimize Model
  optLLTest(modelType)

  # Plot response surface
  surfLL(modelType)
