import numpy as np
import time
from c_routines import solveRK4_aux

# mmHg to Barye Conversion Factor
mmHgToBarye = 1333.22
baryeTommHg = 1.0/1333.22

def extractPatientFromDB(dbFile,columnID,resName,stds):
  '''
  Extract data for a single patient from the dataset
  Note: The first patient has patientID 0
  '''
  # Read DB File in Text
  data = np.loadtxt(dbFile,dtype=str,delimiter=',')
  # Extract Labels
  patLabels = data[:,0]
  # Extract the column for the patient 
  patVals = data[:,columnID+1]

  # Remove the components where the data is "none"
  mask = np.where(np.char.upper(patVals) == 'NONE')
  patLabels = np.delete(patLabels, mask)
  patVals = np.delete(patVals, mask)

  # Find the components of the standard deviations
  patStd = []
  for loopA in range(len(patLabels)):
    # Find the index in resName
    idx = np.where(resName == patLabels[loopA])
    if(len(idx[0]) == 0):
      print('WARNING: Key ' + patLabels[loopA] + ' not found as a model result.')
    # Store the corresponding standard deviation
    patStd.append(stds[idx[0][0]])

  # return the data you found and the associated labels
  return patLabels,patVals.astype(np.float),np.array(patStd).astype(np.float)
      
def extractModelMatch(out,outLabels,data,dataLabels,dataStd):
  '''
  Match the measurements from the dataset with the model output
  '''
  label       = []
  modOut      = []
  measurement = []
  stds        = []
  for loopA in range(len(outLabels)):
    # Get Index in dataLabels
    idx = np.where(dataLabels == outLabels[loopA])[0]
    if(len(idx) > 0):
      modOut.append(out[loopA])
      measurement.append(data[idx[0]])
      stds.append(dataStd[idx[0]])
      label.append(dataLabels[idx[0]])

  return np.array(label).reshape(-1,1),np.array(modOut).reshape(-1,1),np.array(measurement).reshape(-1,1),np.array(stds).reshape(-1,1)

class circuitModel():

  def __init__(self,numParam,numState,numAuxState,numOutputs,
               icName,parName,resName,limits,
               defIC,defParam,
               cycleTime,totalCycles,timeStepsPerCycle=1000,forcing=None,
               debugMode=False,verbose=False):
    # Time integration parameters
    self.timeStepsPerCycle = timeStepsPerCycle
    self.cycleTime = cycleTime
    self.totalCycles = totalCycles
    # Forcing
    self.forcing = forcing
    # Init parameters
    self.numParam    = numParam
    self.numState    = numState
    self.numAuxState = numAuxState
    self.numOutputs  = numOutputs
    self.icName      = icName
    self.parName     = parName
    self.resName     = resName
    self.limits      = limits
    self.defIC       = defIC
    self.defParam    = defParam
    # Debug Mode
    self.debugMode = debugMode
    self.verbose   = verbose

  def eval_IC(self,params):
    return np.zeros(self.numState)

  def evalDeriv(self,t,y,params):
    pass

  def postProcess(self,t,y,aux,start,stop):
    pass

  def plot_model(self,t,y,aux,start,stop):
    pass

  def genDataFile(self,dataSize,stdRatio,dataFileName):
    data = np.zeros((self.numOutputs,dataSize))
    # Get Standard Deviaitons using ratios
    stds = self.solve(self.defParam)*stdRatio
    for loopA in range(dataSize):
      # Get Default Paramters
      data[:,loopA] = self.solve(self.defParam) + np.random.randn(len(stds))*stds
    np.savetxt(dataFileName,data)

  def areValidParams(self,params):
    # Check size
    if(len(params) != len(self.limits)):
      print('ERROR: Parameters and Limits are not compatible.')
      print('Parameter Size: ',len(params))
      print('Limit Size: ',len(self.limits))
      exit(-1)

    res = True
    for loopA in range(len(params)):
      this_res = ((params[loopA] >= self.limits[loopA,0]) and (params[loopA] <= self.limits[loopA,1]))
      res = res and this_res
      if(not this_res):
        print('Component: {}, Parameter: {}, lower bound: {}, upper bound: {}'.format(loopA,params[loopA],self.limits[loopA,0],self.limits[loopA,1]))

    return res

  def solveRK4(self,timeStep,totalSteps,saveEvery,y0,params):
    return solveRK4_aux(lambda t,y: self.evalDeriv(t,y,params),timeStep,totalSteps,saveEvery,y0,params,self.numState,self.numAuxState)

  def solve(self,params=None,y0=None):
    
    # Set Initial guess equal to the default parameter values
    self.params = params
    if(self.params is None):
      self.params = self.defParam    

    # Homogeneous initial conditions with None
    self.y0 = y0
    if(self.y0 is None):
      self.y0 = self.eval_IC(params)

    # Set initial and total time
    totalSteps = self.timeStepsPerCycle*self.totalCycles
    # 200 total steps are saved
    savePerCycle = 100
    # Compute Time Step
    self.timeStep = self.cycleTime/self.timeStepsPerCycle
    # Save Results Every saveEvery setps
    saveEvery = int(self.timeStepsPerCycle/savePerCycle)

    # Integrate System
    time1     = time.time()
    t,y,aux = self.solveRK4(self.timeStep, totalSteps, saveEvery, self.y0, self.params)
    solveTime = (time.time()-time1)*1000 # Microseconds

    # Post Process - Start and Stop should be computed with respect to the saved steps
    start = (self.totalCycles-1)*savePerCycle
    stop  = self.totalCycles*savePerCycle

    # Return Post-processed data
    time1     = time.time()
    modOuts = self.postProcess(t,y,aux,start,stop)
    postTime = (time.time()-time1)*1000 # Microseconds

    # Plot model response if in debug mode
    if(self.debugMode):
      self.plot_model(t,y,aux,start,stop)
    
    return modOuts,solveTime,postTime

  def evalNegLL(self,columnID,dbFile,stds,params=None,y0=None):
    '''
    This matches the row names in the db file with the name of the
    results from the model
    '''
    # Get user or default parameters
    currParams = params
    if(params is None):
      currParams = self.defParam[self.numState:]
    # Assign initial conditions
    currIni = y0
    if(currIni is None):
      currIni = self.eval_IC(params)

    # Extract the label and data from the DB file
    time1                    = time.time()
    patLabels,patData,patStd = extractPatientFromDB(dbFile,columnID,self.resName,stds)
    extractBDTime            = (time.time()-time1)*1000 # Microseconds

    if(self.verbose):
      print("%30s %15s %15s" % ("Label","Data","StD"))
      for loopA in range(len(patLabels)):
        print("%30s %15.1f %15.1f" % (patLabels[loopA],patData[loopA],patStd[loopA]))

    # Are parameters within the limits
    if not(self.areValidParams(currParams)):
      print('WARNING: Parameters out of bounds.')
      
    # Get Model Solution
    modelOut,solveTime,postTime  = self.solve(currParams,currIni)
    
    # Match the measurements from the dataset with the model output
    time1                            = time.time()
    keys,modOut,measurement,stds_out = extractModelMatch(modelOut,self.resName,patData,patLabels,patStd)
    matchOutTime                     = (time.time()-time1)*1000 # Microseconds
    
    if(self.verbose):
      print("")
      print("%30s %15s %15s %15s" % ("Key","Output","Data","Std"))
      for loopA in range(len(modOut)):       
        print("%30s %15.1f %15.1f %15.1f" % (keys[loopA,0],modOut[loopA,0],measurement[loopA,0],stds_out[loopA,0]))

    # Eval LL
    ll1 = -0.5*np.prod(measurement.shape)*np.log(2.0*np.pi)
    ll2 = -0.5*measurement.shape[1]*np.log(np.prod(stds_out))
    ll3 = -0.5*((modOut.reshape(-1,1)-measurement)**2/(stds_out.reshape(-1,1)**2)).sum()
    negLL = -(ll1 + ll2 + ll3)

    if(self.verbose and self.debugMode):
      print("--- Timing")
      print("Extract Patient From DB. Time: %f ms" % (extractBDTime))
      print("Model Solution. Time: %f ms" % (solveTime))
      print("Model Post-processing. Time: %f ms" % (postTime))
      print("Matching Model Ouputs. Time: %f ms" % (matchOutTime))
      print("---")

    return negLL,modOut,measurement,stds_out,keys

  def eval_negll_grad(self,delta,columnID,dbFile,stds,params=None,y0=None):
    """
    Evaluating the gradient of the negative ll.
    delta is the percent change in the inputs, so delta=1 corrisponds to 1% change
    """
    # Create parameter table
    param_table = np.repeat(params.reshape(-1,1),len(params)+1,axis=1)

    incr = np.zeros(len(params))
    # Loop through the parameters and add increments
    for loopA in range(len(params)):
      incr[loopA] = (delta/100.0)*param_table[loopA,loopA+1]
      param_table[loopA,loopA+1] += incr[loopA]
    
    # Compute model outputs
    res = np.zeros(len(params)+1)

    # Solve models
    for loopA in range(len(params)+1):
      negLL,modOut,measurement,stds_out,keys = self.evalNegLL(columnID,dbFile,stds,np.ascontiguousarray(param_table[:,loopA]),y0)
      res[loopA] = negLL

    # Compute gradient
    grad = np.zeros(len(params))
    for loopA in range(len(params)):
      grad[loopA] = (res[loopA+1] - res[0])/incr[loopA]

    return grad
