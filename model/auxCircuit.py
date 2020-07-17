import numpy as np

def rk4(fun(t,y), y0, params, timeStep, totalSteps):

  # Time loop
  for loopA in range(totalSteps):

    // Increment Time Step
    currTime = loopA*timeStep

    # Eval K1
    k1,k1AuxOut = evalDeriv(currTime,Xn,params,forcing)

    # Eval K2
    for loopB in range(totalStates):
      Xk2[loopB] = Xn[loopB] + ((1.0/3.0)*timeStep) * k1[loopB];
    
    k2,k2AuxOut = evalDeriv(currTime + (1.0/3.0) * timeStep,Xk2,params,forcing);

    // Eval K3
    for(int loopB=0;loopB<totalStates;loopB++){
      Xk3[loopB] = Xn[loopB] - (1.0/3.0)*timeStep * k1[loopB] + (1.0*timeStep) * k2[loopB];
    }
    ode->evalDeriv(currTime + (2.0/3.0) * timeStep,Xk3,params,forcing,k3,k3AuxOut,Ind);

    // Eval K4
    for(int loopB=0;loopB<totalStates;loopB++){
      Xk4[loopB] = Xn[loopB] + timeStep*k1[loopB] - timeStep*k2[loopB] + timeStep * k3[loopB];
    }
    ode->evalDeriv(currTime + timeStep,Xk4,params,forcing,k4,k4AuxOut,Ind);

    // Eval Xn1
    for(int loopB=0;loopB<totalStates;loopB++){
      if(Ind[loopB] > 0) {
         Xn1[loopB] = Xn[loopB] + (1.0/8.0)*timeStep*(k1[loopB] + 3.0 * k2[loopB] + 3.0 * k3[loopB] + k4[loopB]);
      }
      else {
         Xn1[loopB] = 0.0;
      }
    }

    // Update Xn
    for(int loopB=0;loopB<totalStates;loopB++){
      Xn[loopB] = Xn1[loopB];
    }

    // Update Current Time
    currTime += timeStep;

    // Copy Auxiliary outputs at every time step
    for(int loopB=0;loopB<totAuxStates;loopB++){
      auxOutVals[loopB][stepId] = k4AuxOut[loopB];
    }

    // Copy solution at each time step
    for(int loopB=0;loopB<totalStates;loopB++){
      outVals[loopB][stepId] = Xn1[loopB];
    }
  }

  // RETURN OK
  return 0;
}


# Define Constant
mmHgToBarye = 1333.22

def evalDerivRC(t,y,params)

  res = np.zeros(self.numState)
  aux = np.zeros(self.numAuxState)

  R  = params[0]
  C  = params[1]
  Pd = 55.0*mmHgToBarye
  P1 = y[0]

  # Interpolate forcing
  Q1    = np.interp(t % self.cycleTime, self.forcing[:,0], self.forcing[:,1])
  Q2    = (P1-Pd)/R
  dP1dt = (Q1-Q2)/C

  # Store the derivatives
  res[0] = dP1dt

  # Get Auxiliary Results  Question. Do I keep t as an auxOut variable?  If so, then I should change # auxVar to 3
  self.auxMat.append([t,Pd,Q1,Q2])

  return res

def evalDerivRCR(t,y,params)

  res = np.zeros(self.numState)
  aux = np.zeros(self.numAuxState)

  R1 = params[0]
  R2 = params[1]
  C  = params[2]
  Pd = 55.0*mmHgToBarye

  P1 = y[0]

  # Compute other variables
  Q1    = np.interp(t % self.cycleTime, self.forcing[:,0], self.forcing[:,1])
  P0    = P1 + R1*Q1;
  Q2    = (P1 - Pd)/ R2
  dP1dt = (Q1 - Q2) / C

  # Store the derivatives
  res[0] = dP1dt

  # Get Auxiliary Results  Question. Do I keep t as an auxOut variable?  If so, then I should change # auxVar to 3
  self.auxMat.append([t,Pd,P0,Q1,Q2])

  return res
