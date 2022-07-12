import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from circuitModels import circuitModel,mmHgToBarye,baryeTommHg
from c_routines import evalDeriv_cvsim6, eval_cvsim6_ic

# STATE VARIABLES
i_p_l  = 0
i_p_a  = 1
i_p_v  = 2
i_p_r  = 3
i_p_pa = 4
i_p_pv = 5
# AUX VARIABLES
i_q_li = 0
i_q_lo = 1
i_q_a  = 2
i_q_ri = 3
i_q_ro = 4
i_q_pv = 5
i_v_l  = 6
i_v_a  = 7
i_v_v  = 8
i_v_r  = 9
i_v_pa = 10
i_v_pv = 11

# MODEL OUTPUTS
i_post_hr             = 0
i_post_sbp            = 1
i_post_dbp            = 2 
i_post_co             = 3
i_post_svr            = 4
i_post_pvr            = 5
i_post_cvp            = 6
i_post_rv_dp          = 7
i_post_rv_sp          = 8
i_post_rvedp          = 9
i_post_aov_mean_pg    = 10
i_post_aov_peak_pg    = 11
i_post_mv_decel_time  = 12
i_post_mv_e_a_ratio   = 13
i_post_pv_at          = 14
i_post_pv_max_pg      = 15
i_post_ra_pressure    = 16
i_post_ra_vol_a4c     = 17
i_post_la_vol_a4c     = 18
i_post_lv_esv         = 19
i_post_lv_vol_a4c     = 20
i_post_lvef           = 21
i_post_lvot_max_flow  = 22
i_post_pap_dia        = 23
i_post_pap_sys        = 24
i_post_wedge_pressure = 25

def getMean(t,y):
  '''
  Get the mean of a signal in time
  Computing the integral and dividing by the total time
  '''
  return np.trapz(y,t)/(t[-1]-t[0])

def getMeanValveOpen(t,y):
  newY = y.copy()
  newY[newY > 0] = 1.0
  timePositive = np.trapz(newY,t)
  return np.trapz(y,t)/timePositive

def zeroAtValveOpening(curve,valveIsOpen):
  """
  Determines valve opening time and cyclically shifts curve
  """
  if(np.count_nonzero(valveIsOpen) > 0):
    if(valveIsOpen[0] == 0):
      valveOpeningIDX = valveIsOpen.nonzero()[0][0]
    else:
      rev = np.asarray(np.logical_xor(valveIsOpen,np.ones(len(valveIsOpen))))
      valveOpeningIDX = np.nonzero(rev)[0][-1]
    # Return circular shif of vector starting from valve opening
    return np.roll(curve,-valveOpeningIDX)
  else:
    return None

class peakRecord(object):
  def __init__(self,iMin,tMin,yMin,iMax,tMax,yMax):
    self.iMin = iMin
    self.tMin = tMin
    self.yMin = yMin
    self.iMax = iMax
    self.tMax = tMax
    self.yMax = yMax

def getPeaks(t,curve):

  # Find Peaks
  idxMax = sp.find_peaks(curve)[0]
  idxMin = sp.find_peaks(-curve)[0]

  # Store Peaks in record and return
  peaks = peakRecord(idxMin,t[idxMin],curve[idxMin],idxMax,t[idxMax],curve[idxMax])
  return peaks

def getAccelerationTime(peaks,t0):
  if(len(peaks.iMax) > 0):
    # The first maximum is assumed to be S
    at = (peaks.tMax[0]-t0) * 1000.0
    return True,at
  else:
    return False,0.0

def getDecelerationTime(peaks):
  if( (len(peaks.iMax) > 0) and (len(peaks.iMin) > 0)):
    # The minimum (M) must follow the maximum (S)
    if(peaks.tMin[0] > peaks.tMax[0]):      
      dt = (peaks.tMin[0] - peaks.tMax[0]) * 1000.0
      return True,dt
    else:
      return False,0.0
  else:
    return False,0.0

def getEARatio(peaks):
  # At least two maxima
  if(len(peaks.iMax) > 1):
    EARatio = (peaks.yMax[0]/peaks.yMax[1])
    return True,EARatio
  else:
    return False,0.0

class cvsim6(circuitModel):

  def __init__(self,cycleTime,totalCycles,forcing=None,debugMode=False):
    # Init parameters
    numParam    = 23
    numState    = 6
    numAuxState = 12
    numOutputs  = 26
    self.debugMode = debugMode

    icName = np.array(['p_lv', # Left ventricular pressure
                       'p_art', # Systemic arterial pressure
                       'p_ven', # Systemic vanous pressure
                       'p_rv', # Right ventricular pressure
                       'p_pa', # Pulmonary arterial pressure
                       'p_pv']) # Pulmonary venous pressure

    parName = np.array([# Heart rate
                        'hr',
                        # Left ventricular input and output resistance
                        'r_li','r_lo',
                        # Arterial resistance
                        'r_a',
                        # Right ventricular input and output resistance
                        'r_ri','r_ro',
                        # Pulmonary venous resistance
                        'r_pv',
                        # Thoracic pressure
                        'p_th',
                        # Arterial and venous systemic capacitances
                        'c_a','c_v',
                        # Arterial and vanous pulmonary capacitances
                        'c_pa','c_pv',
                        # Systolic heart cycle ratio
                        'tr_sys',
                        # Left ventricular systolic and diastolic capacitances
                        'c_l_sys','c_l_dia',
                        # Right ventricular systolic and diastolic capacitances
                        'c_r_sys','c_r_dia',
                        # Unstress volumes
                        'v_0_lv','v_0_a','v_0_v','v_0_rv','v_0_pa','v_0_pv'])

    resName = np.array(["heart_rate2", # heartRate - ip_0002_heart_rate2
                        "systolic_bp_2", # maxAOPress - ip_0002_systolic_bp_2
                        "diastolic_bp_2", # minAOPress - ip_0002_diastolic_bp_2
                        "cardiac_output", # CO - ip_0002_cardiac_output
                        "systemic_vascular_resistan", # params[32]+params[33] - ip_0002_systemic_vascular_resistan
                        "pulmonary_vascular_resista", # params[30] - ip_0002_pulmonary_vascular_resista
                        "cvp", # avRAPress - ip_0002_cvp
                        "right_ventricle_diastole", # minRVPress - ip_0002_right_ventricle_diastole
                        "right_ventricle_systole", # maxRVPress - ip_0002_right_ventricle_systole
                        "left_ventricle_diastole", # minLVPress
                        "left_ventricle_systole", # maxLVPress
                        "rvedp", # RVEDP - ip_0002_rvedp
                        "aov_mean_pg", # meanAOVPG - ip_0002_aov_mean_pg
                        "aov_peak_pg", # maxAOVPG - ip_0002_aov_peak_pg
                        "mv_decel_time", # mvDecelTime - ip_0002_mv_decel_time
                        "mv_e_a_ratio", # mvEARatio - ip_0002_mv_e_a_ratio
                        "pv_at", # pvAccelTime - ip_0002_pv_at
                        "pv_max_pg", # maxPVPG - ip_0002_pv_max_pg
                        "ra_pressure", # avRAPress - ip_0002_ra_pressure
                        "ra_vol_a4c", # minRAVolume - ip_0002_ra_vol_a4c - End Systolic
                        "la_vol_a4c", # minLAVolume - ip_0002_la_vol_a4c - End Systolic
                        "lv_esv", # minLVVolume - ip_0002_lv_esv
                        "lv_vol_a4c", # maxLVVolume - ip_0002_lv_vol
                        "lvef", # LVEF - ip_0002_lvef
                        "pap_diastolic", # minPAPress - ip_0002_pap_diastolic
                        "pap_systolic", # maxPAPress - ip_0002_pap_systolic
                        "wedge_pressure"]) # avPCWPress - ip_0002_wedge_pressure

    limits = np.array([[72.0,72.0], # hr - Heart Rate
                       [0.01*mmHgToBarye,0.01*mmHgToBarye], # r_li - input left ventricular resistance
                       [0.006*mmHgToBarye,0.006*mmHgToBarye], # r_lo - output left ventricular resistance
                       [1.00*mmHgToBarye,1.00*mmHgToBarye], # r_a - arterial resistance
                       [0.05*mmHgToBarye,0.05*mmHgToBarye], # r_ri - input right ventricular resistance
                       [0.003*mmHgToBarye,0.003*mmHgToBarye], # r_ro - output right ventricular resistance
                       [0.08*mmHgToBarye,0.08*mmHgToBarye], # r_pv - pulmonary venous resistance
                       [-4.0*mmHgToBarye,-4.0*mmHgToBarye], # p_th - thoracic pressure
                       [1.6/mmHgToBarye,1.6/mmHgToBarye], # c_a - arterial capacitance
                       [100.0/mmHgToBarye,100.0/mmHgToBarye], # c_v - venous capacitance
                       [4.3/mmHgToBarye,4.3/mmHgToBarye], # c_pa - capacitance of pulmonary arteries
                       [8.4/mmHgToBarye,8.4/mmHgToBarye], # c_pv - capacitance of pulmonary veins
                       [0.33,0.33], # 'tr_sys' - Systolic fraction of heart rate
                       [0.4/mmHgToBarye,0.4/mmHgToBarye], # c_l_sys - Left ventricular systolic compliance
                       [10.0/mmHgToBarye,10.0/mmHgToBarye], # c_l_dia - Left ventricular diastolic compliance
                       [1.2/mmHgToBarye,1.2/mmHgToBarye], # c_r_sys - Right ventricular systolic compliance
                       [20.0/mmHgToBarye,20.0/mmHgToBarye], # c_r_dia - Right ventricular diastolic compliance
                       [15.00,15.00], # v_0_lv - Unstressed left ventricular volume
                       [715.00,715.00], # v_0_a - Unstressed arterial volume
                       [2500.00,2500.00], # v_0_v - Unstressed venous volume
                       [15.00,15.00], # v_0_rv - Unstressed right ventricular volume
                       [90.00,90.00], # v_0_pa - Unstressed pulmonary arterial volume
                       [490.00,490.00]]) # v_0_pv - Unstressed pulmonary venous volume

    # NOTE: CGS Units
    # Default Initial Conditions
    defIC = np.array([0.0,
                      0.0,
                      0.0,
                      0.0,
                      0.0,
                      0.0])
  
    # NOTE: CGS Units
    defParam = np.array([72.0,
                         0.01*mmHgToBarye,
                         0.006*mmHgToBarye,
                         1.00*mmHgToBarye,
                         0.05*mmHgToBarye,
                         0.003*mmHgToBarye,
                         0.08*mmHgToBarye,
                         -4.0*mmHgToBarye,
                         1.6/mmHgToBarye,
                         100.0/mmHgToBarye,
                         4.3/mmHgToBarye,
                         8.4/mmHgToBarye,
                         0.33,
                         0.4/mmHgToBarye,
                         10.0/mmHgToBarye,
                         1.2/mmHgToBarye,
                         20.0/mmHgToBarye,
                         15.00,
                         715.00,
                         2500.00,
                         15.00,
                         90.00,
                         490.00])

    #  Invoke Superclass Constructor
    super().__init__(numParam,numState,numAuxState,numOutputs,
                     icName,parName,resName,
                     limits,defIC,defParam,
                     cycleTime,totalCycles,forcing=None,debugMode=debugMode)

  def eval_IC(self,params):
    return eval_cvsim6_ic(params)

  def evalDeriv(self,t,y,params):
    return evalDeriv_cvsim6(t,y,params,self.numState,self.numAuxState)

  def plot_model(self,t,y,aux,start,stop):
    # Plot results
    plt.figure(figsize=(10,3))
    plt.subplot(1,2,1)
    plt.plot(t,y[i_p_l]/mmHgToBarye,label='p_l')
    plt.plot(t,y[i_p_a]/mmHgToBarye,label='p_a')
    plt.plot(t,y[i_p_v]/mmHgToBarye,label='p_v')
    plt.plot(t,y[i_p_r]/mmHgToBarye,label='p_r')
    plt.plot(t,y[i_p_pa]/mmHgToBarye,label='p_pa')
    plt.plot(t,y[i_p_pv]/mmHgToBarye,label='p_pv')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Pressure [mmHg]')
    plt.subplot(1,2,2)
    plt.plot(aux[i_v_l],y[i_p_l]/mmHgToBarye,label='left')
    plt.legend()
    plt.xlabel('Volume [mL]')
    plt.ylabel('Pressure [mmHg]')
    plt.tight_layout()
    plt.show()
    plt.close()

  def postProcess(self,t,y,aux,start,stop):

    # HEART RATE PARAMETER
    heartRate = self.params[0]
    
    # SYSTOLIC, DIASTOLIC AND AVERAGE BLOOD PRESSURES
    minAOPress = np.min(y[i_p_a,start:stop])
    maxAOPress = np.max(y[i_p_a,start:stop])
    avAOPress  = getMean(t[start:stop],y[i_p_a,start:stop])        
    
    # RA PRESSURE
    minRAPress = np.min(y[i_p_v,start:stop])
    maxRAPress = np.max(y[i_p_v,start:stop])
    avRAPress  = getMean(t[start:stop],aux[i_p_v,start:stop])
    
    # RV PRESSURE
    minRVPress  = np.min(y[i_p_r,start:stop])
    maxRVPress  = np.max(y[i_p_r,start:stop])
    avRVPress   = getMean(t[start:stop],aux[i_p_r,start:stop])
        
    # SYSTOLIC, DIASTOLIC AND AVERAGE PA PRESSURES
    minPAPress = np.min(y[i_p_pa,start:stop])
    maxPAPress = np.max(y[i_p_pa,start:stop])
    avPAPress  = getMean(t[start:stop],y[i_p_pa,start:stop])
    
    # PWD OR AVERAGE LEFT ATRIAL PRESSURE
    # AVERAGE PCWP PRESSURE - INDIRECT MEASURE OF LEFT ATRIAL PRESSURE
    avPCWPress = getMean(t[start:stop],y[i_p_pv,start:stop])
    
    # LEFT VENTRICULAR PRESSURES
    minLVPress  = np.min(y[i_p_l,start:stop])
    maxLVPress  = np.max(y[i_p_l,start:stop])
    avgLVPress  = getMean(t[start:stop],y[i_p_l,start:stop])
        
    # CARDIAC OUTPUT
    CO = getMean(t[start:stop],aux[i_q_a,start:stop])
    
    # LEFT AND RIGHT VENTRICULAR VOLUMES
    minRVVolume = np.min(aux[i_v_r,start:stop])
    maxRVVolume = np.max(aux[i_v_r,start:stop])
    minLVVolume = np.min(aux[i_v_l,start:stop])
    maxLVVolume = np.max(aux[i_v_l,start:stop])

    # END SYSTOLIC RIGHT ATRIAL VOLUME
    minRAVolume = np.min(aux[i_v_r,start:stop])
    
    # END SYSTOLIC LEFT ATRIAL VOLUME
    minLAVolume = np.min(aux[i_v_l,start:stop])
    
    # EJECTION FRACTION
    LVEF = ((maxLVVolume - minLVVolume)/maxLVVolume)*100.0
    RVEF = ((maxRVVolume - minRVVolume)/maxRVVolume)*100.0
    
    # RIGHT VENTRICULAR PRESSURE AT BEGINNING OF SYSTOLE
    RVEDP = y[i_p_r,start]
    
    # PRESSURE GRADIENT ACROSS AORTIC VALVE
    output = np.empty(stop-start)
    output[:] = np.abs(y[i_p_a,start:stop] - y[i_p_l,start:stop]) # fabs(aortic - LV)
    maxAOVPG  = np.max(output)
    meanAOVPG = getMeanValveOpen(t[start:stop],output)
    
    # PRESSURE GRADIENT ACROSS PULMONARY VALVE
    output[:] = np.abs(y[i_p_pa,start:stop] - aux[i_p_r,start:stop]) # fabs(pulmonary - RV)
    maxPVPG  = np.max(output)
    meanPVPG = getMeanValveOpen(t[start:stop],output)
    
    # MITRAL FLOW - REPOSITION TO VALVE OPENING
    if(False):
      mvflowFromValveOpening = zeroAtValveOpening(y[7][start:stop],aux[14][start:stop])
      if(mvflowFromValveOpening is None):
        print("ERROR: Mitral valve is not opening in last heart cycle.")
      # FIND MITRAL FLOW PEAKS
      mitralPeaks = getPeaks(t[start:stop],mvflowFromValveOpening)
      # MITRAL VALVE DECELERATION TIME
      isDecelTimeOK,mvDecelTime = getDecelerationTime(mitralPeaks)
      # MITRAL VALVE E/A RATIO
      isMVEARatioOK,mvEARatio = getEARatio(mitralPeaks)
    else:
      mvDecelTime = 0.0
      mvEARatio = 0.0

    if(False):
      plt.plot(t[start:stop],mvflowFromValveOpening)
      plt.scatter(mitralPeaks.tMax,mitralPeaks.yMax)
      plt.scatter(mitralPeaks.tMin,mitralPeaks.yMin)
      plt.show()
      exit(-1)

    # PULMONARY VALVE ACCELERATION TIME
    # SHIFT CURVE WITH BEGINNING AT VALVE OPENING
    if(False):
      pvflowFromValveOpening = zeroAtValveOpening(y[6][start:stop],aux[13][start:stop])
      if(pvflowFromValveOpening is None):
        print("ERROR: Second Valve is not opening in heart cycle.")
        exit(-1)
      # FIND PULMONARY FLOW PEAKS
      pulmonaryPeaks = getPeaks(t[start:stop],pvflowFromValveOpening)
      isPVAccelTimeOK,pvAccelTime = getAccelerationTime(pulmonaryPeaks,t[start])
    else:
      pvAccelTime = 0.0

    if(False):
      plt.plot(t[start:stop],pvflowFromValveOpening)
      plt.scatter(pulmonaryPeaks.tMax,pulmonaryPeaks.yMax)
      plt.scatter(pulmonaryPeaks.tMin,pulmonaryPeaks.yMin)
      plt.show()
      exit(-1)

    if(self.debugMode):
      print('')
      print("mvDecelTime: %f" % (mvDecelTime))
      print("mvEARatio: %f" % (mvEARatio))
      print("pvAccelTime: %f" % (pvAccelTime))

    # ALTERNATIVE COMPUTATION OF SVR and PVR
    altSVR = 80.0*(((avAOPress/1333.22) - (avRAPress/1333.22))/(CO*(60.0/1000.0)))
    altPVR = 80.0*(((avPAPress/1333.22) - (avPCWPress/1333.22))/(CO*(60.0/1000.0)))

    # Assign Results Based on Model Version
    res = np.array([heartRate, # ip_0002_heart_rate2
                    maxAOPress * baryeTommHg, # ip_0002_systolic_bp_2
                    minAOPress * baryeTommHg, # ip_0002_diastolic_bp_2
                    CO * 60.0/1000.0, # ip_0002_cardiac_output
                    altSVR, # ip_0002_systemic_vascular_resistan
                    altPVR, # ip_0002_pulmonary_vascular_resista
                    avRAPress * baryeTommHg, # ip_0002_cvp
                    minRVPress * baryeTommHg, # ip_0002_right_ventricle_diastole
                    maxRVPress * baryeTommHg, # ip_0002_right_ventricle_systole
                    minLVPress * baryeTommHg, # left_ventricle_diastole
                    maxLVPress * baryeTommHg, # left_ventricle_systole
                    # Right ventricular volume at beginning of systole
                    RVEDP * baryeTommHg, # ip_0002_rvedp
                    meanAOVPG * baryeTommHg, # ip_0002_aov_mean_pg
                    maxAOVPG * baryeTommHg, # ip_0002_aov_peak_pg
                    mvDecelTime, # ip_0002_mv_decel_time
                    mvEARatio, # ip_0002_mv_e_a_ratio
                    pvAccelTime, # ip_0002_pv_at
                    maxPVPG * baryeTommHg, # ip_0002_pv_max_pg
                    avRAPress * baryeTommHg, # ip_0002_ra_pressure
                    # Assume maximum (diastolic) volume
                    minRAVolume, # ip_0002_ra_vol_a4c - End Systolic
                    minLAVolume, # ip_0002_la_vol_a4c - End Systolic
                    minLVVolume, # ip_0002_lv_esv
                    # Assume maximum (diastolic) volume
                    maxLVVolume, # ip_0002_lv_vol
                    LVEF, # ip_0002_lvef
                    minPAPress * baryeTommHg, # ip_0002_pap_diastolic
                    maxPAPress * baryeTommHg, # ip_0002_pap_systolic
                    avPCWPress* baryeTommHg]) # ip_0002_wedge_pressure

    return res

# TESTING MODEL
if __name__ == "__main__":

  cycleTime = 1.07
  totalCycles = 10
  model = cvsim6(cycleTime,totalCycles,debugMode=False)

  # Get Default Initial Conditions
  # y0        = model.defIC
  y0        = None
  params    = model.defParam
  outs      = model.solve(params=params,y0=y0)
  outLabels = model.resName

  if(False):
    for loopA in range(len(outLabels)):
      print("%30s %-8f" % (outLabels[loopA],outs[loopA]))

  # Array with measurement standard deviations - same size of the model result vector
  stds = np.array([3.0,  # ip_0002_heart_rate2
                   1.5,  # ip_0002_systolic_bp_2
                   1.5,  # ip_0002_diastolic_bp_2
                   0.2,  # ip_0002_cardiac_output
                   50.0, # ip_0002_systemic_vascular_resistan
                   5.0,  # ip_0002_pulmonary_vascular_resista
                   0.5,  # ip_0002_cvp
                   1.0,  # ip_0002_right_ventricle_diastole
                   1.0,  # ip_0002_right_ventricle_systole
                   1.0,  # left_ventricle_diastole
                   1.0,  # left_ventricle_systole
                   1.0,  # ip_0002_rvedp
                   0.5,  # ip_0002_aov_mean_pg
                   0.5,  # ip_0002_aov_peak_pg
                   6.0,  # ip_0002_mv_decel_time
                   0.2,  # ip_0002_mv_e_a_ratio
                   6.0,  # ip_0002_pv_at
                   0.5,  # ip_0002_pv_max_pg
                   0.5,  # ip_0002_ra_pressure
                   3.0,  # ip_0002_ra_vol_a4c - End Systolic
                   3.0,  # ip_0002_la_vol_a4c - End Systolic
                   10.0, # ip_0002_lv_esv
                   20.0, # ip_0002_lv_vol
                   2.0,  # ip_0002_lvef
                   1.0,  # ip_0002_pap_diastolic
                   1.0,  # ip_0002_pap_systolic
                   1.0]) # ip_0002_wedge_pressure

  # Evaluate Model Log-Likelihood
  # dbFile = '../data/EHR_dataset.csv'
  dbFile = '../data/validation_dataset.csv'
  columnID = 2 # First Patient

  ll = model.evalNegLL(columnID,dbFile,stds,params,y0)

  print("Model Negative LL: ",ll)