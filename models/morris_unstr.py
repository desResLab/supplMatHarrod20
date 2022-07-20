import numpy as np
from cvsim6 import cvsim6
from scipy.stats.qmc import Sobol
from tqdm import tqdm
from math import fabs
import argparse

def eval_sensitivity(model,num_base2=1):
  # Get Sobol samples
  sampler = Sobol(model.numParam)
  samples = sampler.random_base2(num_base2)

  # Set increment for every parameter
  delta_param = (model.limits[:,1]-model.limits[:,0])/100.0

  # Scale samples to original parameter box
  sample_par = (model.limits[:,0]+delta_param) + samples*(model.limits[:,1]-model.limits[:,0]-2*delta_param)
  
  # Initialize storage 
  morris_res = np.zeros(sample_par.shape)

  # Compute central difference and make it adimensional 
  # Loop on all parameters
  for loopA in range(sample_par.shape[1]):
    print('Computing Morris Single Term Effects for Dimension ',loopA)
    # loop on all Sobol samples
    for loopB in tqdm(range(sample_par.shape[0])):
        curr_param_1 = sample_par[loopB,:].copy()
        curr_param_2 = sample_par[loopB,:].copy()
        curr_param_1[loopA] += delta_param[loopA]
        curr_param_2[loopA] -= delta_param[loopA]
        # Find the ll for the two parameter perturbations
        ll1,_,_,_,_ = model.evalNegLL(columnID,dbFile,stds,curr_param_1,y0=None)
        ll2,_,_,_,_ = model.evalNegLL(columnID,dbFile,stds,curr_param_2,y0=None)
        print(ll1,ll2)
        # Store the adimensional coefficients 
        morris_res[loopB,loopA] = fabs(((ll1-ll2)/(2*delta_param[loopA]))*(curr_param_2[loopA]/ll2))*100

  # Compute mean and standard deviation of sigle effects 
  morris_avg = morris_res.mean(axis=0)
  morris_std = morris_res.std(axis=0)

  return morris_avg,morris_std

# TESTING MODEL
if __name__ == "__main__":

  # Init parser
  parser = argparse.ArgumentParser(description='Perform Morris screening for single-variable effects.')
  
  # num_sample_base2
  parser.add_argument('-n', '--num',
                      action=None,
                      # nargs='+',
                      const=None,
                      default=1,
                      type=int,
                      choices=None,
                      required=True,
                      help='number of samples (base 2) to generate',
                      metavar='',
                      dest='num_samples')


  # Parse Commandline Arguments
  args = parser.parse_args()
  
  # Init model
  cycleTime = 1.07
  totalCycles = 10
  model = cvsim6(cycleTime,totalCycles,debugMode=False)

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
  dbFile = '../data/validation_dataset.csv'
  columnID = 2 # Healty patient

  # Compute Morris Sensitivity Indices
  morris_avg,morris_std = eval_sensitivity(model,args.num_samples)

  # Save file for retrieval
  np.save('cvsim6_morris_avg_'+str(args.num_samples),morris_avg)
  np.save('cvsim6_morris_std_'+str(args.num_samples),morris_std)
  
  print('%-15s %-20s %-20s' % ('Param','Avg Morris Coeff','Std Morris Coeff'))
  for loopA in range(model.numParam):
    print('%-15s %-20.3f %-20.3f' % (model.parName[loopA],morris_avg[loopA],morris_std[loopA]))


  
