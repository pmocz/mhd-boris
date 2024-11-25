# Analysis of Boris Simulation results

import numpy as np
import matplotlib.pyplot as plt



def main():

  max_cfs = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

  rho = {}
  P_B = {}
  dt = {}

  # Load the data (rho, P_B, dt)
  for max_cf in max_cfs:
    rho[max_cf] = np.load('data_rho_'+str(max_cf)+'.npy')
    P_B[max_cf] = np.load('data_P_B_'+str(max_cf)+'.npy')
    dt[max_cf] = np.load('data_dt_'+str(max_cf)+'.npy')

  # stack the rho next to each other and plot it as an image
  rho_all = np.hstack([rho[max_cf] for max_cf in max_cfs])
  P_B_all = np.hstack([P_B[max_cf] for max_cf in max_cfs])

  plt.figure(figsize=(12, 4))
  plt.imshow(rho_all, aspect=1, cmap='jet')
  plt.colorbar()
  plt.xlabel('cf_max = '+str(max_cfs))
  plt.savefig('analysis_rho.png')
  plt.show()

  plt.figure(figsize=(12, 4))
  plt.imshow(P_B_all, aspect=1, cmap='jet')
  plt.colorbar()
  plt.xlabel('cf_max = '+str(max_cfs))
  plt.savefig('analysis_P_B.png')
  plt.show()

  # plot the dt for each max_cf
  plt.figure()
  for max_cf in max_cfs:
    plt.plot(np.cumsum(dt[max_cf]), dt[max_cf], label='cf_max = '+str(max_cf))
  plt.legend(loc='center left')
  plt.xlabel('t')
  plt.savefig('analysis_dt.png')
  plt.show()


  return 0



if __name__== "__main__":
  main()

