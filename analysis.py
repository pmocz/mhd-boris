# Analysis of Boris Simulation results

import numpy as np
import matplotlib.pyplot as plt



def main():

  cf_limits = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

  rho = {}
  P_B = {}
  dt = {}

  # Load the data (rho, P_B, dt)
  for cf_limit in cf_limits:
    rho[cf_limit] = np.load('data_rho_'+str(cf_limit)+'.npy')
    P_B[cf_limit] = np.load('data_P_B_'+str(cf_limit)+'.npy')
    dt[cf_limit] = np.load('data_dt_'+str(cf_limit)+'.npy')

  # stack the rho next to each other and plot it as an image
  rho_all = np.hstack([rho[cf_limit] for cf_limit in cf_limits])
  P_B_all = np.hstack([P_B[cf_limit] for cf_limit in cf_limits])

  plt.figure(figsize=(12, 4))
  plt.imshow(rho_all, aspect=1, cmap='jet')
  plt.colorbar()
  plt.xlabel('cf_max = '+str(cf_limits))
  plt.savefig('analysis_rho.png')
  plt.show()

  plt.figure(figsize=(12, 4))
  plt.imshow(P_B_all, aspect=1, cmap='jet')
  plt.colorbar()
  plt.xlabel('cf_max = '+str(cf_limits))
  plt.savefig('analysis_P_B.png')
  plt.show()

  # plot the dt for each cf_limit
  plt.figure()
  for cf_limit in cf_limits:
    plt.plot(np.cumsum(dt[cf_limit]), dt[cf_limit], label='cf_max = '+str(cf_limit))
  plt.legend(loc='lower left')
  plt.xlabel('t')
  plt.savefig('analysis_dt.png')
  plt.show()


  return 0



if __name__== "__main__":
  main()

