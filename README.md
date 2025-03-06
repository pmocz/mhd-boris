# mhd-boris
2.5D Constrained Transport Magnetohydrodynamics

Philip Mocz (2024), [@PMocz](https://twitter.com/PMocz)

Based on 
[algorithm write-up](https://levelup.gitconnected.com/create-your-own-constrained-transport-magnetohydrodynamics-simulation-with-python-276f787f537d)

See also: https://arxiv.org/abs/1902.02810

Simulate the Orszag-Tang vortex MHD problem
with the Boris-like Integrator

Run the simulations (`python mhd-boris.py <prob_id> <c_limit>`) and produce analysis with:

```bash
python mhd-boris.py 1 2.0
python mhd-boris.py 1 1.8
python mhd-boris.py 1 1.6
python mhd-boris.py 1 1.4
python mhd-boris.py 1 1.2
python mhd-boris.py 1 1.0
python mhd-boris.py 2 1.2
python mhd-boris.py 2 1.0
python mhd-boris.py 2 0.8
python mhd-boris.py 2 0.6
python mhd-boris.py 2 0.4
python analysis.py
```


## 1. Orszag-Tang

Timesteps:

![Analysis1](./p1_dt.png)


Density:

![Analysis2](./p1_rho.png)

Magnetic Pressure:

![Analysis3](./p1_P_B.png)

Velocity:

![Analysis4](./p1_v.png)

Alfven Speed: 

![Analysis5](./p1_ca.png)

Fast Speed:

![Analysis6](./p1_cf.png)


## 2. Alfven Wave

Timesteps:

![Analysis1](./p2_dt.png)


Bz lineout at final time:

![Analysis2](./p2_Bz.png)

