# mhd-boris
Finite Volume Constrained Transport simulation of the Orszag-Tang vortex

Philip Mocz (2024), [@PMocz](https://twitter.com/PMocz)

Based on 
[📝 Read the Algorithm Write-up on Medium](https://levelup.gitconnected.com/create-your-own-constrained-transport-magnetohydrodynamics-simulation-with-python-276f787f537d)


Simulate the Orszag-Tang vortex MHD problem
with the Boris Integrator

Run the simulations and produce analysis with:

```python
python mhd-boris.py 2.0
python mhd-boris.py 1.8
python mhd-boris.py 1.6
python mhd-boris.py 1.4
python mhd-boris.py 1.2
python mhd-boris.py 1.0
python mhd-boris.py 0.8
python mhd-boris.py 0.6
python mhd-boris.py 0.4
python analysis.py
```

Density:

![Analysis1](./analysis_rho.png)

Magnetic Pressure:

![Analysis2](./analysis_P_B.png)

Timesteps:

![Analysis3](./analysis_dt.png)

