# mhd-boris
Finite Volume Constrained Transport simulation of the Orszag-Tang vortex

Philip Mocz (2024), [@PMocz](https://twitter.com/PMocz)

Based on 
[📝 Read the Algorithm Write-up on Medium](https://levelup.gitconnected.com/create-your-own-constrained-transport-magnetohydrodynamics-simulation-with-python-276f787f537d)


Simulate the Orszag-Tang vortex MHD problem
with the Boris Integrator

Run the simulations with:

```python
python mhd-boris.py 2.4
python mhd-boris.py 2.2
python mhd-boris.py 2.0
python mhd-boris.py 1.8
python mhd-boris.py 1.6
python mhd-boris.py 1.4
```

And then a anlyze with:
```python
python analysis.py
```

![Analysis1](./analysis_rho.png)

![Analysis2](./analysis_P_B.png)

![Analysis3](./analysis_dt.png)

