#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 10:15:54 2018

@author: jpfeser
"""
import numpy as np
from numpy import logspace, pi, sqrt, exp
import tdtr
import matplotlib.pyplot as plt

#%% test Romberg Integration
f = lambda x: np.block([[x],[x**2],[x**3]])
result = tdtr.romberg_integration(f,0,1,3)

#%% test dT script
# test main script
abslayer =10;
# Thermal Conductivity, W/m-K
ky = np.array([150*abslayer, 150, 0.1, 1.35])
# Estimated Uncertainty in ky, W/m-K - optional
err_ky = 0.1*np.ones(ky.size)

# Heat Capacity in J/m^3-K
C =np.array([2.42*abslayer, 2.42, 0.1, 1.2])*1e6
# Estimated Uncertainty in C, J/m^3-K - optional
err_C = 0.05*np.ones(ky.size)

#Thickness of each layer in m ("Al" thickness estimated at 84nm=81nm Al+3nm oxide, from picosecond acoustic)
h =np.array([1, 70-abslayer, 1, 1e6])*1e-9
# Estimated Uncertainty in h in meters- optional
err_h = 0.05*np.ones(ky.size)

#eta: in-plane vs through-plane ratio kx/ky;
eta=np.ones(ky.size) 
# Estimated Uncertainty in eta - optional
err_eta = 0.00*np.ones(ky.size)

# focused laser spot size, m
r = 25e-6
r_pump=r #pump 1/e^2 radius, m
r_probe=r #probe 1/e^2 radius, m
err_r = 0.2*r

# expected uncertainty in setting RF lockin amplifier phase in DEGREES
err_phase = 0.1

# laser properties 
f_mod = 12.6e6 # Modulation frequency, Hz
f_rep = 76e6 # pulse repition rate, Hz

# Other laser/absorbance properties
absorbance = 0.1
A_tot_powermeter = 200e-3
A_pump=30e-3; #laser power (Watts) . . . only used for amplitude est.
tau_rep=1/f_rep; # laser repetition period, s
TCR=1e-4 #coefficient of thermal reflectance . . . only used for amplitude est.

NN = 200
f = np.transpose([logspace(2,9,NN)])
kmin=1/(10000*max([r_pump,r_probe]))
kmax=1/sqrt(r_pump**2+r_probe**2)*1.5
temp_integrand = lambda x: tdtr.temp_fluctuation(x,f,ky,C,h,eta,r_pump,r_probe,A_tot_powermeter*absorbance)
(dT,n)  =tdtr.romberg_integration(temp_integrand,kmin,kmax,NN)

plt.figure(1)
plt.loglog(f,np.real(dT))

plt.figure(2)
plt.semilogx(f,np.angle(dT))

#%%
td = np.logspace(-10,-9,200)
(ratio,dR)=tdtr.tdtr_reflectivity(td,TCR,tau_rep,f_mod,ky,C,h,eta,r_pump,r_probe,A_pump)

plt.figure(3)
plt.semilogx(td,ratio)
plt.axis([1e-10,1e-9,0,2])