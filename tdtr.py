#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:29:53 2018

@author: jpfeser
"""

import numpy as np
from numpy import pi,sqrt,exp

def main():
    
    #-------------TYPE THERMAL SYSTEM PARAMTERS HERE--------------
    # Anticipated system properties (initial guess for fitting, if you do fitting/errorbar estimation)
    
    abslayer =10;
    # Thermal Conductivity, W/m-K
    ky = np.array([4*abslayer, 4,  .1, 1000])
    # Estimated Uncertainty in ky, W/m-K - optional
    err_ky = 0.1*np.ones(ky.size)
    
    # Heat Capacity in J/m^3-K
    C =np.array([2*abslayer, 2, 0.1, 1.2])*1e6
    # Estimated Uncertainty in C, J/m^3-K - optional
    err_C = 0.05*np.ones(ky.size)
    
    #Thickness of each layer in m ("Al" thickness estimated at 84nm=81nm Al+3nm oxide, from picosecond acoustic)
    h =np.array([1, (100-abslayer), 1, 1e6])*1e-9
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
    
    dT_SS = SS_Heating(ky,C,h,eta,r_pump,r_probe,absorbance,A_tot_powermeter)
    print(dT_SS)

def SS_Heating(ky,C,h,eta,r_pump,r_probe,absorbance,A_tot_powermeter):
    f=np.array([[0]]); # set laser Modulation frequency, Hz (i.e. steady state)
    A_abs = absorbance*A_tot_powermeter
    kmin=1/(10000*max(r_pump,r_probe))
    kmax=1/sqrt(r_pump**2+r_probe**2)*1.5
    temp_integrand = lambda x: temp_fluctuation(x,f,ky,C,h,eta,r_pump,r_probe,A_abs)
    (dT_SS,n)  =romberg_integration(temp_integrand,kmin,kmax,1)
    return dT_SS

def romberg_integration(f,xstart,xend,NN):
    """
    Romberg method for integrating multiple scalar functions at the same time
    sol = int_a^b f*dx where f contains multiple vectorized functions as separate rows
    f: is an anonymous function that produces a MATRIX as follows,
        f integrates across x, represented as separate columns of f.
        each seperate row is a seperate "scalar function")
    a: start of x
    b: end of x
    NN: number of functions to be integrated.
    
    example: Integrate the first three polynomials...
        f = lambda x: np.block([[x],[x**2],[x**3]])
        result = tdtr.romberg_integration(f,0,1,3)
    """
    relerr=1
    abserr=1
    limit=1e-5
    nmax=25
    mmax=25
    R=np.zeros((NN,nmax+1,mmax+1),dtype='complex64')
    
    n=0
    m=0
    R[0:NN,0,0]=0.5*(xend-xstart)*(f(xstart).flatten()+f(xend).flatten())
    while np.max(relerr)>limit:
        n=n+1
        h=(xend-xstart)/(2**(n))
        k=np.arange(1,2**(n-1)+1)
        x=xstart*np.ones(k.shape)+h*(2*k-1)
        feval=np.empty((NN,x.size))
        feval=f(x)
        R[0:NN,n,0]=0.5*R[0:NN,n-1,0]+h*np.sum(feval,1);
        for m in np.arange(1,n+1):
            fourm=4**m
            R[0:NN,n,m]=1/(fourm-1)*(fourm*R[0:NN,n,m-1]-R[0:NN,n-1,m-1]) 
            relerr=np.abs(1-R[0:NN,n-1,m-1]/R[0:NN,n,m])
            abserr=np.abs(R[0:NN,n,m]-R[0:NN,n-1,m-1])
            if n==nmax:
                print('max iterations reached')
                sol=R[1:NN,n+1,m+1]
                nfinal=n
                return (sol,nfinal) 
        
    sol=R[0:NN,n,m];
    nfinal=n
    return (sol,nfinal) 

def temp_fluctuation(k_rvect,f_cvect,ky,C,h,eta,r_pump,r_probe,A_pump):
    """
    Computes frequency domain average temperature response to 
    Periodic gaussian pump beam, probed by another gaussian beam
    
    This program is vectorized/optimized to handle all frequencies 
    simultaneously (f is a column vector)
    
    Definitions
    k_rvect: wavenumber of integrand (1/m), row vector..to be integrated across
    f_cvect: excitation frequency (Hz), column vector
    ky: vector of thermal conductivities, 
       ky(1)=top surface,(W/m-K)
    C: vector of volumetric specific heat (J/m3-K)
    h: thicknesses of each layer (layer N will NOT be used, semiinfinite)
    r_pump: Pump spot size (m)
    r_probe: Probe spot size (m)
    A_pump: Pump power (W), used to ESTIMATE amplitude (not used for fitting)
    """
    # first check to see if either of these are scalars (not numpy arrays)
    if np.isscalar(k_rvect):
        k_rvect = np.array([[k_rvect]])
    if np.isscalar(f_cvect):
        f_cvect = np.array([[f_cvect]])
    (fmat,kmat)=np.meshgrid(f_cvect.flatten(),k_rvect.flatten(),indexing='ij')
    Nlayers=ky.size
    Nint = k_rvect.size
    Nfreq = f_cvect.size
    alpha = ky/C
    omegamat = 2*pi*fmat
    kmat2 = kmat**2
    kterm2=4*pi**2*kmat2;
    
    #intialize some stuff
    q2 = 1j*omegamat/alpha[-1]
    un=sqrt(4*pi**2*eta[-1]*kmat**2+q2)
    gamman=ky[-1]*un
    Bplus=np.zeros((Nfreq,Nint))
    Bminus=np.ones((Nfreq,Nint))
    
    if Nlayers!=1:
        for n in np.arange(Nlayers,1,-1):
            q2 = 1j*omegamat/alpha[n-2]
            unminus=sqrt(eta[n-2]*kterm2+q2)
            gammanminus=ky[n-2]*unminus;
            AA=gammanminus+gamman
            BB=gammanminus-gamman
            temp1=AA*Bplus+BB*Bminus
            temp2=BB*Bplus+AA*Bminus
            expterm=exp(unminus*h[n-2]);
            Bplus=(0.5/(gammanminus*expterm))*temp1;
            Bminus=0.5/(gammanminus)*expterm*temp2;
            # These next 3 lines fix a numerical stability issue if one of the
            # layers is very thick or resistive;
            penetration_logic=h[n-2]*np.abs(unminus)>100  #if pentration is smaller than layer...set to semi-inf
            Bplus[penetration_logic]=0;
            Bminus[penetration_logic]=1;
            ### 
            un=unminus;
            gamman=gammanminus;
    
    G=(Bplus+Bminus)/(Bminus-Bplus)/gamman #The layer G(k)
    kernal=2*pi*A_pump*exp(-pi**2*(r_pump**2+r_probe**2)/2*kmat**2)*kmat; #The rest of the integrand
    integrand=G*kernal

    return integrand
    
def tdtr_reflectivity(td,TCR,tau_rep,f_mod,ky,C,h,eta,r_pump,r_probe,A_pump): 
    """
    %Calculates the Reflectivity Signal and Ratio
    %In order to speed up the code, it is parallelized...the convention is...
    %tdelay 1D array of desired delay times
    %mvect (the fourier components/frequencies) is a column vector
    %Matrices have size, # rows=length(tdelay) x #columns=length(Mvect)
    """
    if np.isscalar(td):
        td = np.array([td])
    elif td.ndim>1:
        print('Warning: td is not 1D.  Flattening.')
        td = td.flatten()
        
    fmax=10/np.min(np.abs(td))
    M=int(np.ceil(tau_rep*fmax)) #Highest Fourier component considered
    mvect=np.arange(-M,M+1,1).reshape(-1,1)
    fudge1=exp(-pi*((mvect/tau_rep+f_mod)/fmax)**2);#artificial decay (see RSI paper)
    fudge2=exp(-pi*((mvect/tau_rep-f_mod)/fmax)**2); # shape (2M+1)x(1)
    
    kmin = 0.0
    kmax=1/sqrt(r_pump**2+r_probe**2)*1.5;
    NN=2*M+1
    
    
    temp_integrand_plusf = lambda x: temp_fluctuation(x,mvect/tau_rep+f_mod,ky,C,h,eta,r_pump,r_probe,A_pump)
    temp_integrand_minusf = lambda x: temp_fluctuation(x,mvect/tau_rep+f_mod,ky,C,h,eta,r_pump,r_probe,A_pump)
    (dT1,n)  =romberg_integration(temp_integrand_plusf,kmin,kmax,NN)
    (dT2,n)  =romberg_integration(temp_integrand_minusf,kmin,kmax,NN)
    dT1 = dT1.reshape(-1,1)
    dT2 = dT1.reshape(-1,1)
    #dTs are column vectors
    
    expterm=exp(1j*2*pi/tau_rep*(td.reshape(-1,1) @ mvect.reshape(1,-1))); # shape: (Ntd) x (2M+1)
    Retemp= (np.ones((td.size,1)) @ (dT1.transpose()*fudge1.transpose() + dT2.transpose()*fudge2.transpose()))*expterm; 
    Imtemp= -1j*(np.ones((td.size,1)) @ (dT1.transpose()*fudge1.transpose()-dT2.transpose()*fudge2.transpose()))*expterm; # shape: (Ntd) x (2M+1)
#
    Resum=np.sum(Retemp,axis=1); #Sum over all Fourier series components
    Imsum=np.sum(Imtemp,axis=1);
#
#%TCR =1;
    deltaRm=TCR*(Resum +1j*Imsum);
    deltaR=deltaRm*exp(1j*2*pi*f_mod*td); #Reflectance Fluxation (Complex)
#deltaT = deltaR;
#
    ratio=-np.real(deltaR)/np.imag(deltaR);
    return (ratio,deltaR)