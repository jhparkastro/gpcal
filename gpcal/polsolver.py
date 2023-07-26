#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:59:06 2021

@author: jpark
"""


import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

from astropy.coordinates import EarthLocation
import astropy.time as at

from AIPS import AIPS
from AIPSTask import AIPSTask
from AIPSData import AIPSUVData, AIPSImage
from Wizardry.AIPSData import AIPSUVData as WAIPSUVData

import aipsutil as au
import obshelpers as oh
import plothelpers as ph

import os
from os import path

from IPython import embed


from multiprocessing import cpu_count, Pool


max_count = cpu_count()
nproc = max_count - 1


def cleanqu(direc, data, mask, save, bif, eif, ms, ps, uvbin, uvpower, dynam, shift_x, shift_y, stokes, log = None): # Version 1.1
    """
    Perform imaging of Stokes Q and U in Difmap.
    
    Args:
        data (str): the name of the UVFITS file to be CLEANed.
        mask (str): the name of the CLEAN windows to be used.
        save (str): the name of the output Difmap save file.
        log (str): the name of the Difmap log file.
    """
    
    # logname = ''
    
    
    # Write a simple Difmap script for CLEAN in the working directory.
#        if not path.exists(self.direc+'GPCAL_Difmap_v1'):
    f = open(direc+'GPCAL_Difmap_v1','w')
    
    f.write('observe %1\nmapcolor rainbow, 1, 0.5\nselect %13, %2, %3\nmapsize %4, %5\nuvweight %6, %7\nrwin %8\nshift %9,%10\ndo i=1,100\nclean 100, 0.02, imstat(rms)*%11\nend do\nselect i\nsave %12.%13\nexit')
        
    f.close()
        
    
    curdirec = os.getcwd()
    
    logname = 'gpcal_difmap.log'

    os.chdir(direc)
    if log != None:
        command = "echo @GPCAL_Difmap_v1 %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s | difmap > %s" %(data,bif,eif,ms,ps,uvbin,uvpower,mask,shift_x,shift_y,dynam,save,stokes,logname) # Version 1.1!
    else:
        command = "echo @GPCAL_Difmap_v1 %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s | difmap > /dev/null 2>&1" %(data,bif,eif,ms,ps,uvbin,uvpower,mask,shift_x,shift_y,dynam,save,stokes) # Version 1.1!
    
    # log.info('\nMaking CLEAN models for Stokes Q & U maps for {:s}...'.format(data))
    
    os.system(command)
                
    if log != None:
        os.system('cat ' + logname + ' > ' + log)
    
    os.chdir(curdirec)
    


def cleanqu2(args): # Version 1.1
    cleanqu3(*args)

def cleanqu3(direc, data, mask, save, bif, eif, ms, ps, uvbin, uvpower, dynam, shift_x, shift_y, stokes, log): # Version 1.1
    cleanqu(direc, data, mask, save, bif, eif, ms, ps, uvbin, uvpower, dynam, shift_x, shift_y, stokes, log = None)
    

def cleanqu_run(args): # Version 1.1
    """
    Perform imaging of Stokes Q and U in Difmap.
    
    Args:
        data (str): the name of the UVFITS file to be CLEANed.
        mask (str): the name of the CLEAN windows to be used.
        save (str): the name of the output Difmap save file.
        log (str): the name of the Difmap log file.
    """

    pool = Pool(processes = nproc)
    pool.map(cleanqu2, args)
    pool.close()
    pool.join()
    
    # cleanqu(direc, data, mask, save, bif, eif, ms, ps, uvbin, uvpower, dynam, shift_x, shift_y, stokes, log = log)
    


def deq(x, *p):
    """
    The D-term models for the initial D-term estimation using the similarity assumption.
    
    Args:
        x: dummy parameters (not to be used).
        *p (args): the best-fit parameters args.
    """
    
    nant, cnum, calsour, modamp, modphas, pang1, pang2, ant1, ant2, sourcearr, rramp, rrphas, llamp, llphas, model_rlreal, model_rlimag, model_rlamp, model_rlphas, model_lrreal, model_lrimag, model_lramp, model_lrphas = x
    
    RiLj_Real = np.zeros(len(pang1))
    RiLj_Imag = np.zeros(len(pang1))
    LiRj_Real = np.zeros(len(pang1))
    LiRj_Imag = np.zeros(len(pang1))
    
    dump = np.array(p)
    
    dumreal = dump[2*ant1]
    dumimag = dump[2*ant1 + 1]
    Tot_D_iR_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_iR_phas = np.angle(dumreal + 1j*dumimag)

    dumreal = dump[2*nant + 2*ant2]
    dumimag = dump[2*nant + 2*ant2 + 1]
    Tot_D_jL_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_jL_phas = np.angle(dumreal + 1j*dumimag)
    
    dumreal = dump[2*nant + 2*ant1]
    dumimag = dump[2*nant + 2*ant1 + 1]
    Tot_D_iL_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_iL_phas = np.angle(dumreal + 1j*dumimag)
    
    dumreal = dump[2*ant2]
    dumimag = dump[2*ant2 + 1]
    Tot_D_jR_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_jR_phas = np.angle(dumreal + 1j*dumimag)


    P_ij_amp = [np.absolute(it1 + 1j*it2) for it1, it2 in zip([p[nant * 4 + s * 2] for s in range(sum(cnum))], [p[nant * 4 + s * 2 + 1] for s in range(sum(cnum))])]
    P_ij_phas = [np.angle(it1 + 1j*it2) for it1, it2 in zip([p[nant * 4 + s * 2] for s in range(sum(cnum))], [p[nant * 4 + s * 2 + 1] for s in range(sum(cnum))])]
    
    for l in range(len(calsour)):
        select = (sourcearr == calsour[l])
        
        if(cnum[l] != 0.):
            for t in range(cnum[l]):
                if(l==0):
                    dummodamp = np.array(modamp[t])
                    dummodphas = np.array(modphas[t])
                    Pick = t
                else:
                    dummodamp = np.array(modamp[sum(cnum[0:l])+t])
                    dummodphas = np.array(modphas[sum(cnum[0:l])+t])
                    Pick = sum(cnum[0:l]) + t
                    
                submodamp = dummodamp[select]
                submodphas = dummodphas[select]
                
        
                Pamp = P_ij_amp[Pick]
                Pphas = P_ij_phas[Pick]
                
    
                RiLj_Real[select] += Pamp * submodamp * np.cos(submodphas + Pphas) + \
                  Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * Pamp * submodamp * np.cos(submodphas + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] - Pphas + 2. * (pang1[select] + pang2[select]))
        
                RiLj_Imag[select] += Pamp * submodamp * np.sin(submodphas + Pphas) + \
                  Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * Pamp * submodamp * np.sin(submodphas + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] - Pphas + 2. * (pang1[select] + pang2[select]))
        
                LiRj_Real[select] += Pamp * submodamp * np.cos(submodphas - Pphas) + \
                  Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * Pamp * submodamp * np.cos(submodphas + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] + Pphas - 2. * (pang1[select] + pang2[select]))
        
                LiRj_Imag[select] += Pamp * submodamp * np.sin(submodphas - Pphas) + \
                  Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * Pamp * submodamp * np.sin(submodphas + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] + Pphas - 2. * (pang1[select] + pang2[select]))     
    
    
    RiLj_Real += \
      Tot_D_iR_amp * llamp * np.cos(Tot_D_iR_phas + llphas + 2. * pang1) + \
      Tot_D_jL_amp * rramp * np.cos(-Tot_D_jL_phas + rrphas + 2. * pang2)
    
    RiLj_Imag += \
      Tot_D_iR_amp * llamp * np.sin(Tot_D_iR_phas + llphas + 2. * pang1) + \
      Tot_D_jL_amp * rramp * np.sin(-Tot_D_jL_phas + rrphas + 2. * pang2)

    LiRj_Real += \
      Tot_D_iL_amp * rramp * np.cos(Tot_D_iL_phas + rrphas - 2. * pang1) + \
      Tot_D_jR_amp * llamp * np.cos(-Tot_D_jR_phas + llphas - 2. * pang2)

    LiRj_Imag += \
      Tot_D_iL_amp * rramp * np.sin(Tot_D_iL_phas + rrphas - 2. * pang1) + \
      Tot_D_jR_amp * llamp * np.sin(-Tot_D_jR_phas + llphas - 2. * pang2)
   

    compv = np.concatenate([LiRj_Real, LiRj_Imag, RiLj_Real, RiLj_Imag])
        
    return compv




def synthetic_deq(nant, pang1, pang2, ant1, ant2, llamp, llphas, rramp, rrphas, model_ireal, model_iimag, model_rlreal, model_rlimag, model_lrreal, model_lrimag, stokes, *p):
    
    RiRj_Real = np.zeros(len(pang1))
    RiRj_Imag = np.zeros(len(pang1))
    LiLj_Real = np.zeros(len(pang1))
    LiLj_Imag = np.zeros(len(pang1))
    
    RiLj_Real = np.zeros(len(pang1))
    RiLj_Imag = np.zeros(len(pang1))
    LiRj_Real = np.zeros(len(pang1))
    LiRj_Imag = np.zeros(len(pang1))
    
    model_iamp = np.abs(model_ireal + 1j * model_iimag)
    model_iphas = np.angle(model_ireal + 1j * model_iimag)
    model_rlamp = np.abs(model_rlreal + 1j * model_rlimag)
    model_rlphas = np.angle(model_rlreal + 1j * model_rlimag)
    model_lramp = np.abs(model_lrreal + 1j * model_lrimag)
    model_lrphas = np.angle(model_lrreal + 1j * model_lrimag)
    
    dump = np.array(p)
    
    dumreal = dump[2*ant1]
    dumimag = dump[2*ant1 + 1]
    Tot_D_iR_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_iR_phas = np.angle(dumreal + 1j*dumimag)

    dumreal = dump[2*nant + 2*ant2]
    dumimag = dump[2*nant + 2*ant2 + 1]
    Tot_D_jL_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_jL_phas = np.angle(dumreal + 1j*dumimag)
    
    dumreal = dump[2*nant + 2*ant1]
    dumimag = dump[2*nant + 2*ant1 + 1]
    Tot_D_iL_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_iL_phas = np.angle(dumreal + 1j*dumimag)
    
    dumreal = dump[2*ant2]
    dumimag = dump[2*ant2 + 1]
    Tot_D_jR_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_jR_phas = np.angle(dumreal + 1j*dumimag)
    
    
    RiRj_Real += model_ireal + Tot_D_iR_amp * Tot_D_jR_amp * model_iamp * np.cos(model_iphas + Tot_D_iR_phas - Tot_D_jR_phas + 2. * (pang1 - pang2))
    RiRj_Imag += model_iimag + Tot_D_iR_amp * Tot_D_jR_amp * model_iamp * np.sin(model_iphas + Tot_D_iR_phas - Tot_D_jR_phas + 2. * (pang1 - pang2))
    LiLj_Real += model_ireal + Tot_D_iL_amp * Tot_D_jL_amp * model_iamp * np.cos(model_iphas + Tot_D_iL_phas - Tot_D_jL_phas - 2. * (pang1 - pang2))
    LiLj_Imag += model_iimag + Tot_D_iL_amp * Tot_D_jL_amp * model_iamp * np.sin(model_iphas + Tot_D_iL_phas - Tot_D_jL_phas - 2. * (pang1 - pang2))
    
    RiRj_Real += \
        Tot_D_iR_amp * model_lramp * np.cos(Tot_D_iR_phas + model_lrphas + 2. * pang1) + \
        Tot_D_jR_amp * model_rlamp * np.cos(-Tot_D_jR_phas + model_rlphas - 2. * pang2)
    
    RiRj_Imag += \
        Tot_D_iR_amp * model_lramp * np.sin(Tot_D_iR_phas + model_lrphas + 2. * pang1) + \
        Tot_D_jR_amp * model_rlamp * np.sin(-Tot_D_jR_phas + model_rlphas - 2. * pang2)
    
    LiLj_Real += \
        Tot_D_iL_amp * model_rlamp * np.cos(Tot_D_iL_phas + model_rlphas - 2. * pang1) + \
        Tot_D_jL_amp * model_lramp * np.cos(-Tot_D_jL_phas + model_lrphas + 2. * pang2)
    
    LiLj_Imag += \
        Tot_D_iL_amp * model_rlamp * np.sin(Tot_D_iL_phas + model_rlphas - 2. * pang1) + \
        Tot_D_jL_amp * model_lramp * np.sin(-Tot_D_jL_phas + model_lrphas + 2. * pang2)
        
    
    RiLj_Real += model_rlreal + Tot_D_iR_amp * Tot_D_jL_amp * model_lramp * np.cos(model_lrphas + Tot_D_iR_phas - Tot_D_jL_phas + 2. * (pang1 + pang2))
    RiLj_Imag += model_rlimag + Tot_D_iR_amp * Tot_D_jL_amp * model_lramp * np.sin(model_lrphas + Tot_D_iR_phas - Tot_D_jL_phas + 2. * (pang1 + pang2))
    LiRj_Real += model_lrreal + Tot_D_iL_amp * Tot_D_jR_amp * model_rlamp * np.cos(model_rlphas + Tot_D_iL_phas - Tot_D_jR_phas - 2. * (pang1 + pang2))
    LiRj_Imag += model_lrimag + Tot_D_iL_amp * Tot_D_jR_amp * model_rlamp * np.sin(model_rlphas + Tot_D_iL_phas - Tot_D_jR_phas - 2. * (pang1 + pang2))
        
    RiLj_Real += \
      Tot_D_iR_amp * model_iamp * np.cos(Tot_D_iR_phas + model_iphas + 2. * pang1) + \
      Tot_D_jL_amp * model_iamp * np.cos(-Tot_D_jL_phas + model_iphas + 2. * pang2)
    
    RiLj_Imag += \
      Tot_D_iR_amp * model_iamp * np.sin(Tot_D_iR_phas + model_iphas + 2. * pang1) + \
      Tot_D_jL_amp * model_iamp * np.sin(-Tot_D_jL_phas + model_iphas + 2. * pang2)

    LiRj_Real += \
      Tot_D_iL_amp * model_iamp * np.cos(Tot_D_iL_phas + model_iphas - 2. * pang1) + \
      Tot_D_jR_amp * model_iamp * np.cos(-Tot_D_jR_phas + model_iphas - 2. * pang2)

    LiRj_Imag += \
      Tot_D_iL_amp * model_iamp * np.sin(Tot_D_iL_phas + model_iphas - 2. * pang1) + \
      Tot_D_jR_amp * model_iamp * np.sin(-Tot_D_jR_phas + model_iphas - 2. * pang2)  
   
    if(stokes == 'RR'): return RiRj_Real + 1j * RiRj_Imag
    if(stokes == 'LL'): return LiLj_Real + 1j * LiLj_Imag
    if(stokes == 'LR'): return LiRj_Real + 1j * LiRj_Imag
    if(stokes == 'RL'): return RiLj_Real + 1j * RiLj_Imag



def pol_deq(x, *p):
    """
    The D-term models for the D-term estimation with instrumental polarization self-calibration.
    
    Args:
        x: dummy parameters (not to be used).
        *p (args): the best-fit parameters args.
    """
    
    nant, pang1, pang2, ant1, ant2, rramp, rrphas, llamp, llphas, model_rlreal, model_rlimag, model_rlamp, model_rlphas, model_lrreal, model_lrimag, model_lramp, model_lrphas = inputz
    
    RiLj_Real = np.zeros(len(pang1))
    RiLj_Imag = np.zeros(len(pang1))
    LiRj_Real = np.zeros(len(pang1))
    LiRj_Imag = np.zeros(len(pang1))
    
    
    dump = np.array(p)
    
    dumreal = dump[2*ant1]
    dumimag = dump[2*ant1 + 1]
    Tot_D_iR_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_iR_phas = np.angle(dumreal + 1j*dumimag)

    dumreal = dump[2*nant + 2*ant2]
    dumimag = dump[2*nant + 2*ant2 + 1]
    Tot_D_jL_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_jL_phas = np.angle(dumreal + 1j*dumimag)
    
    dumreal = dump[2*nant + 2*ant1]
    dumimag = dump[2*nant + 2*ant1 + 1]
    Tot_D_iL_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_iL_phas = np.angle(dumreal + 1j*dumimag)
    
    dumreal = dump[2*ant2]
    dumimag = dump[2*ant2 + 1]
    Tot_D_jR_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_jR_phas = np.angle(dumreal + 1j*dumimag)
    
    
    RiLj_Real += model_rlreal + \
      Tot_D_iR_amp * Tot_D_jL_amp * model_lramp * np.cos(model_lrphas + Tot_D_iR_phas - Tot_D_jL_phas + 2. * (pang1 + pang2))

    RiLj_Imag += model_rlimag + \
      Tot_D_iR_amp * Tot_D_jL_amp * model_lramp * np.sin(model_lrphas + Tot_D_iR_phas - Tot_D_jL_phas + 2. * (pang1 + pang2))

    LiRj_Real += model_lrreal + \
      Tot_D_iL_amp * Tot_D_jR_amp * model_rlamp * np.cos(model_rlphas + Tot_D_iL_phas - Tot_D_jR_phas - 2. * (pang1 + pang2))

    LiRj_Imag += model_lrimag + \
      Tot_D_iL_amp * Tot_D_jR_amp * model_rlamp * np.sin(model_rlphas + Tot_D_iL_phas - Tot_D_jR_phas - 2. * (pang1 + pang2))
    
    
    RiLj_Real += \
      Tot_D_iR_amp * llamp * np.cos(Tot_D_iR_phas + llphas + 2. * pang1) + \
      Tot_D_jL_amp * rramp * np.cos(-Tot_D_jL_phas + rrphas + 2. * pang2)
    
    RiLj_Imag += \
      Tot_D_iR_amp * llamp * np.sin(Tot_D_iR_phas + llphas + 2. * pang1) + \
      Tot_D_jL_amp * rramp * np.sin(-Tot_D_jL_phas + rrphas + 2. * pang2)

    LiRj_Real += \
      Tot_D_iL_amp * rramp * np.cos(Tot_D_iL_phas + rrphas - 2. * pang1) + \
      Tot_D_jR_amp * llamp * np.cos(-Tot_D_jR_phas + llphas - 2. * pang2)

    LiRj_Imag += \
      Tot_D_iL_amp * rramp * np.sin(Tot_D_iL_phas + rrphas - 2. * pang1) + \
      Tot_D_jR_amp * llamp * np.sin(-Tot_D_jR_phas + llphas - 2. * pang2)  
   

    compv = np.concatenate([LiRj_Real, LiRj_Imag, RiLj_Real, RiLj_Imag])
    
    
    return compv



def pol_deq_comp(comp, parmset, *p):
    """
    Extract component-wise best-fit models with instrumental polarization self-calibration.
    
    Args:
        comp (str): the name of the component to be extracted.
        *p (args): the best-fit parameters args.
    """
    
    (nant, polcalsour, sourcearr, pang1, pang2, ant1, ant2, model_rlreal, model_rlimag, model_lrreal, model_lrimag, rramp, rrphas, llamp, llphas) = parmset
    
    model_rlamp = np.abs(model_rlreal + 1j * model_rlimag)
    model_rlphas = np.angle(model_rlreal + 1j * model_rlimag)
    model_lramp = np.abs(model_lrreal + 1j * model_lrimag)
    model_lrphas = np.angle(model_lrreal + 1j * model_lrimag)
    
    
    pol_RiLj_Real = np.zeros(len(pang1))
    pol_RiLj_Imag = np.zeros(len(pang1))
    pol_LiRj_Real = np.zeros(len(pang1))
    pol_LiRj_Imag = np.zeros(len(pang1))
    
    dterm_RiLj_Real = np.zeros(len(pang1))
    dterm_RiLj_Imag = np.zeros(len(pang1))
    dterm_LiRj_Real = np.zeros(len(pang1))
    dterm_LiRj_Imag = np.zeros(len(pang1))
    
    second_RiLj_Real = np.zeros(len(pang1))
    second_RiLj_Imag = np.zeros(len(pang1))
    second_LiRj_Real = np.zeros(len(pang1))
    second_LiRj_Imag = np.zeros(len(pang1))
    
    
    dump = np.array(p)
    
    
    dumreal = np.array([dump[2*int(s)] for s in ant1])
    dumimag = np.array([dump[2*int(s)+1] for s in ant1])
    Tot_D_iR_amp = np.sqrt(dumreal**2 + dumimag**2)
    Tot_D_iR_phas = np.angle(dumreal + 1j*dumimag)

    dumreal = np.array([dump[2*nant + 2*int(s)] for s in ant2])
    dumimag = np.array([dump[2*nant + 2*int(s)+1] for s in ant2])
    Tot_D_jL_amp = np.sqrt(dumreal**2 + dumimag**2)
    Tot_D_jL_phas = np.angle(dumreal + 1j*dumimag)


    dumreal = np.array([dump[2*nant + 2*int(s)] for s in ant1])
    dumimag = np.array([dump[2*nant + 2*int(s)+1] for s in ant1])
    Tot_D_iL_amp = np.sqrt(dumreal**2 + dumimag**2)
    Tot_D_iL_phas = np.angle(dumreal + 1j*dumimag)


    dumreal = np.array([dump[2*int(s)] for s in ant2])
    dumimag = np.array([dump[2*int(s)+1] for s in ant2])
    Tot_D_jR_amp = np.sqrt(dumreal**2 + dumimag**2)
    Tot_D_jR_phas = np.angle(dumreal + 1j*dumimag)
    

    for l in range(len(polcalsour)):
      
        select = (sourcearr == polcalsour[l])
        
        pol_RiLj_Real[select] += model_rlreal[select]
        pol_RiLj_Imag[select] += model_rlimag[select]
        pol_LiRj_Real[select] += model_lrreal[select]
        pol_LiRj_Imag[select] += model_lrimag[select]
        
        second_RiLj_Real[select] += Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * model_rlamp[select] * np.cos(model_rlphas[select] + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] + 2. * (pang1[select] + pang2[select]))
        second_RiLj_Imag[select] += Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * model_rlamp[select] * np.sin(model_rlphas[select] + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] + 2. * (pang1[select] + pang2[select]))
        second_LiRj_Real[select] += Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * model_lramp[select] * np.cos(model_lrphas[select] + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] - 2. * (pang1[select] + pang2[select]))
        second_LiRj_Imag[select] += Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * model_lramp[select] * np.sin(model_lrphas[select] + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] - 2. * (pang1[select] + pang2[select]))
          
        
    dterm_RiLj_Real += Tot_D_iR_amp * llamp * np.cos(Tot_D_iR_phas + llphas + 2. * pang1) + Tot_D_jL_amp * rramp * np.cos(-Tot_D_jL_phas + rrphas + 2. * pang2)
    dterm_RiLj_Imag += Tot_D_iR_amp * llamp * np.sin(Tot_D_iR_phas + llphas + 2. * pang1) + Tot_D_jL_amp * rramp * np.sin(-Tot_D_jL_phas + rrphas + 2. * pang2)
    dterm_LiRj_Real += Tot_D_iL_amp * rramp * np.cos(Tot_D_iL_phas + rrphas - 2. * pang1) + Tot_D_jR_amp * llamp * np.cos(-Tot_D_jR_phas + llphas - 2. * pang2)
    dterm_LiRj_Imag += Tot_D_iL_amp * rramp * np.sin(Tot_D_iL_phas + rrphas - 2. * pang1) + Tot_D_jR_amp * llamp * np.sin(-Tot_D_jR_phas + llphas - 2. * pang2)  


    pol_rl, pol_lr = pol_RiLj_Real + 1j * pol_RiLj_Imag, pol_LiRj_Real + 1j * pol_LiRj_Imag
    pol_q, pol_u = (pol_rl + pol_lr) / 2., -1j * (pol_rl - pol_lr) / 2.
    pol_qamp, pol_qphas, pol_uamp, pol_uphas = np.absolute(pol_q), np.angle(pol_q), np.absolute(pol_u), np.angle(pol_u)
    
    dterm_rl, dterm_lr = dterm_RiLj_Real + 1j * dterm_RiLj_Imag, dterm_LiRj_Real + 1j * dterm_LiRj_Imag
    dterm_q, dterm_u = (dterm_rl + dterm_lr) / 2., -1j * (dterm_rl - dterm_lr) / 2.
    dterm_qamp, dterm_qphas, dterm_uamp, dterm_uphas = np.absolute(dterm_q), np.angle(dterm_q), np.absolute(dterm_u), np.angle(dterm_u)
    
    second_rl, second_lr = second_RiLj_Real + 1j * second_RiLj_Imag, second_LiRj_Real + 1j * second_LiRj_Imag
    second_q, second_u = (second_rl + second_lr) / 2., -1j * (second_rl - second_lr) / 2.
    second_qamp, second_qphas, second_uamp, second_uphas = np.absolute(second_q), np.angle(second_q), np.absolute(second_u), np.angle(second_u)


    if(comp == 'pol'): return pol_qamp, pol_qphas, pol_uamp, pol_uphas
    if(comp == 'dterm'): return dterm_qamp, dterm_qphas, dterm_uamp, dterm_uphas
    if(comp == 'second'): return second_qamp, second_qphas, second_uamp, second_uphas



def pol_deq_inputz(inputz, *p):
    """
    The D-term models for the D-term estimation with instrumental polarization self-calibration.
    
    Args:
        x: dummy parameters (not to be used).
        *p (args): the best-fit parameters args.
    """
    
    nant, pang1, pang2, ant1, ant2, rramp, rrphas, llamp, llphas, model_rlreal, model_rlimag, model_rlamp, model_rlphas, model_lrreal, model_lrimag, model_lramp, model_lrphas = inputz
    
    RiLj_Real = np.zeros(len(pang1))
    RiLj_Imag = np.zeros(len(pang1))
    LiRj_Real = np.zeros(len(pang1))
    LiRj_Imag = np.zeros(len(pang1))
    
    
    dump = np.array(p)
    
    dumreal = dump[2*ant1]
    dumimag = dump[2*ant1 + 1]
    Tot_D_iR_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_iR_phas = np.angle(dumreal + 1j*dumimag)

    dumreal = dump[2*nant + 2*ant2]
    dumimag = dump[2*nant + 2*ant2 + 1]
    Tot_D_jL_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_jL_phas = np.angle(dumreal + 1j*dumimag)
    
    dumreal = dump[2*nant + 2*ant1]
    dumimag = dump[2*nant + 2*ant1 + 1]
    Tot_D_iL_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_iL_phas = np.angle(dumreal + 1j*dumimag)
    
    dumreal = dump[2*ant2]
    dumimag = dump[2*ant2 + 1]
    Tot_D_jR_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_jR_phas = np.angle(dumreal + 1j*dumimag)
    

    RiLj_Real += model_rlreal + \
      Tot_D_iR_amp * Tot_D_jL_amp * model_lramp * np.cos(model_lrphas + Tot_D_iR_phas - Tot_D_jL_phas + 2. * (pang1 + pang2))

    RiLj_Imag += model_rlimag + \
      Tot_D_iR_amp * Tot_D_jL_amp * model_lramp * np.sin(model_lrphas + Tot_D_iR_phas - Tot_D_jL_phas + 2. * (pang1 + pang2))

    LiRj_Real += model_lrreal + \
      Tot_D_iL_amp * Tot_D_jR_amp * model_rlamp * np.cos(model_rlphas + Tot_D_iL_phas - Tot_D_jR_phas - 2. * (pang1 + pang2))

    LiRj_Imag += model_lrimag + \
      Tot_D_iL_amp * Tot_D_jR_amp * model_rlamp * np.sin(model_rlphas + Tot_D_iL_phas - Tot_D_jR_phas - 2. * (pang1 + pang2))
    
        
    RiLj_Real += \
      Tot_D_iR_amp * llamp * np.cos(Tot_D_iR_phas + llphas + 2. * pang1) + \
      Tot_D_jL_amp * rramp * np.cos(-Tot_D_jL_phas + rrphas + 2. * pang2)
    
    RiLj_Imag += \
      Tot_D_iR_amp * llamp * np.sin(Tot_D_iR_phas + llphas + 2. * pang1) + \
      Tot_D_jL_amp * rramp * np.sin(-Tot_D_jL_phas + rrphas + 2. * pang2)

    LiRj_Real += \
      Tot_D_iL_amp * rramp * np.cos(Tot_D_iL_phas + rrphas - 2. * pang1) + \
      Tot_D_jR_amp * llamp * np.cos(-Tot_D_jR_phas + llphas - 2. * pang2)

    LiRj_Imag += \
      Tot_D_iL_amp * rramp * np.sin(Tot_D_iL_phas + rrphas - 2. * pang1) + \
      Tot_D_jR_amp * llamp * np.sin(-Tot_D_jR_phas + llphas - 2. * pang2)  
   

    compv = np.concatenate([LiRj_Real, LiRj_Imag, RiLj_Real, RiLj_Imag])
    
    return compv


def rescale_weight(ant1, ant2, inputsigma, antname, weightfactors, logger):
    
    nant = len(antname)
    
    sigmaant1 = np.concatenate([ant1, ant1, ant1, ant1])
    sigmaant2 = np.concatenate([ant2, ant2, ant2, ant2])
    
    outputweight = 1. / inputsigma ** 2
    for m in range(nant):
        if(antname[m] in weightfactors):
            outputweight[(sigmaant1 == m) | (sigmaant2 == m)] = outputweight[(sigmaant1 == m) | (sigmaant2 == m)] * weightfactors.get(antname[m])
            logger.info('The visibility weights for {:s} station are rescaled by a factor of {:4.2f}.'.format(antname[m], weightfactors.get(antname[m])))
        
    logger.info(' ')
    
    outputsigma = 1. / outputweight ** (1. / 2.)
    
    return outputsigma


def arrange_index(orig_nant, ant1, ant2, logger, antname, printmessage = True):

    nant = np.copy(orig_nant)
    
    dum = 0
    
    removed_Index = []
    
    for m in range(orig_nant):
        if((sum(ant1 == dum) == 0) & (sum(ant2 == dum) == 0)):
            ant1[ant1 > dum] -= 1
            ant2[ant2 > dum] -= 1
            nant -= 1
            removed_Index.append(m)
            if printmessage:
                logger.info('{:s} has no data, the fitting will not solve the D-Terms for it.'.format(antname[m]))
        else:
            dum += 1
    
    if(len(removed_Index) != 0): antname = np.delete(antname, removed_Index)
    
    return nant, antname, removed_Index, ant1, ant2
    
        


def pol_gpcal(direc, outputname, data, calsour, antname, logger, filetype = 'png', printmessage = True, Dbound = 1.0, Pbound = 1.0, manualweight = False, weightfactors = None, fixdterm = False, fixdr = None, fixdl = None, \
              multiproc = True, nproc = 2, colors = None, vplot = False, allplot = False, vplot_title = None, vplot_scanavg = False, vplot_avg_nat = False, tsep = None, \
              resplot = False, markerarr = None):
    
    """
    Estimate the D-terms using instrumental polarization self-calibration.
    
    Args:
        spoliter (int): the number of iteration of instrumental polarization self-calibration.
    """  

    result_DRArr, result_DLArr, result_ant = [], [], []
          
    time, dayarr, sourcearr, pang1, pang2, ant1, ant2, rlreal, rlimag, lrreal, lrimag, rlsigma, lrsigma, rramp, rrphas, llamp, llphas, \
    model_rlreal, model_rlimag, model_lrreal, model_lrimag, model_rlamp, model_rlphas, model_lramp, model_lrphas = \
        np.array(data["time"]), np.array(data["day"]), np.array(data["source"]), np.array(data["pang1"]), np.array(data["pang2"]), np.array(data["ant1"]), np.array(data["ant2"]), \
        np.array(data["rlreal"]), np.array(data["rlimag"]), np.array(data["lrreal"]), np.array(data["lrimag"]), \
        np.array(data["rlsigma"]), np.array(data["lrsigma"]), np.array(data["rramp"]), np.array(data["rrphas"]), np.array(data["llamp"]), np.array(data["llphas"]), \
        np.array(data["model_rlreal"]), np.array(data["model_rlimag"]), np.array(data["model_lrreal"]), np.array(data["model_lrimag"]), \
        np.array(data["model_rlamp"]), np.array(data["model_rlphas"]), np.array(data["model_lramp"]), np.array(data["model_lrphas"])
    
    qamp = np.array(data["qamp"])
    qphas = np.array(data["qphas"])
    qsigma = np.array(data["qsigma"])
    uamp = np.array(data["uamp"])
    uphas = np.array(data["uphas"])
    usigma = np.array(data["usigma"])
    
    nant = len(antname)


    inputx = np.concatenate([pang1, pang1, pang1, pang1])
    inputy = np.concatenate([lrreal, lrimag, rlreal, rlimag])
    inputsigma = np.concatenate([lrsigma, lrsigma, rlsigma, rlsigma])
    
    # Rescale the visibility weights of specific stations if requested.
    if manualweight:
        outputsigma = rescale_weight(ant1, ant2, inputsigma, antname, weightfactors, logger)
    else:
        outputsigma = np.copy(inputsigma)
        if printmessage:
            logger.info('No visibility weight rescaling applied.\n')
    
    
    dumantname = np.copy(antname)
    orig_ant1 = np.copy(ant1)
    orig_ant2 = np.copy(ant2)
    
    orig_nant = np.copy(nant)
    
    
    nant, dumantname, removed_Index, ant1, ant2 = arrange_index(orig_nant, ant1, ant2, logger, antname, printmessage = printmessage)
    
    init = np.zeros(2*2*nant)
    
    # The boundaries of parameters allowed for the least-square fitting.
    lbound = [-Dbound]*(2*2*nant)
    ubound = [Dbound]*(2*2*nant)
    

    if fixdterm:
        for i in range(nant):
            if dumantname[i] in fixdr:
                lbound[2*i] = np.real(fixdr.get(dumantname[i])) - 1e-8
                ubound[2*i] = np.real(fixdr.get(dumantname[i])) + 1e-8
                lbound[2*i+1] = np.imag(fixdr.get(dumantname[i])) - 1e-8
                ubound[2*i+1] = np.imag(fixdr.get(dumantname[i])) + 1e-8
                lbound[2*nant+2*i] = np.real(fixdl.get(dumantname[i])) - 1e-8
                ubound[2*nant+2*i] = np.real(fixdl.get(dumantname[i])) + 1e-8
                lbound[2*nant+2*i+1] = np.imag(fixdl.get(dumantname[i])) - 1e-8
                ubound[2*nant+2*i+1] = np.imag(fixdl.get(dumantname[i])) + 1e-8
                init[2*i] = np.real(fixdr.get(dumantname[i]))
                init[2*i+1] = np.imag(fixdr.get(dumantname[i]))
                init[2*nant+2*i] = np.real(fixdl.get(dumantname[i]))
                init[2*nant+2*i+1] = np.imag(fixdl.get(dumantname[i]))
    
    
    bounds=(lbound,ubound)
    
    
    global inputz
    
    inputz = (nant, pang1, pang2, ant1, ant2, rramp, rrphas, llamp, llphas, model_rlreal, model_rlimag, model_rlamp, model_rlphas, model_lrreal, model_lrimag, model_lramp, model_lrphas)
    
    
    Iteration, pco = curve_fit(pol_deq, inputx, inputy, p0=init, sigma = outputsigma, absolute_sigma = False, bounds = bounds)
        
    error = np.sqrt(np.diag(pco))


    # Restore the original antenna numbers.
    insert_index = []
    
    dum = 0
    for it in removed_Index:
        insert_index.append(2*it - 2*dum)
        insert_index.append(2*it - 2*dum)
        insert_index.append(2*nant + 2*it - 2*dum)
        insert_index.append(2*nant + 2*it - 2*dum)
        dum += 1
        
    Iteration = np.insert(Iteration, insert_index, [0.]*len(insert_index))
    error = np.insert(error, insert_index, [0.]*len(insert_index))
    
    # logger.info('The fitting is completed within {:d} seconds.\n'.format(int(round(time2 - time1))))
    
    
    nant = np.copy(orig_nant)
    
    ant1 = np.copy(orig_ant1)
    ant2 = np.copy(orig_ant2)
    
    
    for m in range(nant):
        result_ant.append(antname[m])
        result_DRArr.append(Iteration[2*m] + 1j*Iteration[2*m+1])
        result_DLArr.append(Iteration[2*nant+2*m] + 1j*Iteration[2*nant+2*m+1])


    
    inputz = (nant, pang1, pang2, ant1, ant2, rramp, rrphas, llamp, llphas, model_rlreal, model_rlimag, model_rlamp, model_rlphas, model_lrreal, model_lrimag, model_lramp, model_lrphas)
    
    # Calculate the reduced chi-square of the fitting.
    dumfit = pol_deq(inputx, *Iteration)

    ydata = np.concatenate([lrreal+1j*lrimag, rlreal+1j*rlimag])
    yfit = np.concatenate([dumfit[0:len(lrreal)]+1j*dumfit[len(lrreal):len(lrreal)*2], dumfit[len(lrreal)*2:len(lrreal)*3]+1j*dumfit[len(lrreal)*3:len(lrreal)*4]])
    ysigma = np.concatenate([lrsigma, rlsigma])
    
    chisq_num = np.sum(np.abs(((ydata - yfit) / ysigma) ** 2))
    chisq_den = (2. * len(ydata))
    # chisq_den = (2. * len(ydata) - float(len(Iteration)))
    
    chisq = chisq_num / chisq_den
    
    
    del inputz
    
    if printmessage:
        logger.info('The reduced chi-square of the fitting is {:5.3f}.\n'.format(chisq))
    
        for m in range(nant):
            logger.info('{:s}: RCP - amplitude = {:6.3f} %, phase = {:7.2f} deg, LCP - amplitude = {:6.3f} %, phase = {:7.2f} deg'.format(antname[m], \
              np.abs(Iteration[2*m] + 1j*Iteration[2*m+1]) * 1e2, np.degrees(np.angle((Iteration[2*m] + 1j*Iteration[2*m+1]))), \
              np.abs(Iteration[2*nant+2*m] + 1j*Iteration[2*nant+2*m+1]) * 1e2, np.degrees(np.angle(Iteration[2*nant+2*m] + 1j*Iteration[2*nant+2*m+1]))))
    
        logger.info(' ')
    
    
    mod_lr, mod_rl = dumfit[0:len(rlreal)] + 1j*dumfit[len(rlreal):len(rlreal)*2], dumfit[len(rlreal)*2:len(rlreal)*3] + 1j*dumfit[len(rlreal)*3:len(rlreal)*4]
    mod_q, mod_u = (mod_rl + mod_lr) / 2., -1j * (mod_rl - mod_lr) / 2.
    mod_qamp, mod_qphas, mod_uamp, mod_uphas = np.absolute(mod_q), np.angle(mod_q), np.absolute(mod_u), np.angle(mod_u)
    
    
        
    parmset = (nant, calsour, sourcearr, pang1, pang2, ant1, ant2, model_rlreal, model_rlimag, model_lrreal, model_lrimag, rramp, rrphas, llamp, llphas)
        
    if allplot:
        mod_pol_qamp, mod_pol_qphas, mod_pol_uamp, mod_pol_uphas = pol_deq_comp('pol', parmset, *Iteration)
        mod_dterm_qamp, mod_dterm_qphas, mod_dterm_uamp, mod_dterm_uphas = pol_deq_comp('dterm', parmset, *Iteration)
        mod_second_qamp, mod_second_qphas, mod_second_uamp, mod_second_uphas = pol_deq_comp('second', parmset, *Iteration)
        allplots = (mod_pol_qamp, mod_pol_qphas, mod_pol_uamp, mod_pol_uphas, mod_dterm_qamp, mod_dterm_qphas, mod_dterm_uamp, mod_dterm_uphas)
    else:
        allplots = None
    
    # Create vplots if requested.
    if vplot:
        
        logger.info('\nCreating vplots for all baselines... It may take some time.')
        
        ifn = np.unique(data.loc[:, "IF"])[0]
        
        parmset = []

        for l in range(len(calsour)):
            for m in range(nant):
                for n in range(nant):
                    if(m==n):
                        continue
                    dumm = m
                    dumn = n
                    if(m>n):
                        dumm = n
                        dumn = m
                        
                        select = (ant1 == dumm) & (ant2 == dumn) & (sourcearr == calsour[l])
                    
                        selected_time, selected_day, selected_qamp, selected_qphas, selected_qsigma, selected_uamp, selected_uphas, selected_usigma, selected_mod_qamp, selected_mod_qphas, selected_mod_uamp, selected_mod_uphas = \
                            time[select], dayarr[select], qamp[select], qphas[select], qsigma[select], uamp[select], uphas[select], usigma[select], \
                            mod_qamp[select], mod_qphas[select], mod_uamp[select], mod_uphas[select]
                            
                        if allplot:
                            selected_mod_pol_qamp = mod_pol_qamp[select]
                            selected_mod_pol_qphas = mod_pol_qphas[select]
                            selected_mod_pol_uamp = mod_pol_uamp[select]
                            selected_mod_pol_uphas = mod_pol_uphas[select]
                            
                            selected_mod_dterm_qamp = mod_dterm_qamp[select]
                            selected_mod_dterm_qphas = mod_dterm_qphas[select]
                            selected_mod_dterm_uamp = mod_dterm_uamp[select]
                            selected_mod_dterm_uphas = mod_dterm_uphas[select]
                            
                            allplots = (selected_mod_pol_qamp, selected_mod_pol_qphas, selected_mod_pol_uamp, selected_mod_pol_uphas, \
                                        selected_mod_dterm_qamp, selected_mod_dterm_qphas, selected_mod_dterm_uamp, selected_mod_dterm_uphas)
                        
                            
                        if multiproc:
                            parmset.append((colors[l], calsour[l], ifn, antname[dumm], antname[dumn], \
                                            selected_day, selected_time, selected_qamp, selected_qphas, selected_qsigma, selected_uamp, selected_uphas, selected_usigma, \
                                            selected_mod_qamp, selected_mod_qphas, selected_mod_uamp, selected_mod_uphas, direc+'gpcal/'+outputname+'pol.vplot.IF'+str(ifn), filetype, \
                                            allplots, vplot_title, vplot_scanavg, vplot_avg_nat, tsep))
                        else:
                            ph.visualplot(calsour[l], ifn, antname[dumm], antname[dumn], \
                                            selected_day, selected_time, selected_qamp, selected_qphas, selected_qsigma, selected_uamp, selected_uphas, selected_usigma, \
                                            selected_mod_qamp, selected_mod_qphas, selected_mod_uamp, selected_mod_uphas, direc+'gpcal/'+outputname+'pol.vplot.IF'+str(ifn), filetype, \
                                            allplots = allplots, title = vplot_title, scanavg = vplot_scanavg, avg_nat = vplot_avg_nat, tsep = tsep, color = colors[l])


        if multiproc:
            ph.visualplot_run(parmset, nproc = nproc)

        logger.info('...completed.\n')
    
    if resplot:
        ph.residualplot(markerarr, colors, dayarr, ifn-1, nant, antname, calsour, \
                        dumfit, time, ant1, ant2, sourcearr, qamp, qphas, uamp, uphas, qsigma, usigma, direc+'gpcal/'+outputname+'pol.resplot.IF'+str(ifn), tsep = 2. / 60., title = outputname, filetype = filetype)
    
    result_ant, result_DRArr, result_DLArr = np.array(result_ant), np.array(result_DRArr), np.array(result_DLArr)
    
    return (Iteration, error), (result_ant, result_DRArr, result_DLArr), (chisq_num, chisq_den, chisq)



