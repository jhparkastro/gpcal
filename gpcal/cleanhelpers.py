#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:59:06 2021

@author: jpark
"""

import os


def cleanqu(direc, data, mask, save, bif, eif, ms, ps, uvbin, uvpower, dynam, shift_x, shift_y, stokes): # Version 1.1
    """
    Perform imaging of Stokes Q and U in Difmap.
    
    Args:
        data (str): the name of the UVFITS file to be CLEANed.
        mask (str): the name of the CLEAN windows to be used.
        save (str): the name of the output Difmap save file.
        log (str): the name of the Difmap log file.
    """

    curdirec = os.getcwd()
    
    os.chdir(direc)
    
    command = "echo @GPCAL_Difmap_v1 %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s | difmap > /dev/null 2>&1" %(data,bif,eif,ms,ps,uvbin,uvpower,mask,shift_x,shift_y,dynam,save,stokes) # Version 1.1!
    os.system(command)

    os.chdir(curdirec)
    

def cleanqu2(args): # Version 1.1
    cleanqu(*args)


