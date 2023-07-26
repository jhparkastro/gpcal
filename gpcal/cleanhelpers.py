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
    
    
    # f = open(direc+'GPCAL_Difmap_v1','w')
    # f.write('observe %1\nmapcolor rainbow, 1, 0.5\nselect %13, %2, %3\nmapsize %4, %5\nuvweight %6, %7\nrwin %8\nshift %9,%10\ndo i=1,100\nclean 100, 0.02, imstat(rms)*%11\nend do\nselect i\nsave %12.%13\nexit')
    # f.close()
        
    
    curdirec = os.getcwd()
    
    os.chdir(direc)
    
    command = "echo @GPCAL_Difmap_v1 %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s | difmap > /dev/null 2>&1" %(data,bif,eif,ms,ps,uvbin,uvpower,mask,shift_x,shift_y,dynam,save,stokes) # Version 1.1!
    os.system(command)
    
    # os.system('rm difmap.log*')
    
    os.chdir(curdirec)
    

def cleanqu2(args): # Version 1.1
    cleanqu(*args)


