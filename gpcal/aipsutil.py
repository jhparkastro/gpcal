#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:18:58 2021

@author: jpark
"""


import numpy as np
import pandas as pd
import os, sys
from os import path

import timeit
import logging
import copy

from AIPS import AIPS
from AIPSTask import AIPSTask
from AIPSData import AIPSUVData, AIPSImage
from Wizardry.AIPSData import AIPSUVData as WAIPSUVData


# Suppress AIPS printing on terminal
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Enable AIPS printing on terminal
def enablePrint():
    sys.stdout = sys.__stdout__
    
    
def runfitld(outname, outclass, datain):
    """
    Run FITLD in AIPS.
    
    Args:
        outname (str): a FITLD parameter outname.
        outclass (str): a FITLD parameter outclass.
        datain (str): a FITLD parameter datain.
    """
    
    data = AIPSUVData(outname, outclass, 1, 1)
    if(data.exists() == True):
        data.clrstat()
        data.zap()
    
    image = AIPSImage(outname, outclass, 1, 1)
    if(image.exists() == True):
        image.clrstat()
        image.zap()
        
    fitld = AIPSTask('fitld')
    fitld.outname = outname
    fitld.outclass = outclass
    dumdatain = datain.split('/')[-1]
    if(len(dumdatain) >= 46):
        newdatain = "/".join(datain.split('/')[:-1]) + '/' + dumdatain[0:45]
        os.system('cp {:} {:}'.format(datain, newdatain))
        fitld.datain = newdatain
    else:
        fitld.datain = datain

    fitld.outseq = 1
    blockPrint()
    fitld.go()    
    enablePrint()
    if(len(dumdatain) >= 46):
        os.system('rm {:}'.format(newdatain))


def runlpcal(inname, inclass, in2class, cnum):
    """
    Run LPCAL in AIPS.
    
    Args:
        inname (str): a FITLD parameter inname.
        inclass (str): a FITLD parameter inclass.
        in2class (str): a FITLD parameter in2class.
        cnum (int): a FITLD parameter in2vers.
    """
    lpcal = AIPSTask('lpcal')
    lpcal.inname = inname
    lpcal.inclass = inclass
    lpcal.indisk = 1
    lpcal.inseq = 1
    if(cnum >= 1):
        lpcal.in2name = inname
        lpcal.in2class = in2class
        lpcal.in2disk = 1
        lpcal.in2vers = 1
        lpcal.in2seq = 1
        if(cnum >= 2):
            lpcal.in2vers = 2
        lpcal.cmethod = 'DFT'
        lpcal.cmodel = 'COMP'
    
    blockPrint()
    lpcal.go()
    enablePrint()


def runccedt(inname, boxfile, cnum, autoccedt):
    """
    Run CCEDT in AIPS.
    
    Args:
        inname (str): a FITLD parameter inname.
        boxfile (str): a FITLD parameter boxfile.
        cnum (int): a FITLD parameter cparm(3).
        autoccedt (boolean): if it is True, then CCEDT splits CLEAN models into sub-models automatically. If it is False, then the boxfile which specifies the locations of sub-model boxes should be provided.
    """
    ccedt = AIPSTask('ccedt')
    ccedt.inname = inname
    ccedt.inclass = 'CMAP'
    ccedt.indisk = 1
    ccedt.inseq = 1
    if (cnum == 1):
        return
    
    if autoccedt:
        ccedt.cparm[3] = cnum
        ccedt.cparm[4] = 1
    else:
        ccedt.nboxes = cnum
        dumboxfile = boxfile.split('/')[-1]
        if(len(dumboxfile) >= 46):
            newboxfile = "/".join(boxfile.split('/')[:-1]) + '/' + dumboxfile[0:45]
            os.system('cp {:} {:}'.format(boxfile, newboxfile))
            ccedt.boxfile = newboxfile
        else:
            ccedt.boxfile = boxfile
        ccedt.nccbox = -cnum
    
    blockPrint()
    ccedt.go()
    enablePrint()
    
    if not autoccedt:
        if(len(dumboxfile) >= 46):
            os.system('rm {:}'.format(newboxfile))
        

def runprtab(data, outprint):
    """
    Run PRTAB in AIPS.
    
    Args:
        data (ParselTongue UVData): an input ParselTongue UVData.
        outprint (str): the output filename of the D-terms in the antenna table.
    """
    prtab = data.table('AN', 1)

    anname = np.empty([0])
    DRArr = np.empty([0])
    DLArr = np.empty([0])
    
    ifnum = data.header.naxis[3]
    
    for row in prtab:
        anname = np.append(anname, np.repeat(row.anname.replace(' ', ''), ifnum*2))
        DRArr = np.append(DRArr, row.polcala)
        DLArr = np.append(DLArr, row.polcalb)
    
    
    df = pd.DataFrame(DRArr.transpose())
    df['antennas'] = np.array(anname)
    df['DRArr'] = np.array(DRArr)
    df['DLArr'] = np.array(DLArr)
    del df[0]
    
    df.to_csv(outprint, sep = "\t")


def runtbout(inname, inclass, outtext):
    """
    Run TBOUT in AIPS.
    
    Args:
        inname (str): a TBOUT parameter inname.
        inclass (str): a TBOUT parameter inclass.
        outtext (str): the filename of the antenna table output.
    """
    tbout = AIPSTask('tbout')
    tbout.inname = inname
    tbout.inclass = inclass
    tbout.inseq = 1
    tbout.indisk = 1
    tbout.inext = 'AN'
    tbout.invers = 1
    dumouttext = outtext.split('/')[-1]
    if(len(dumouttext) >= 46):
        newouttext = "/".join(outtext.split('/')[:-1]) + '/' + dumouttext[0:45]
        tbout.outtext = newouttext
    else:
        tbout.outtext = outtext
    tbout.docrt = -3
    blockPrint()
    tbout.go()
    enablePrint()
    
    if(len(dumouttext) >= 46):
        os.system('mv {:} {:}'.format(newouttext, outtext))
 

def runcalib(inname, inclass, in2name, in2class, outclass, solint, soltype, solmode, weightit, indisk = 1, inseq = 1, in2disk = 1, in2seq = 1, snver = 1, calsour = None):
    """
    Run CALIB in AIPS.
    
    Args:
        inname (str): a CALIB parameter inname.
        inclass (str): a CALIB parameter inclass.
        outclass (str): a CALIB parameter outclass.
        solint (float): a CALIB parameter solint.
        soltype (str): a CALIB parameter soltype.
        solmode (str): a CALIB parameter solmode.
        weightit (int): a CALIB parameter weightit.
    """
    calib = AIPSTask('calib')
    calib.inname = inname
    calib.inclass = inclass
    calib.indisk = indisk
    calib.inseq = inseq
    if(calib.in2name == 'None'):
        calib.in2name = inname
    else:
        calib.in2name = in2name
    calib.in2class = in2class
    calib.in2disk = in2disk
    calib.in2seq = in2seq
    calib.invers = 1
    calib.outclass = outclass
    calib.cmethod = 'DFT'
    calib.cmodel = 'COMP'
    calib.solint = solint
    calib.refant = 0
    if(solmode == 'a&p'):
        calib.aparm[1] = 4
    else:
        calib.aparm[1] = 3
    calib.aparm[7] = 0.1
    calib.weightit = weightit
    calib.soltype = soltype
    calib.solmode = solmode
    calib.snver = snver
    if(calsour != None):
        calib.calsour[1] = calsour
    
    blockPrint()
    calib.go()
    enablePrint()
    
                        
def runsplit(inname, inclass, inseq = 1, indisk = 1, docal = -1, gainuse = 1, sources = None, bif = 0, eif = 0, bchan = 0, echan = 0, outclass = 'SPLIT', outseq = 1, dopol = 3, aparm = 0):
    """
    Run SPLIT in AIPS to apply the D-terms.
    
    Args:
        inname (str): a SPLIT parameter inname.
        inclass (str): a SPLIT parameter inclass.
    """
    split = AIPSTask('split')
    split.inname = inname
    split.inclass = inclass
    split.inseq = inseq
    split.indisk = indisk
    split.docal = docal
    split.gainuse = gainuse
    if(sources != None):
        split.sources[1] = sources
    split.bif = bif
    split.eif = eif
    split.bchan = bchan
    split.echan = echan
    split.outclass = outclass
    split.outseq = outseq
    split.dopol = dopol
    split.aparm[1] = aparm
    blockPrint()
    split.go()
    enablePrint()



def runsplat(inname, inclass, inseq = 1, indisk = 1, outclass = 'SPLAT', docal = -1, gainuse = 1):
    """
    Run SPLIT in AIPS to apply the D-terms.
    
    Args:
        inname (str): a splat parameter inname.
        inclass (str): a splat parameter inclass.
    """
    splat = AIPSTask('splat')
    splat.inname = inname
    splat.inclass = inclass
    splat.inseq = inseq
    splat.indisk = indisk
    splat.outclass = outclass
    splat.docal = docal
    splat.gainuse = gainuse
    blockPrint()
    splat.go()
    enablePrint()



def runindxr(inname, inclass, inseq, indisk, solint):
    indxr = AIPSTask('indxr')
    indxr.inname = inname
    indxr.inclass = inclass
    indxr.inseq = inseq
    indxr.indisk = indisk
    indxr.cparm[3] = solint
    blockPrint()
    indxr.go()
    enablePrint()


def runmulti(inname, inclass, inseq, indisk):
    """
    Run MULTI in AIPS for the EVPA calibration.
    
    Args:
        inname (str): a MULTI parameter inname.
        inclass (str): a MULTI parameter inclass.
    """
    multi = AIPSTask('multi')
    multi.inname = inname
    multi.inclass = inclass
    multi.indisk = indisk
    multi.inseq = inseq
    multi.outname = inname
    blockPrint()
    multi.go()
    enablePrint()


def rundbcon(source1, inclass, inseq, indisk, source2, in2class, in2seq, in2disk, outname, outclass = 'DBCON', outseq = 1, fqcenter = -1):
        
    dbcon = AIPSTask('dbcon')
    dbcon.inname = source1
    dbcon.inclass = inclass
    dbcon.inseq = inseq
    dbcon.indisk = indisk
    
    dbcon.in2name = source2
    dbcon.in2class = in2class
    dbcon.in2seq = in2seq
    dbcon.in2disk = in2disk
    
    dbcon.outname = outname
    dbcon.outclass = outclass
    dbcon.outseq = outseq
    dbcon.fqcenter = fqcenter
    
    blockPrint()
    dbcon.go()
    enablePrint()


def runclcal(inname, inclass, indisk, inseq, snver, interpol = 'SELF', gainver = 0, gainuse = 0):
    """
    Run CLCOR in AIPS for the EVPA calibration.
    
    Args:
        inname (str): a CLCOR parameter inname.
        clcorprm (list): the list of floats of the amounts of the RCP and LCP phase offset at the reference antenna.
    """
    clcal = AIPSTask('clcal')
    clcal.inname = inname
    clcal.inclass = inclass
    clcal.indisk = indisk
    clcal.inseq = inseq
    clcal.interpol = interpol
    clcal.snver = snver
    clcal.gainver = gainver
    clcal.gainuse = gainuse
    blockPrint()
    clcal.go()
    enablePrint()
    
    

def runclcor(inname, inclass, indisk, inseq, clcorprm):
    """
    Run CLCOR in AIPS for the EVPA calibration.
    
    Args:
        inname (str): a CLCOR parameter inname.
        clcorprm (list): the list of floats of the amounts of the RCP and LCP phase offset at the reference antenna.
    """
    clcor = AIPSTask('clcor')
    clcor.inname = inname
    clcor.inclass = inclass
    clcor.indisk = indisk
    clcor.inseq = inseq
    clcor.stokes = 'L'
    clcor.opcode = 'PHAS'
    for i in range(len(clcorprm)):
        clcor.clcorprm[i+1] = float(clcorprm[i])
    blockPrint()
    clcor.go()
    enablePrint()

    
    
def runsplitpang(inname):
    """
    Run SPLIT in AIPS for the EVPA calibration.
    
    Args:
        inname (str): a SPLIT parameter inname.
    """             
    split = AIPSTask('split')
    split.inname = inname
    split.inclass = 'MULTI'
    split.indisk = 1
    split.inseq = 1
    split.dopol = -1
    split.outclass = 'PANG'
    split.docalib = 1
    split.gainuse = 2
    blockPrint()
    split.go()
    enablePrint()       


def runuvsub(inname, inclass, in2class, invers, outseq):
    """
    Run UVSUB in AIPS.
    
    Args:
        inname (str): a UVSUB parameter inname.
        inclass (str): a UVSUB parameter inclass.
        in2class (str): a UVSUB parameter in2class.
        invers (int): a UVSUB parameter invers.
        outseq (int): a UVSUB parameter outseq.
    """      
    uvsub = AIPSTask('uvsub')
    uvsub.inname = inname
    uvsub.inclass = inclass
    uvsub.indisk = 1
    uvsub.inseq = 1
    uvsub.in2name = inname
    uvsub.in2class = in2class
    uvsub.in2disk = 1
    uvsub.in2seq = 1
    uvsub.outseq = outseq
    uvsub.cmethod = 'DFT'
    uvsub.cmodel = 'COMP'
    uvsub.opcode = 'MODL'
    uvsub.invers = invers
    
    blockPrint()
    uvsub.go()
    enablePrint()



def runuvcop(inname, inclass, inseq = 1, indisk = 1, outname = None, outclass = None, fqcenter = -1):
    """
    Run UVSUB in AIPS.
    
    Args:
        inname (str): a UVSUB parameter inname.
        inclass (str): a UVSUB parameter inclass.
        in2class (str): a UVSUB parameter in2class.
        invers (int): a UVSUB parameter invers.
        outseq (int): a UVSUB parameter outseq.
    """      
    uvcop = AIPSTask('uvcop')
    uvcop.inname = inname
    uvcop.inclass = inclass
    uvcop.indisk = indisk
    uvcop.inseq = inseq
    if outname != None:
        uvcop.outname = outname
    if outclass != None:
        uvcop.outclass = outclass
    uvcop.fqcenter = fqcenter
    
    blockPrint()
    uvcop.go()
    enablePrint()




def runfittp(inname, inclass, dataout):
    """
    Run FITTP in AIPS.
    
    Args:
        inname (str): a FITTP parameter inname.
        inclass (str): a FITTP parameter inclass.
        dataout (str): the FITTP output filename.
    """
    
    if path.exists(dataout):
        os.system('rm {:}'.format(dataout))
        
    fittp = AIPSTask('fittp')
    fittp.inname = inname
    fittp.inclass = inclass
    fittp.indisk = 1
    fittp.inseq = 1
    dumdataout = dataout.split('/')[-1]
    if(len(dumdataout) >= 46):
        newdataout = "/".join(dataout.split('/')[:-1]) + '/' + dumdataout[0:45]
        fittp.dataout = newdataout
    else:
        fittp.dataout = dataout
    blockPrint()
    
    if path.exists(fittp.dataout):
        os.system('rm ' + fittp.dataout)
        
    fittp.go()
    if(len(dumdataout) >= 46):
        os.system('mv {:} {:}'.format(newdataout, dataout))
    enablePrint()


def runtacop(inname, inclass, outname, outclass):
    """
    Run TACOP in AIPS.
    
    Args:
        inname (str): a TACOP parameter inname.
        inclass (str): a TACOP parameter inclass.
        outname (str): a TACOP parameter outname.
        outclass (str): a TACOP parameter outclass.
    """
    tacop = AIPSTask('tacop')
    tacop.inname = inname
    tacop.inclass = inclass
    tacop.inseq = 1
    tacop.indisk = 1
    tacop.inext = 'AN'
    tacop.invers = 1
    tacop.outname = outname
    tacop.outclass = outclass
    blockPrint()
    tacop.go()
    enablePrint()
    


def freqavg(datain, dataout, logger = None): # Version 1.1

    """
    Run SPLIT in AIPS to apply the D-terms.
    
    Args:
        inname (str): a SPLIT parameter inname.
        inclass (str): a SPLIT parameter inclass.
    """
    
    if not (logger == None):
        logger.info('Averaging data over frequency... \n Input file: {:} \n Output file: {:}'.format(datain, dataout))
        
    data = AIPSUVData('AVG', 'EDIT', 1, 1)
    if(data.exists() == True):
        data.zap()
        
    runfitld('AVG', 'EDIT', datain)
    
    split = AIPSUVData('AVG', 'SPLIT', 1, 1)
    if(split.exists() == True):
        split.zap()
        
    split = AIPSTask('split')
    split.inname = 'AVG'
    split.inclass = 'EDIT'
    split.outclass = 'SPLIT'
    split.indisk = 1
    split.aparm[1] = 3
    blockPrint()
    split.go()
    enablePrint()
    
    runfittp('AVG', 'SPLIT', dataout)
    
    data = AIPSUVData('AVG', 'EDIT', 1, 1)
    
    data.zap()
    
    split = AIPSUVData('AVG', 'SPLIT', 1, 1)
    
    split.zap()
