

import sys, os
from os import path

import pandas as pd
import numpy as np
from numpy.linalg import inv

import timeit
import logging
import copy
import glob

from AIPS import AIPS
from AIPSTask import AIPSTask
from AIPSData import AIPSUVData, AIPSImage
from Wizardry.AIPSData import AIPSUVData as WAIPSUVData

from astropy.coordinates import EarthLocation
import astropy.time as at

from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import aipsutil as au
import obshelpers as oh
import cleanhelpers as ch
import plothelpers as ph
import polsolver as ps

import gc
import psutil

from astropy.io import fits

from multiprocessing import cpu_count, Pool

from IPython import embed
        
    
class channelcal(object):
    """
    This is a class to calibrate instrumental polarization in VLBI data and produce D-term corrected UVFITS files.
    
    Attributes:
        aips_userno (int): AIPS user ID for ParselTongue
        direc (str): the working directory where input *.uvf and *.fits files are located.
        dataname (str): the data name. The input files should have the names like dataname.sourcename.uvf and dataname.sourcename.fits (e.g., bm413i.OJ287.uvf).
        calsour (list): the list of calibrators which will be used for an initial D-term estimation using the similarity assumption (e.g., [OJ287, OQ208, 3C345]).
        source (list): the list of sources to which the best-fit D-terms will be applied (e.g., [OJ287, OQ208, 3C345, M87]).
        cnum (list): the list of the number of CLEAN sub-models for calsour (e.g., [3, 0, 3]).
        autoccedt (list): the list of booleans specifying whether the sub-model division will be done automatically or manually (e.g., [False, False, True]).
        
        Dbound (float): the boundary of D-terms allowed for the least-square fitting. The default value is 1.0, which means that the real and imaginary parts of D-terms are allowed to vary within the range of (-1, 1) for the fitting.
        Pbound (float): the boundary of source-polarization terms allowed for the least-square fitting.
        
        outputname (str): the name of the output files. If not specified, the output name will be the same as the dataname.
        drange (float): the range of D-term plots on the complex plane in units of percent. If not specified, then the range will be determined automatically.
        
        selfcal (boolean): if it is True, then additional self-calibration with CALIB in AIPS will be performed. This is recommended when the input UVFITS files are self-calibrated in Difmap, which assumes that the gains of two polarizations are the same.
        solint (float): the solution interval of CALIB. See the AIPS help file for CALIB for more details.
        solmode (str): a CALIB parameter. See the AIPS help file for CALIB for more details.
        soltype (str): a CALIB parameter. See the AIPS help file for CALIB for more details.
        weightit (str): a CALIB parameter. See the AIPS help file for CALIB for more details.
        
        zblcal (boolean): if it is True, then the zero-baseline D-term estimation will be performed. These D-terms will be fixed for the fitting for the rest of the arrays.
        zblcalsour (list): the list of calibrators which will be used for the zero-baseline D-term estimation. Multiple calibrators can be used.
        zblant (list): the list of tuples specifying the zero-baselines. If multiple baselines are given, then all the D-terms comprising those baselines will be determined by assuming the same D-terms and source-polarization terms (e.g., [('AA', 'AP'), ('JC', 'SM')]).
        
        fixdterm (boolean): if it is True, then the D-terms of the specified antennas will be fixed to be certain values for an initial D-term estimation using the similarity assumption.
        pol_fixdterm (boolean): same as fixdterm but for the fitting using instrumental polarization self-calibration.
        fixdr (dictionary): the dictionary which specifies the RCP D-terms of some antennas that will be fixed for fitting. The key and value should be the antenna name and the complex RCP D-term, respectively (e.g., {"BR": 0. + 1j*2., "FD": 1. - 1j*5.}).
        fixdl (dictionary): same as fixdr but for LCP.
        transferdterm (dictionary): if it is True, then the D-terms of some antennas determined by an initial D-term estimation using the similarity assumption will be fixed for the following D-term estimation with instrumental polarization self-calibration.
        transferdtermant (list): the list of antennas for which transferdterm will be applied.
        
        selfpol (boolean): if it is True, then GPCAL performs additional D-term estimation using instrumental polarization self-calibration.
        polcalsour (list): the list of calibrators which will be used for additional D-term estimation using instrumental polarization self-calibration. This list does not have to be the same as calsour.
        selfpoliter (int): the number of iterations of instrumental polarization self-calibration.
        ms (int): mapsize for CLEAN in Difmap.
        ps (float): pixelsize for CLEAN in Difmap.
        uvbin (int): bin_size of uvweight in Difmap.
        uvpower (int): error_power of uvweight in Difmap.
        dynam (float): cutoff of CLEAN in Difmap. GPCAL will perform CLEAN until the peak intensity within the CLEAN windows reach the map rms-noise times this variable.
        
        manualweight (boolean): if it is True, then visibility weights of specified antennas are scaled by certain factors.
        weightfactors (dictionary): the dictionary of the weight scaling factors. The key and value should be the antenna name and the scaling factors, respectively (e.g., {"Y":0.1, "MK":5.0}).
        
        lpcal (boolean): if it is True, then the D-terms of individual sources in calsour derived from LPCAL are shown together with the GPCAL D-terms in the D-term plots.
        
        vplot (boolean): if it is True, then vplots are created.
        resplot (boolean): if it is True, then the fitting residual plots are created.
        parplot (boolean): if it is True, then the field-rotation angle plots are created.
        allplot (boolean): if it is True, then different terms (source-polarization and D-terms) of the best-fit models are shown in the vplots.
        dplot_IFsep (boolean): if it is True, then the plots showing the best-fit D-terms are created for each IF separately.
        tsep (float): the minimal time gap to define scan separation in units of hour. The default is 2 minutes.
        
        filetype (str): the extension of the name of all the plots created. The default is pdf.
        
        aipslog (boolean): if it is True, then the output log will contain messages created by AIPS.
        difmaplog (boolean): if it is True, then the output log will contain messages created by Difmap.
        
    Returns:
        gpcal.polcal object
            
    """
    def __init__(self, aips_userno, direc, dataname, channel_calsour, channel_source, Dbound = 1.0, Pbound = np.inf, \
                 clcorprm = None, outputname = None, drange = None, doevpacal = False, \
                 inname = None, inclass = None, inseq = None, indisk = None, snver = None, clver = None, channel_calsour_models = None, channel_source_models = None, \
                 channel_calsour_uvf = None, channel_calsour_win = None, channel_source_uvf = None, channel_source_win = None, \
                 imodelname = None, qmodelname = None, umodelname = None, selfcal_doclean = True, calsour_doclean = True, uvfname = None, doselfcal = True, \
                 solint = None, solmode = None, soltype = None, weightit = None, fixdterm = False, fixdr = None, fixdl = None, polcal_unpol = None, remove_weight_outlier_threshold = 10000, \
                 ms = None, ps = None, uvbin = None, uvpower = None, shift_x = 0, shift_y = 0, dynam = None, clean_IF_combine = False, \
                 manualweight = False, weightfactors = None, tsep = 2./60., filetype = 'pdf', aipslog = True, difmaplog = True, multiproc = True, nproc = 2):
        
        
        self.aips_userno = aips_userno
        
        AIPS.userno = self.aips_userno
        
        self.direc = direc
        
        # If direc does not finish with a slash, then append it.
        if(self.direc[-1] != '/'): self.direc = self.direc + '/'

        
        # Create a folder named 'gpcal' in the working directory if it does not exist.
        if(os.path.exists(self.direc+'gpcal') == False):
            os.system('mkdir ' + self.direc+'gpcal') 
        
        self.dataname = dataname
        
        self.doevpacal = doevpacal
        self.Dbound = Dbound
        self.Pbound = Pbound
        
        self.inname = inname
        self.inclass = inclass
        self.indisk = indisk
        self.inseq = inseq
        
        self.snver = snver
        self.clver = clver
        
        if(snver == None):
            self.snver = 0
        if(clver == None):
            self.clver = 1
        
        self.channel_calsour_uvf = channel_calsour_uvf
        self.channel_calsour_win = channel_calsour_win
        self.channel_calsour = channel_calsour
        self.channel_calsour_models = channel_calsour_models
        
        self.channel_source_uvf = channel_source_uvf
        self.channel_source_win = channel_source_win
        self.channel_source = channel_source
        self.channel_source_models = channel_source_models
        
        self.imodelname = imodelname
        self.qmodelname = qmodelname
        self.umodelname = umodelname
        
        self.selfcal_doclean = selfcal_doclean
        self.calsour_doclean = calsour_doclean
        
        self.doselfcal = doselfcal
        
        self.clcorprm = clcorprm
        
        self.remove_weight_outlier_threshold = remove_weight_outlier_threshold
        
        if(self.dataname[-1] != '.'): self.dataname = self.dataname + '.'
        
        self.outputname = copy.deepcopy(outputname)
        if(self.outputname == None):
            self.outputname = copy.deepcopy(self.dataname)
        else:
            if(self.outputname[-1] != '.'): self.outputname = self.outputname + '.'
        
        self.solint = solint
        self.solmode = solmode
        self.soltype = soltype
        self.weightit = weightit
        
        self.fixdterm = fixdterm
        self.fixdr = copy.deepcopy(fixdr)
        self.fixdl = copy.deepcopy(fixdl)
        
        self.polcal_unpol = polcal_unpol
        if(self.polcal_unpol == None): 
            self.polcal_unpol = [False] * len(self.channel_calsour)
        
        self.clean_IF_combine = clean_IF_combine
        
        self.ms = ms
        self.ps = ps
        self.uvbin = uvbin
        self.uvpower = uvpower
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.dynam = dynam
        
        self.manualweight = manualweight
        self.weightfactors = weightfactors
        
        self.tsep = tsep
        self.filetype = filetype
        
        self.aipslog = aipslog
        self.difmaplog = difmaplog
        
        self.aipstime = 0.
        self.difmaptime = 0.
        self.gpcaltime = 0.
        
        self.multiproc = multiproc
        self.nproc = nproc
        

        # Create a list of colors
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf'] + \
                       ['blue', 'red', 'orange', 'steelblue', 'green', 'slategrey', 'cyan', 'royalblue', 'purple', 'blueviolet', 'darkcyan', 'darkkhaki', 'magenta', 'navajowhite', 'darkred'] + \
                           ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf'] + \
                                          ['blue', 'red', 'orange', 'steelblue', 'green', 'slategrey', 'cyan', 'royalblue', 'purple', 'blueviolet', 'darkcyan', 'darkkhaki', 'magenta', 'navajowhite', 'darkred']
        # Define a list of markers
        self.markerarr = ['o', '^', 's', '<', 'p', '*', 'X', 'P', 'D', 'v', 'd', 'x'] * 5
        
        
        # Setup of logging
        if path.exists(direc+'gpcal/'+dataname+'gpcal.log'):
            os.system('rm ' + direc+'gpcal/'+dataname+'gpcal.log')
        self.logfile = direc+'gpcal/'+dataname+'gpcal.log'
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        logs = logging.StreamHandler()
        logformat = logging.Formatter('%(message)s')
        logs.setFormatter(logformat)
        if not len(self.logger.handlers):
            self.logger.addHandler(logs)
            logf = logging.FileHandler(self.logfile)
            self.logger.addHandler(logf)
        

    def get_channel_data(self, data):
    
        pdkey = ["IF", "channel", "year", "month", "day", "time", "source", "ant1", "ant2", "u", "v", "pang1", "pang2", \
                  "rrreal", "rrimag", "rrsigma", "llreal", "llimag", "llsigma", "rlreal", "rlimag", "rlsigma", "lrreal", "lrimag", "lrsigma", \
                  "rramp", "rrphas", "rramp_sigma", "rrphas_sigma", "llamp", "llphas", "llamp_sigma", "llphas_sigma", "rlamp", "rlphas", "rlamp_sigma", "rlphas_sigma", "lramp", "lrphas", "lramp_sigma", "lrphas_sigma", \
                  "qamp", "qphas", "qamp_sigma", "qphas_sigma", "qsigma", "uamp", "uphas", "uamp_sigma", "uphas_sigma", "usigma"] # Version 1.1!
    
            
        self.data = pd.DataFrame(columns = pdkey)       
        
        self.year, self.month, self.day = oh.get_obsdate(data)
                
        sutable = data.table('SU', 1)
        
        wholesource, obsra, obsdec, sourceid = [], [], [], []
        
        for row in sutable:
            wholesource.append(row.source.replace(' ', ''))
            sourceid.append(row.id__no)
            obsra.append(row.raobs)
            obsdec.append(row.decobs)
            
        wholesource = np.array(wholesource)
        sourceid = np.array(sourceid)
        
        calsourarr = np.array(self.channel_calsour)
        obsra = np.array(obsra)
        obsdec = np.array(obsdec)
        
        dumobsra = []
        dumobsdec = []
        
        for i in range(len(calsourarr)):
            dumobsra.append(obsra[wholesource == calsourarr[i]])
            dumobsdec.append(obsdec[wholesource == calsourarr[i]])
        
        self.obsra = dumobsra
        self.obsdec = dumobsdec
        
        
        self.antname, self.antx, self.anty, self.antz, self.antmount, self.f_par, self.f_el, self.phi_off = oh.get_antcoord(data)
        
        self.ifnum, self.freq = oh.get_freqinfo(data)
        
        self.lonarr, self.latarr, self.heightarr = oh.coord(self.antname, self.antx, self.anty, self.antz)

        self.nant = len(self.antname)

        
        
        f = open(self.direc + 'GPCAL_Difmap_uvsub','w')
        f.write('observe %1\nselect i\nmapsize %2, %3\nrmodel %4\nsave %5\nrmodel %6\nsave %7\nexit')
        f.close()
        
        curdirec = os.getcwd()
        
        
        timearr, ifarr, chanarr, sourcearr, ant1arr, ant2arr, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        
        qrealarr, qimagarr, urealarr, uimagarr = [], [], [], []
        
        self.logger.info(' ')
        
        for l in range(len(self.channel_calsour)):
            calsour = self.channel_calsour[l]
            
            self.logger.info('Getting the whole-channel data for {:}...'.format(calsour))
            
            split = AIPSUVData(calsour, 'SPLIT', 1, 1)
            
            if(split.exists() == True):
                split.clrstat()
                split.zap()
            
            au.runsplit(self.inname, self.inclass, inseq = self.inseq, indisk = self.indisk, docal = 1, gainuse = self.newclver, sources = calsour, dopol = -1)
                        
            split = WAIPSUVData(calsour, 'SPLIT', 1, 1)
                
            time = np.zeros(len(split))
            dumu = np.zeros(len(split))
            dumv = np.zeros(len(split))
            
            ant1 = np.zeros(len(split), dtype = int)
            ant2 = np.zeros(len(split), dtype = int)
            
            dumsourcearr = np.chararray(len(split), itemsize = 16)
            
            dumvis = np.zeros((len(split), self.ifnum, self.nchan, 4, 3))
            qvis = np.zeros((len(split), self.ifnum, self.nchan, 4, 3))
            uvis = np.zeros((len(split), self.ifnum, self.nchan, 4, 3))
                    
            dum = 0
            for row in split:
                time[dum] = row.time
                dumu[dum] = row.uvw[0]
                dumv[dum] = row.uvw[1]
                dumsourcearr[dum] = calsour
                ant1[dum] = row.baseline[0]
                ant2[dum] = row.baseline[1]
                dumvis[dum,:,:,:,:] = row.visibility
                
                dum += 1
            
            
            
            if not self.polcal_unpol[l]:
                os.chdir(self.direc)
                command = "echo @GPCAL_Difmap_uvsub %s,%s,%s,%s,%s,%s,%s | difmap > /dev/null 2>&1" %(self.channel_calsour_uvf[l], self.ms[l], self.ps[l], self.qmodelname[l][0], \
                                                                                      self.qmodelname[l][0].replace('.mod', '.uvsub'), self.umodelname[l][0], self.umodelname[l][0].replace('.mod', '.uvsub'))
                os.system(command)
                
                os.chdir(curdirec)
                    
                
                qmap = AIPSImage(calsour, 'QMAP', 1, 1)
                if(qmap.exists() == True):
                    qmap.clrstat()
                    qmap.zap()
                    
                umap = AIPSImage(calsour, 'UMAP', 1, 1)
                if(umap.exists() == True):
                    umap.clrstat()
                    umap.zap()
                
                au.runfitld(calsour, 'QMAP', self.direc + self.qmodelname[l][0].replace('.mod', '.uvsub.fits'))
                au.runfitld(calsour, 'UMAP', self.direc + self.umodelname[l][0].replace('.mod', '.uvsub.fits'))
                
                    
                uvsub_q = AIPSUVData(calsour, 'UVSUB', 1, 1)
                if(uvsub_q.exists() == True):
                    uvsub_q.clrstat()
                    uvsub_q.zap()
                
                uvsub_u = AIPSUVData(calsour, 'UVSUB', 1, 2)
                if(uvsub_u.exists() == True):
                    uvsub_u.clrstat()
                    uvsub_u.zap()
                
                
                au.runuvsub(calsour, 'SPLIT', 'QMAP', 1, 1)
                au.runuvsub(calsour, 'SPLIT', 'UMAP', 1, 2)
                
                
                uvsub_q = WAIPSUVData(calsour, 'UVSUB', 1, 1)
                uvsub_u = WAIPSUVData(calsour, 'UVSUB', 1, 2)
            
                    
                dum = 0
                for row in uvsub_q:
                    qvis[dum,:,:,:,:] = row.visibility
                    
                    dum += 1
                
                dum = 0
                for row in uvsub_u:
                    uvis[dum,:,:,:,:] = row.visibility
                    
                    dum += 1
            
            else:
                qvis = qvis.astype('complex64')
                uvis = uvis.astype('complex64')
            
                
            indnum = len(ant1)
            
            time = np.tile(time, self.ifnum * self.nchan)
            dumsourcearr = np.tile(dumsourcearr, self.ifnum * self.nchan)
            
            ant1 = np.tile(ant1, self.ifnum * self.nchan)
            ant2 = np.tile(ant2, self.ifnum * self.nchan)
            
            dumu = np.tile(dumu, self.ifnum * self.nchan)
            dumv = np.tile(dumv, self.ifnum * self.nchan)
            
            
            # ravel order = time first, then IF, then channel
            rrreal = dumvis[:,:,:,0,0].ravel(order = 'F')
            rrimag = dumvis[:,:,:,0,1].ravel(order = 'F')
            rrweight = dumvis[:,:,:,0,2].ravel(order = 'F')
            
            llreal = dumvis[:,:,:,1,0].ravel(order = 'F')
            llimag = dumvis[:,:,:,1,1].ravel(order = 'F')
            llweight = dumvis[:,:,:,1,2].ravel(order = 'F')
            
            rlreal = dumvis[:,:,:,2,0].ravel(order = 'F')
            rlimag = dumvis[:,:,:,2,1].ravel(order = 'F')
            rlweight = dumvis[:,:,:,2,2].ravel(order = 'F')
                    
            lrreal = dumvis[:,:,:,3,0].ravel(order = 'F')
            lrimag = dumvis[:,:,:,3,1].ravel(order = 'F')
            lrweight = dumvis[:,:,:,3,2].ravel(order = 'F')
            
            qreal = qvis[:,:,:,0,0].ravel(order = 'F')
            qimag = qvis[:,:,:,0,1].ravel(order = 'F')
            
            ureal = uvis[:,:,:,0,0].ravel(order = 'F')
            uimag = uvis[:,:,:,0,1].ravel(order = 'F')
                        
            
            dumchanarr = np.zeros(len(ant1)).astype(int)
            dumifarr = np.zeros(len(ant1)).astype(int)
            
            for chan in range(self.nchan):
                for ifn in range(self.ifnum):    
                    dumchanarr[indnum*ifn + indnum*self.ifnum*chan : indnum*(ifn+1) + indnum*self.ifnum*chan] = chan + 1
                    dumifarr[indnum*ifn + indnum*self.ifnum*chan : indnum*(ifn+1) + indnum*self.ifnum*chan] = ifn + 1
            
            
            
            timearr = timearr + time.tolist()
            ifarr = ifarr + dumifarr.tolist()
            chanarr = chanarr + dumchanarr.tolist()
            sourcearr = sourcearr + dumsourcearr.tolist()
            ant1arr = ant1arr + ant1.tolist()
            ant2arr = ant2arr + ant2.tolist()
            uarr = uarr + dumu.tolist()
            varr = varr + dumv.tolist()
            rrrealarr = rrrealarr + rrreal.tolist()
            rrimagarr = rrimagarr + rrimag.tolist()
            rrweightarr = rrweightarr + rrweight.tolist()
            llrealarr = llrealarr + llreal.tolist()
            llimagarr = llimagarr + llimag.tolist()
            llweightarr = llweightarr + llweight.tolist()
            rlrealarr = rlrealarr + rlreal.tolist()
            rlimagarr = rlimagarr + rlimag.tolist()
            rlweightarr = rlweightarr + rlweight.tolist()
            lrrealarr = lrrealarr + lrreal.tolist()
            lrimagarr = lrimagarr + lrimag.tolist()
            lrweightarr = lrweightarr + lrweight.tolist()
            qrealarr = qrealarr + qreal.tolist()
            qimagarr = qimagarr + qimag.tolist()
            urealarr = urealarr + ureal.tolist()
            uimagarr = uimagarr + uimag.tolist()
            
            
            split = AIPSUVData(calsour, 'SPLIT', 1, 1)
            if(split.exists() == True):
                split.clrstat()
                split.zap()
            
            uvsub_q = AIPSUVData(calsour, 'UVSUB', 1, 1)
            if(uvsub_q.exists() == True):
                uvsub_q.clrstat()
                uvsub_q.zap()
            
            uvsub_u = AIPSUVData(calsour, 'UVSUB', 1, 2)
            if(uvsub_u.exists() == True):
                uvsub_u.clrstat()
                uvsub_u.zap()
            
            qmap = AIPSImage(calsour, 'QMAP', 1, 1)
            if(qmap.exists() == True):
                qmap.clrstat()
                qmap.zap()
            
            umap = AIPSImage(calsour, 'UMAP', 1, 1)
            if(umap.exists() == True):
                umap.clrstat()
                umap.zap()
                
            
            # self.logger.info('RAM memory {:4.2f}% used:'.format(psutil.virtual_memory()[2]))
            
            del time, dumifarr, dumchanarr, dumsourcearr, ant1, ant2, dumu, dumv, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight, qreal, qimag, ureal, uimag
            gc.collect()
    
    
        # Convert the lists to numpy arrays.
        ifarr, chanarr, timearr, sourcearr, ant1arr, ant2arr, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr = \
            np.array(ifarr), np.array(chanarr), np.array(timearr), np.array(sourcearr), np.array(ant1arr), np.array(ant2arr), np.array(uarr), np.array(varr), np.array(rrrealarr), np.array(rrimagarr), np.array(rrweightarr), np.array(llrealarr), np.array(llimagarr), np.array(llweightarr), \
            np.array(rlrealarr), np.array(rlimagarr), np.array(rlweightarr), np.array(lrrealarr), np.array(lrimagarr), np.array(lrweightarr)
        
        qrealarr, qimagarr, urealarr, uimagarr = np.array(qrealarr), np.array(qimagarr), np.array(urealarr), np.array(uimagarr)
        
        
        mod_q, mod_u = qrealarr + 1j*qimagarr, urealarr + 1j*uimagarr
        mod_rlreal, mod_rlimag, mod_lrreal, mod_lrimag = np.real(mod_q + 1j*mod_u), np.imag(mod_q + 1j*mod_u), np.real(mod_q - 1j*mod_u), np.imag(mod_q - 1j*mod_u)
        mod_rlamp, mod_rlphas, mod_lramp, mod_lrphas = np.absolute(mod_rlreal + 1j*mod_rlimag), np.angle(mod_rlreal + 1j*mod_rlimag), np.absolute(mod_lrreal + 1j*mod_lrimag), np.angle(mod_lrreal + 1j*mod_lrimag)
        mod_qamp, mod_qphas, mod_uamp, mod_uphas = np.absolute(mod_q), np.angle(mod_q), np.absolute(mod_u), np.angle(mod_u)
        
        
        rlweightarr[rlweightarr > self.remove_weight_outlier_threshold * np.median(rlweightarr)] = np.median(rlweightarr)
        lrweightarr[lrweightarr > self.remove_weight_outlier_threshold * np.median(lrweightarr)] = np.median(lrweightarr)
        
        
        # Combine the numpy arrays into a single pandas dataframe.
        self.data.loc[:,"IF"], self.data.loc[:, "channel"], self.data.loc[:,"time"], self.data.loc[:,"source"], self.data.loc[:,"ant1"], self.data.loc[:,"ant2"], self.data.loc[:,"u"], self.data.loc[:,"v"], \
        self.data.loc[:,"rrreal"], self.data.loc[:,"rrimag"], self.data.loc[:,"rrweight"], self.data.loc[:,"llreal"], self.data.loc[:,"llimag"], self.data.loc[:,"llweight"], \
        self.data.loc[:,"rlreal"], self.data.loc[:,"rlimag"], self.data.loc[:,"rlweight"], self.data.loc[:,"lrreal"], self.data.loc[:,"lrimag"], self.data.loc[:,"lrweight"] = \
                ifarr, chanarr, timearr * 24., sourcearr, ant1arr - 1, ant2arr - 1, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr


        # Append the model visibilities to the existing pandas dataframe as new columns.
        self.data.loc[:,"model_rlreal"], self.data.loc[:,"model_rlimag"], self.data.loc[:,"model_lrreal"], self.data.loc[:,"model_lrimag"], \
        self.data.loc[:,"model_rlamp"], self.data.loc[:,"model_rlphas"], self.data.loc[:,"model_lramp"], self.data.loc[:,"model_lrphas"], \
        self.data.loc[:,"model_qamp"], self.data.loc[:,"model_qphas"], self.data.loc[:,"model_uamp"], self.data.loc[:,"model_uphas"] = \
        mod_rlreal, mod_rlimag, mod_lrreal, mod_lrimag, mod_rlamp, mod_rlphas, mod_lramp, mod_lrphas, mod_qamp, mod_qphas, mod_uamp, mod_uphas
        
        
        
        # Filter bad data points.
        select = (rrweightarr > 0.) & (llweightarr > 0.) & (rlweightarr > 0.) & (lrweightarr > 0.) & (~np.isnan(rrweightarr)) & (~np.isnan(llweightarr)) & (~np.isnan(rlweightarr)) & (~np.isnan(lrweightarr))


        del ifarr, chanarr, timearr, sourcearr, ant1arr, ant2arr, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr
        del mod_rlreal, mod_rlimag, mod_lrreal, mod_lrimag, mod_rlamp, mod_rlphas, mod_lramp, mod_lrphas, mod_qamp, mod_qphas, mod_uamp, mod_uphas
        
        gc.collect()

        
        self.data = self.data.loc[select].reset_index(drop=True)
        
        dumant1 = self.data.loc[:,"ant1"]
        dumant2 = self.data.loc[:,"ant2"]
        dumsource = self.data.loc[:,"source"]
        
        longarr1, latarr1, f_el1, f_par1, phi_off1 = oh.coordarr(self.lonarr, self.latarr, self.f_el, self.f_par, self.phi_off, dumant1)
        longarr2, latarr2, f_el2, f_par2, phi_off2 = oh.coordarr(self.lonarr, self.latarr, self.f_el, self.f_par, self.phi_off, dumant2)
        
        yeararr, montharr, dayarr, raarr, decarr = oh.calendar(dumsource, self.channel_calsour, [self.year] * len(self.channel_calsour), [self.month] * len(self.channel_calsour), [self.day] * len(self.channel_calsour), self.obsra, self.obsdec)
        
        
        
        timearr = np.array(self.data.loc[:,"time"])
        
        for i in range(10):
            dayarr[timearr>=24.] += 1 # Version 1.1!
            timearr[timearr>=24.] -= 24. # Version 1.1!
            
        self.data.loc[:,"time"] = timearr
        self.data.loc[:,"year"] = yeararr
        self.data.loc[:,"month"] = montharr
        self.data.loc[:,"day"] = dayarr
                
        self.data.loc[:,"pang1"] = oh.get_parang(yeararr, montharr, dayarr, timearr, raarr, decarr, longarr1, latarr1, f_el1, f_par1, phi_off1)
        self.data.loc[:,"pang2"] = oh.get_parang(yeararr, montharr, dayarr, timearr, raarr, decarr, longarr2, latarr2, f_el2, f_par2, phi_off2)
        
        self.data = oh.pd_modifier(self.data)
        
    
    
    def get_selfcaled_data(self, calsourarr, inname, inclass, inseq, indisk, newclver, outclass):
        
        for l in range(len(calsourarr)):
            
            calsour = calsourarr[l]
            
            data = AIPSUVData(calsour, 'SPLIT', 1, 1)
            if(data.exists() == True):
                data.clrstat()
                data.zap()
            
            
            au.runsplit(inname, inclass, inseq = inseq, indisk = indisk, docal = 1, gainuse = newclver, sources = calsour, dopol = -1)
            
            
            multi = AIPSUVData(calsour, 'MULTI', 1, 1)
            if(multi.exists() == True):
                multi.clrstat()
                multi.zap()
            
            au.runmulti(calsour, 'SPLIT', 1, 1)
                
            
        for l in range(len(calsourarr) - 1):
            if(l == 0):
                
                if(len(calsourarr) == 2):
                    
                    dbcon = AIPSUVData(inname, 'DBCON', 1, 1)
                    if(dbcon.exists() == True):
                        dbcon.clrstat()
                        dbcon.zap()
                        
                    au.rundbcon(calsourarr[l], 'MULTI', 1, 1, calsourarr[l+1], 'MULTI', 1, 1, inname, outclass = 'DBCON', outseq = 1, fqcenter = -1)
                else:
                    
                    dbcon = AIPSUVData(calsourarr[l+1], 'DBCON', 1, 1)
                    if(dbcon.exists() == True):
                        dbcon.clrstat()
                        dbcon.zap()
                        
                    au.rundbcon(calsourarr[l], 'MULTI', 1, 1, calsourarr[l+1], 'MULTI', 1, 1, calsourarr[l+1], outclass = 'DBCON', outseq = 1, fqcenter = -1)

                
            elif(l != len(calsourarr) - 2):
                
                dbcon = AIPSUVData(calsourarr[l+1], 'DBCON', 1, 1)
                if(dbcon.exists() == True):
                    dbcon.clrstat()
                    dbcon.zap()
                
                au.rundbcon(calsourarr[l], 'DBCON', 1, 1, calsourarr[l+1], 'MULTI', 1, 1, calsourarr[l+1], outclass = 'DBCON', outseq = 1, fqcenter = -1)
                    
                
            else:
                
                dbcon = AIPSUVData(inname, 'DBCON', 1, 1)
                if(dbcon.exists() == True):
                    dbcon.clrstat()
                    dbcon.zap()
                
                au.rundbcon(calsourarr[l], 'DBCON', 1, 1, calsourarr[l+1], 'MULTI', 1, 1, inname, outclass = 'DBCON', outseq = 1, fqcenter = -1)
                
        
        for l in range(len(calsourarr)):            
            data = AIPSUVData(calsourarr[l], 'SPLIT', 1, 1)
            if(data.exists() == True):
                data.clrstat()
                data.zap()
                
            multi = AIPSUVData(calsourarr[l], 'MULTI', 1, 1)
            if(multi.exists() == True):
                multi.clrstat()
                multi.zap()
                
        
        for l in range(len(calsourarr) - 1):
            if(l == 0):
                
                dbcon = AIPSUVData(calsourarr[l+1], 'DBCON', 1, 1)
                if(dbcon.exists() == True):
                    dbcon.clrstat()
                    dbcon.zap()
                    
                    
            elif(l != len(calsourarr) - 2):
                
                dbcon = AIPSUVData(calsourarr[l+1], 'DBCON', 1, 1)
                if(dbcon.exists() == True):
                    dbcon.clrstat()
                    dbcon.zap()
                    
            
            
        newdata = AIPSUVData(inname, outclass, 1, 1)
        if(newdata.exists() == True):
            newdata.clrstat()
            newdata.zap()
        
        
        au.runuvcop(inname, 'DBCON', outclass = outclass)
        
        if (newclver != 1):
            dbcon = AIPSUVData(inname, 'DBCON', 1, 1)
            dbcon.zap()
            
            data = WAIPSUVData(inname, outclass, indisk, inseq)
            
        else:
            data = WAIPSUVData(inname, inclass, indisk, inseq)
        
        return data
    

    def dtermsolve(self):
        
        data = AIPSUVData(self.inname, self.inclass, self.indisk, self.inseq)
        
        self.nchan = data.header['naxis'][2]
        self.ifnum = data.header['naxis'][3]
        
        self.newsnver = self.snver
        self.newclver = self.clver
        
        data.zap_table('SN', self.snver+1)
        data.zap_table('SN', self.snver+2)
        
        data.zap_table('CL', self.clver+1)
        data.zap_table('CL', self.clver+2)
        data.zap_table('CL', self.clver+3)
        
        if self.doselfcal:
            self.newsnver += 1
            self.newclver += 1
        
        if self.doevpacal:
            au.runclcor(self.inname, self.inclass, self.indisk, self.inseq, self.clcorprm)
            self.newclver += 1
        
        
        if(type(self.ms) == int) | (type(self.ms) == float):
            self.ms = [self.ms] * len(self.channel_calsour)
        if(type(self.ps) == int) | (type(self.ps) == float):
            self.ps = [self.ps] * len(self.channel_calsour)
        if(type(self.uvbin) == int) | (type(self.uvbin) == float):
            self.uvbin = [self.uvbin] * len(self.channel_calsour)
        if(type(self.uvpower) == int) | (type(self.uvpower) == float):
            self.uvpower = [self.uvpower] * len(self.channel_calsour)
        if(type(self.dynam) == int) | (type(self.dynam) == float):
            self.dynam = [self.dynam] * len(self.channel_calsour)
        if(type(self.shift_x) == int) | (type(self.shift_x) == float):
            self.shift_x = [self.shift_x] * len(self.channel_calsour)
        if(type(self.shift_y) == int) | (type(self.shift_y) == float):
            self.shift_y = [self.shift_y] * len(self.channel_calsour)
            
                
        if self.doselfcal:
                        
            data.zap_table('SN', 1)
            
            if self.selfcal_doclean:
                
                self.channel_calsour_models = []
                
                if self.multiproc:
                    parmset = []
                    
                for l in range(len(self.channel_calsour)):
                                        
                    bif = 1
                    eif = self.ifnum
                
                    if self.multiproc:
                        parmset.append([self.direc, self.channel_calsour_uvf[l], self.channel_calsour_win[l], self.dataname + self.channel_calsour[l] + '.allIF', bif, eif, \
                                    self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'i'])
                    
                    else:
                        
                        ch.cleanqu(self.direc, self.channel_calsour_uvf[l], self.channel_calsour_win[l], self.dataname + self.channel_calsour[l] + '.allIF', bif, eif, \
                                    self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'i')
                            
                        self.logger.info('\nMaking CLEAN models for Stokes I map using a single core...\n')
                        
                        self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}\n'.format(self.channel_calsour_uvf[l], self.channel_calsour_win[l], \
                                          self.dataname+self.channel_calsour[l] + '.allIF.i'))
                            
                    
                    self.channel_calsour_models.append(self.dataname+self.channel_calsour[l] + '.allIF.i.mod')
                                            
                
                if self.multiproc:                
                    pool = Pool(processes = self.nproc)
                    pool.map(ch.cleanqu2, parmset)
                    pool.close()
                    pool.join()
                    
                    self.logger.info('Making CLEAN models for Stokes I maps using multiple cores...\n')
                    
                    for i in range(len(parmset)):
                        self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}.i'.format(parmset[i][1], parmset[i][2], parmset[i][3]))
                
                
            
            self.logger.info('')
            
            for l in range(len(self.channel_calsour)):
                
                self.logger.info('Performing self-calibration on {:s}...'.format(self.channel_calsour[l]))
                
                cmap = AIPSImage(self.channel_calsour[l], 'CMAP', 1, 1)
                if(cmap.exists() == True):
                    cmap.clrstat()
                    cmap.zap()
                
                au.runfitld(self.channel_calsour[l], 'CMAP', self.direc + self.channel_calsour_models[l].replace('.mod', '.fits'))
                
                calib = AIPSUVData(self.channel_calsour[l], 'CALIB', 1, 1)
                if(calib.exists() == True):
                    calib.clrstat()
                    calib.zap()
                    
                au.runcalib(self.inname, self.inclass, self.channel_calsour[l], 'CMAP', self.inclass, self.solint, 'L1R', 'A&P', 1, indisk = self.indisk, inseq = self.inseq, \
                            in2disk = 1, in2seq = 1, snver = self.newsnver, calsour = self.channel_calsour[l])
                
                cmap.zap()
                # calib.zap()
                    
            au.runclcal(self.inname, self.inclass, self.indisk, self.inseq, self.newsnver, interpol = 'SELF', gainver = self.newclver - 1, gainuse = self.newclver)
        
        
            self.logger.info('')
        
        
        data = self.get_selfcaled_data(self.channel_calsour, self.inname, self.inclass, self.inseq, self.indisk, self.newclver, 'SPLAT')
        
        
        if self.calsour_doclean:
            
            f = open(self.direc+'GPCAL_Difmap_v1','w')
            f.write('observe %1\nmapcolor rainbow, 1, 0.5\nselect %13, %2, %3\nmapsize %4, %5\nuvweight %6, %7\nrwin %8\nshift %9,%10\ndo i=1,100\nclean 100, 0.02, imstat(rms)*%11\nend do\nselect i\nsave %12.%13\nexit')
            f.close()
            
            
            self.qmodelname = []
            self.umodelname = []
            
            
            if self.multiproc:
                parmset = []
                
            for l in range(len(self.channel_calsour)):
                if not self.polcal_unpol[l]:
                    if self.clean_IF_combine:
                        bif = 1
                        eif = self.ifnum
                        
                        if self.multiproc:
                            parmset.append([self.direc, self.channel_calsour_uvf[l], self.channel_calsour_win[l], self.dataname + self.channel_calsour[l] + '.allIF', bif, eif, \
                                            self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q'])
                            parmset.append([self.direc, self.channel_calsour_uvf[l], self.channel_calsour_win[l], self.dataname + self.channel_calsour[l] + '.allIF', bif, eif, \
                                            self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u'])
                            
                        else:
                            ch.cleanqu(self.direc, self.channel_calsour_uvf[l], self.channel_calsour_win[l], self.dataname + self.channel_calsour[l] + '.allIF', bif, eif, \
                                        self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q')
                            ch.cleanqu(self.direc, self.channel_calsour_uvf[l], self.channel_calsour_win[l], self.dataname + self.channel_calsour[l] + '.allIF', bif, eif, \
                                        self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u')
                                
                            self.logger.info('\nMaking CLEAN models for Stokes Q and U maps using a single core...\n')
                            
                            self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}\n'.format(self.channel_calsour_uvf[l], self.channel_calsour_win[l], \
                                              self.dataname + self.channel_calsour[l] + '.allIF.q,u'))
                            
                    
                    else:
                        for ifn in range(self.ifnum):
                            bif = ifn + 1
                            eif = ifn + 1
                            
                            if self.multiproc:
                                parmset.append([self.direc, self.channel_calsour_uvf[l], self.channel_calsour_win[l], self.dataname + self.channel_calsour[l] + '.IF{:}'.format(ifn+1), bif, eif, \
                                                self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q'])
                                parmset.append([self.direc, self.channel_calsour_uvf[l], self.channel_calsour_win[l], self.dataname + self.channel_calsour[l] + '.IF{:}'.format(ifn+1), bif, eif, \
                                                self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u'])
                                    
                            else:
                                ch.cleanqu(self.direc, self.channel_calsour_uvf[l], self.channel_calsour_win[l], self.dataname + self.channel_calsour[l] + '.IF{:}'.format(ifn+1), bif, eif, \
                                            self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q')
                                ch.cleanqu(self.direc, self.channel_calsour_uvf[l], self.channel_calsour_win[l], self.dataname + self.channel_calsour[l] + '.IF{:}'.format(ifn+1), bif, eif, \
                                            self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u')
                                
                                self.logger.info('\nMaking CLEAN models for Stokes Q and U maps using a single core...\n')
                                
                                self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}\n'.format(self.channel_calsour_uvf[l], self.channel_calsour_win[l], \
                                                  self.dataname + self.channel_calsour[l] + '.IF*.q,u'))
                
                
                if self.clean_IF_combine:
                    if not self.polcal_unpol[l]:
                        self.qmodelname.append(self.dataname+self.channel_calsour[l] + '.allIF.q.mod')
                        self.umodelname.append(self.dataname+self.channel_calsour[l] + '.allIF.u.mod')
                
                else:
                    dumqmodelname = []
                    dumumodelname = []
                    for k in range(self.ifnum):
                        dumqmodelname.append(self.dataname+self.channel_calsour[l] + '.IF{:}.q.mod'.format(k+1))
                        dumumodelname.append(self.dataname+self.channel_calsour[l] + '.IF{:}.u.mod'.format(k+1))
                    if not self.polcal_unpol[l]:
                        self.qmodelname.append(dumqmodelname)
                        self.umodelname.append(dumumodelname)

            
            if self.multiproc:                
                pool = Pool(processes = self.nproc)
                pool.map(ch.cleanqu2, parmset)
                pool.close()
                pool.join()
                
                self.logger.info('\nMaking CLEAN models for Stokes I maps using multiple cores...\n')
                
                for i in range(len(parmset)):
                    self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}.i'.format(parmset[i][1], parmset[i][2], parmset[i][3]))
    
    
        self.get_channel_data(data)
        
        for k in range(self.ifnum):
            for i in range(self.nchan):
                
                dumselect = (self.data["IF"] == k+1) & (self.data["channel"] == i + 1)
                channel_data = self.data[dumselect]
                                
                self.logger.info('\n####################################################################')
                self.logger.info('Processing IF {:d}, channel {:d}'.format(k+1, i+1) + '...')
                self.logger.info('####################################################################\n')
                
                if(len(channel_data) == 0):
                    self.logger.info('Will skip this channel because there is no data.\n')
                    continue
                
                
                fitresults, fitdterms, chisq = ps.pol_gpcal(self.direc, self.outputname, channel_data, self.channel_calsour, self.antname, self.logger, Dbound = self.Dbound, Pbound = self.Pbound, \
                                                      manualweight = self.manualweight, weightfactors = self.weightfactors)
                
                channel_ant, channel_DRArr, channel_DLArr = fitdterms
                channel_IF = np.repeat(k+1, len(channel_ant))
                
                
                df = pd.DataFrame(channel_IF.transpose())
                df['antennas'] = np.array(channel_ant)
                df['IF'] = np.array(channel_IF)
                df['DRArr'] = np.array(channel_DRArr)
                df['DLArr'] = np.array(channel_DLArr)
                del df[0]
                
                df.to_csv(self.direc+'gpcal/'+self.outputname+'IF{:d}.channel{:02d}.dterm.csv'.format(k+1, i+1), sep = "\t")
                
        
        self.dtermcorrect()
        
                
    def dtermcorrect(self):
        """
        Apply the best-fit D-terms estimated by using the similarity assumption to UVFITS data.
        
        Args:
            source (list): a list of the sources for which D-terms will be corrected.
            DRArr (list): a list of the best-fit RCP D-terms.
            DLArr (list): a list of the best-fit LCP D-terms.
        """
        
        # self.newsnver = self.snver
        # self.newclver = self.clver
            
        data = WAIPSUVData(self.inname, self.inclass, self.indisk, self.inseq)
        
        self.nchan = data.header['naxis'][2]
        self.ifnum = data.header['naxis'][3]
        
        if self.doselfcal:
            self.newsnver += 1
            self.newclver += 1
        
            
        if(type(self.ms) == int) | (type(self.ms) == float):
            self.ms = [self.ms] * len(self.channel_source)
        if(type(self.ps) == int) | (type(self.ps) == float):
            self.ps = [self.ps] * len(self.channel_source)
        if(type(self.uvbin) == int) | (type(self.uvbin) == float):
            self.uvbin = [self.uvbin] * len(self.channel_source)
        if(type(self.uvpower) == int) | (type(self.uvpower) == float):
            self.uvpower = [self.uvpower] * len(self.channel_source)
        if(type(self.dynam) == int) | (type(self.dynam) == float):
            self.dynam = [self.dynam] * len(self.channel_source)
        if(type(self.shift_x) == int) | (type(self.shift_x) == float):
            self.shift_x = [self.shift_x] * len(self.channel_source)
        if(type(self.shift_y) == int) | (type(self.shift_y) == float):
            self.shift_y = [self.shift_y] * len(self.channel_source)
        
        
        if self.doselfcal:
                
            if self.selfcal_doclean:
                
                if self.multiproc:
                    parmset = []
                
                self.channel_source_models = []
                
                for l in range(len(self.channel_source)):
                    
                    bif = 1
                    eif = self.ifnum
                    
                    if self.multiproc:
                        parmset.append([self.direc, self.channel_source_uvf[l], self.channel_source_win[l], self.dataname + self.channel_source[l] + '.allIF', bif, eif, \
                                   self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'i'])
                            
                    else:
                        ch.cleanqu(self.direc, self.channel_source_uvf[l], self.channel_source_win[l], self.dataname + self.channel_source[l] + '.allIF', bif, eif, \
                                   self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'i')
                            
                        self.logger.info('\nMaking CLEAN models for Stokes I map using a single core...\n')
                        
                        self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}\n'.format(self.channel_source_uvf[l], self.channel_source_win[l], \
                                          self.dataname + self.channel_source[l] + '.allIF.i'))
                    
                    self.channel_source_models.append(self.dataname+self.channel_source[l] + '.allIF.i.mod')
                        
                
                if self.multiproc:                
                    pool = Pool(processes = self.nproc)
                    pool.map(ch.cleanqu2, parmset)
                    pool.close()
                    pool.join()
                    
                    self.logger.info('\nMaking CLEAN models for Stokes I maps using multiple cores...\n')
                    
                    for i in range(len(parmset)):
                        self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}.i'.format(parmset[i][1], parmset[i][2], parmset[i][3]))
        
            
            data.zap_table('SN', self.newsnver)
            data.zap_table('CL', self.newclver)
            
            for l in range(len(self.channel_source)):
                
                self.logger.info('Performing self-calibration on {:s}...'.format(self.channel_source[l]))
                
                cmap = AIPSImage(self.channel_source[l], 'CMAP', 1, 1)
                if(cmap.exists() == True):
                    cmap.clrstat()
                    cmap.zap()
                    
                au.runfitld(self.channel_source[l], 'CMAP', self.direc + self.channel_source_models[l].replace('.mod', '.fits'))
                
                calib = AIPSUVData(self.channel_source[l], 'CALIB', 1, 1)
                if(calib.exists() == True):
                    calib.clrstat()
                    calib.zap()
                    
                au.runcalib(self.inname, self.inclass, self.channel_source[l], 'CMAP', self.inclass, self.solint, 'L1R', 'a&p', 1, indisk = self.indisk, inseq = self.inseq, \
                            in2disk = 1, in2seq = 1, snver = self.newsnver, calsour = self.channel_source[l])
                
                cmap.zap()
                
            
            au.runclcal(self.inname, self.inclass, self.indisk, self.inseq, self.newsnver, interpol = 'SELF', gainver = self.newclver - 2, gainuse = self.newclver)
        
        
        data = self.get_selfcaled_data(self.channel_source, self.inname, self.inclass, self.inseq, self.indisk, self.newclver, 'DTCOR')
        
        
        year, month, day = oh.get_obsdate(data)
                
        sutable = data.table('SU', 1)
        
        wholesource, obsra, obsdec, sourceid = [], [], [], []
        
        for row in sutable:
            wholesource.append(row.source.replace(' ', ''))
            sourceid.append(row.id__no)
            obsra.append(row.raobs)
            obsdec.append(row.decobs)
            
        wholesource = np.array(wholesource)
        sourceid = np.array(sourceid)
        
        self.antname, self.antx, self.anty, self.antz, self.antmount, self.f_par, self.f_el, self.phi_off = oh.get_antcoord(data)
        
        self.ifnum, self.freq = oh.get_freqinfo(data)
        
        self.lonarr, self.latarr, self.heightarr = oh.coord(self.antname, self.antx, self.anty, self.antz)

        self.nant = len(self.antname)

        
        self.logger.info('\nLoading channel-dependent D-terms...')
        # self.logger.info('\n####################################################################')

        dterm_antarr, dterm_ifarr, dterm_chanarr, dterm_drarr, dterm_dlarr = [], [], [], [], []
    
        for k in range(self.ifnum):            
            for i in range(self.nchan):
                dtermfile = glob.glob('{:}gpcal/{:}IF{:d}.channel{:02d}.dterm.csv'.format(self.direc, self.dataname, k+1, i+1))
                
                if(len(dtermfile) > 0):
                    dtermread = pd.read_csv(dtermfile[0], header = 0, skiprows=0, delimiter = '\t', index_col = 0)
                    dterm_antarr = dterm_antarr + list(dtermread['antennas'])
                    dterm_ifarr = dterm_ifarr + [k+1]*len(dtermread)
                    dterm_chanarr = dterm_chanarr + [i+1]*len(dtermread)
                    dterm_drarr = dterm_drarr + list(dtermread['DRArr'])
                    dterm_dlarr = dterm_dlarr + list(dtermread['DLArr'])
                else:
                    dterm_antarr = dterm_antarr + self.antname
                    dterm_ifarr = dterm_ifarr + [k+1]*len(self.antname)
                    dterm_chanarr = dterm_chanarr + [i+1]*len(self.antname)
                    dterm_drarr = dterm_drarr + [0j]*len(self.antname)
                    dterm_dlarr = dterm_dlarr + [0j]*len(self.antname)
                
        dterm_drarr = [complex(it) for it in dterm_drarr]   
        dterm_dlarr = [complex(it) for it in dterm_dlarr]
        
        dterm_antarr, dterm_ifarr, dterm_chanarr, dterm_drarr, dterm_dlarr = np.array(dterm_antarr), np.array(dterm_ifarr), np.array(dterm_chanarr), np.array(dterm_drarr), np.array(dterm_dlarr)
        
        
        self.logger.info('\nLoading data for all IFs and channels...')


        time, ant1, ant2 = np.zeros(len(data)), np.zeros(len(data), dtype = int), np.zeros(len(data), dtype = int)
        sourcearr = np.chararray(len(data), itemsize = 16)
        dumvis = np.zeros((len(data), self.ifnum, self.nchan, 4, 3))
                
        dum = 0
        for row in data:
            time[dum] = row.time
            dumsel = (sourceid == row.source)
            if(np.sum(dumsel) == 0):
                sourcearr[dum] = 'None'
            else:
                sourcearr[dum] = wholesource[dumsel][0]
            ant1[dum] = row.baseline[0]
            ant2[dum] = row.baseline[1]
            dumvis[dum,:,:,:,:] = row.visibility
            
            dum += 1
            

        dumvisshape = dumvis.shape
                
        time = time * 24.
        ant1 = ant1 - 1
        ant2 = ant2 - 1
        
        
        longarr1, latarr1, f_el1, f_par1, phi_off1 = oh.coordarr(self.lonarr, self.latarr, self.f_el, self.f_par, self.phi_off, ant1)
        longarr2, latarr2, f_el2, f_par2, phi_off2 = oh.coordarr(self.lonarr, self.latarr, self.f_el, self.f_par, self.phi_off, ant2)
        
        yeararr, montharr, dayarr, raarr, decarr = oh.calendar(sourcearr, wholesource, [year]*len(wholesource), [month]*len(wholesource), [day]*len(wholesource), obsra, obsdec)
        
        for i in range(10):
            dayarr[time>=24.] += 1 
            time[time>=24.] -= 24. 
            
            
        self.logger.info('\nComputing the field rotation angles...')
        
        pang1 = oh.get_parang(yeararr, montharr, dayarr, time, raarr, decarr, longarr1, latarr1, f_el1, f_par1, phi_off1)
        pang2 = oh.get_parang(yeararr, montharr, dayarr, time, raarr, decarr, longarr2, latarr2, f_el2, f_par2, phi_off2)
        
        del time, sourcearr
        gc.collect()
       
        
        self.logger.info('\nComputing D-term matrices...')
        
        tot_drarr, tot_dlarr = [], []
        
        for chan in range(self.nchan):
            for ifn in range(self.ifnum):    
                dum_drarr = []
                dum_dlarr = []
                for ant in range(self.nant):
                    dumselect = (dterm_antarr == self.antname[ant]) & (dterm_ifarr == ifn+1) & (dterm_chanarr == chan+1)
                    dum_drarr.append(dterm_drarr[dumselect][0])
                    dum_dlarr.append(dterm_dlarr[dumselect][0])
                    
                tot_drarr = tot_drarr + dum_drarr
                tot_dlarr = tot_dlarr + dum_dlarr

        tot_drarr, tot_dlarr = np.array(tot_drarr).astype('complex64'), np.array(tot_dlarr).astype('complex64')
                
        
        self.logger.info('\nMaking other necessary arrays...')
        
        indnum = len(ant1)
        
        ant1 = np.tile(ant1, self.ifnum * self.nchan)
        ant2 = np.tile(ant2, self.ifnum * self.nchan)
        pang1 = np.tile(pang1, self.ifnum * self.nchan)
        pang2 = np.tile(pang2, self.ifnum * self.nchan)
        
        
        self.logger.info('\nConverting the visibilities into numpy arrays...')

        # ravel order = time first, then IF, then channel
        rrreal = dumvis[:,:,:,0,0].ravel(order = 'F')
        rrimag = dumvis[:,:,:,0,1].ravel(order = 'F')
        rrvis = (rrreal + 1j * rrimag).astype('complex64')
        
        
        del rrreal, rrimag
        gc.collect()
        
        
        llreal = dumvis[:,:,:,1,0].ravel(order = 'F')
        llimag = dumvis[:,:,:,1,1].ravel(order = 'F')
        llvis = (llreal + 1j * llimag).astype('complex64')
        
        
        del llreal, llimag
        gc.collect()
        
        
        rlreal = dumvis[:,:,:,2,0].ravel(order = 'F')
        rlimag = dumvis[:,:,:,2,1].ravel(order = 'F')
        rlvis = (rlreal + 1j * rlimag).astype('complex64')
        
        
        del rlreal, rlimag
        gc.collect()
        
        
        lrreal = dumvis[:,:,:,3,0].ravel(order = 'F')
        lrimag = dumvis[:,:,:,3,1].ravel(order = 'F')
        lrvis = (lrreal + 1j * lrimag).astype('complex64')
        
        
        del lrreal, lrimag, dumvis
        gc.collect()
        
        
        totnum = len(rrvis)
                
        Vmat = np.array([rrvis, rlvis, lrvis, llvis]).reshape(2,2,totnum).astype('complex64')
        
        del rrvis, rlvis, lrvis, llvis
        gc.collect()
        
        
        self.logger.info('\nMaking channel and IF arrays...')
        
        
        chanarr = np.zeros(len(ant1)).astype(int)
        ifarr = np.zeros(len(ant1)).astype(int)
        
        for chan in range(self.nchan):
            for ifn in range(self.ifnum):    
                chanarr[indnum*ifn + indnum*self.ifnum*chan : indnum*(ifn+1) + indnum*self.ifnum*chan] = chan
                ifarr[indnum*ifn + indnum*self.ifnum*chan : indnum*(ifn+1) + indnum*self.ifnum*chan] = ifn
        
        
        self.logger.info('\nMaking the full D-term arrays...')
        
        dumindex1 = ant1 + ifarr*self.nant + chanarr*self.nant*self.ifnum
        dumindex2 = ant2 + ifarr*self.nant + chanarr*self.nant*self.ifnum
        
        tot_D_iR = tot_drarr[dumindex1].astype('complex64')
        tot_D_iL = tot_dlarr[dumindex1].astype('complex64')
        
        
        del dumindex1
        gc.collect()
        
                
        self.logger.info('\nComputing the inverse of D_i matrices...')
                
        inv_mat_Di = np.array([np.repeat(1., totnum), -tot_D_iR * np.exp(2j * pang1), -tot_D_iL * np.exp(-2j * pang1), np.repeat(1., totnum)], dtype='complex64')
        inv_mat_Di = inv_mat_Di / (1. - tot_D_iR * np.exp(2j * pang1) * tot_D_iL * np.exp(-2j * pang1))
        
        inv_mat_Di = inv_mat_Di.reshape(2,2,totnum)
        
        
        tot_D_jR = tot_drarr[dumindex2].astype('complex64')
        tot_D_jL = tot_dlarr[dumindex2].astype('complex64')
        
        
        self.logger.info('\nComputing the inverse of D_j matrices...')
        
        inv_mat_Dj = np.array([np.repeat(1., totnum), -tot_D_jL.conj() * np.exp(2j * pang2), -tot_D_jR.conj() * np.exp(-2j * pang2), np.repeat(1., totnum)], dtype='complex64')
        inv_mat_Dj = inv_mat_Dj / (1. - tot_D_jL.conj() * np.exp(2j * pang2) * tot_D_jR.conj() * np.exp(-2j * pang2))
        
        inv_mat_Dj = inv_mat_Dj.reshape(2,2,totnum)
        
        
        del tot_D_jR, tot_D_jL, ant1, ant2, pang1, pang2
        gc.collect()
        
        
        self.logger.info('\nRemoving D-terms from the visibilities...')     
        
        dum_mat = np.array([inv_mat_Di[0,0,:] * Vmat[0,0,:] + inv_mat_Di[0,1,:] * Vmat[1,0,:], inv_mat_Di[0,0,:] * Vmat[0,1,:] + inv_mat_Di[0,1,:] * Vmat[1,1,:], \
                            inv_mat_Di[1,0,:] * Vmat[0,0,:] + inv_mat_Di[1,1,:] * Vmat[1,0,:], inv_mat_Di[1,0,:] * Vmat[0,1,:] + inv_mat_Di[1,1,:] * Vmat[1,1,:]], dtype = 'complex64').reshape(2,2,totnum)
        
        
        del Vmat
        gc.collect()
        
        true_mat = np.array([dum_mat[0,0,:] * inv_mat_Dj[0,0,:] + dum_mat[0,1,:] * inv_mat_Dj[1,0,:], dum_mat[0,0,:] * inv_mat_Dj[0,1,:] + dum_mat[0,1,:] * inv_mat_Dj[1,1,:], \
                        dum_mat[1,0,:] * inv_mat_Dj[0,0,:] + dum_mat[1,1,:] * inv_mat_Dj[1,0,:], dum_mat[1,0,:] * inv_mat_Dj[0,1,:] + dum_mat[1,1,:] * inv_mat_Dj[1,1,:]], dtype = 'complex64').reshape(2,2,totnum)
        

        del dum_mat, inv_mat_Dj, inv_mat_Di
        gc.collect()


        self.logger.info('\nComputing the arrays of RR, RL, LR, LL visibilities from the true matrices...')
        
        true_vis_rr = true_mat[0,0,:].reshape(dumvisshape[0:3], order = 'F')
        true_vis_rl = true_mat[0,1,:].reshape(dumvisshape[0:3], order = 'F')
        true_vis_lr = true_mat[1,0,:].reshape(dumvisshape[0:3], order = 'F')
        true_vis_ll = true_mat[1,1,:].reshape(dumvisshape[0:3], order = 'F')
        
        
        
        self.logger.info('\nReplacing the original visibilities with the new D-term corrected visibilities...')

        dum = 0
        for vis in data:
            if(vis.baseline[0] == vis.baseline[1]):
                continue
            
            vis.visibility[:,:,0,0] = np.real(true_vis_rr[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,0,1] = np.imag(true_vis_rr[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,1,0] = np.real(true_vis_ll[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,1,1] = np.imag(true_vis_ll[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,2,0] = np.real(true_vis_rl[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,2,1] = np.imag(true_vis_rl[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,3,0] = np.real(true_vis_lr[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,3,1] = np.imag(true_vis_lr[dum,:,:])
            vis.update()
                        
            dum += 1
            
        dum = 0
        for vis in data:
            vis.update()
            vis.update()
            vis.update()
            vis.update()
            
            dum += 1

        dum = 0
        for vis in data:
            if(vis.baseline[0] == vis.baseline[1]):
                continue
            
            vis.visibility[:,:,0,0] = np.real(true_vis_rr[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,0,1] = np.imag(true_vis_rr[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,1,0] = np.real(true_vis_ll[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,1,1] = np.imag(true_vis_ll[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,2,0] = np.real(true_vis_rl[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,2,1] = np.imag(true_vis_rl[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,3,0] = np.real(true_vis_lr[dum,:,:])
            vis.update()
            
            vis.visibility[:,:,3,1] = np.imag(true_vis_lr[dum,:,:])
            vis.update()
                        
            dum += 1
            
        dum = 0
        for vis in data:
            vis.update()
            vis.update()
            vis.update()
            vis.update()
            
            dum += 1


