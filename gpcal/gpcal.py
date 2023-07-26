
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
        

# Default matplotlib parameters
plt.rc('font', size=21)
matplotlib.rc('font', family='Dejavu Sans')
matplotlib.rc('font', serif='Helvetica Neue')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', titlesize=25)
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.subplots_adjust(left = 0.15, bottom = 0.15)

np.set_printoptions(threshold=sys.maxsize)


# -----------------------------------------------------------------------------
# GPCAL Class
# -----------------------------------------------------------------------------    
class polcal(object):
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
    def __init__(self, aips_userno, direc, dataname, calsour, source, cnum, autoccedt, \
                 Dbound = 1.0, Pbound = np.inf, outputname = None, drange = None, multiproc = True, nproc = 2, \
                 selfcal = False, solint = 10./60., solmode = 'A&P', soltype = 'L1R', weightit = 1, \
                 zblcal = False, zblcalsour = None, zblant = None, \
                 fixdterm = False, pol_fixdterm = False, fixdr = None, fixdl = None, \
                 selfpol = False, polcalsour = None, polcal_unpol = None, selfpoliter = None, ms = None, ps = None, uvbin = None, uvpower = None, shift_x = 0, shift_y = 0, dynam = None, pol_IF_combine = False, \
                 manualweight = False, weightfactors = None, lpcal = True, remove_weight_outlier_threshold = 10000, \
                 vplot = False, vplot_title = None, vplot_scanavg = False, vplot_avg_nat = False, resplot = False, parplot = True, allplot = False, dplot_IFsep = False, tsep = 2./60., filetype = 'pdf', \
                 aipslog = True, difmaplog = True):


        self.aips_userno = aips_userno
        
        self.direc = direc
        
        self.multiproc = multiproc
        self.nproc = nproc
        
        self.dataname = dataname
        self.calsour = copy.deepcopy(calsour)
        self.source = copy.deepcopy(source)
        self.cnum = copy.deepcopy(cnum)
        self.unpolsource = []
        for i in range(len(self.calsour)):
            if(self.cnum[i] == 0):
                self.unpolsource.append(self.calsour[i])
        self.autoccedt = copy.deepcopy(autoccedt)
        
        self.Dbound = Dbound
        self.Pbound = Pbound
        
        self.drange = drange
        
        self.remove_weight_outlier_threshold = remove_weight_outlier_threshold
                
        
        if(self.dataname[-1] != '.'): self.dataname = self.dataname + '.'
        
        self.outputname = copy.deepcopy(outputname)
        if(self.outputname == None):
            self.outputname = copy.deepcopy(self.dataname)
        else:
            if(self.outputname[-1] != '.'): self.outputname = self.outputname + '.'
        
        self.selfcal = selfcal
        self.solint = solint
        self.solmode = solmode
        self.soltype = soltype
        self.weightit = weightit
        
        self.zblcal = zblcal
        self.zblcalsour = copy.deepcopy(zblcalsour)
        self.zblant = copy.deepcopy(zblant)
        
        self.fixdterm = fixdterm
        self.pol_fixdterm = pol_fixdterm
        self.fixdr = copy.deepcopy(fixdr)
        self.fixdl = copy.deepcopy(fixdl)
        
        self.selfpol = selfpol
        self.polcalsour = copy.deepcopy(polcalsour)
        self.polcal_unpol = polcal_unpol
        if(self.polcal_unpol == None) & (self.polcalsour != None): 
            self.polcal_unpol = [False] * len(self.polcalsour)
        self.selfpoliter = selfpoliter
        self.pol_IF_combine = pol_IF_combine
        
        self.ms = ms
        self.ps = ps
        self.uvbin = uvbin
        self.uvpower = uvpower
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.dynam = dynam
        
        self.manualweight = manualweight
        self.weightfactors = weightfactors
        self.lpcal = lpcal
        
        
        self.vplot = vplot
        self.vplot_title = vplot_title
        self.vplot_scanavg = vplot_scanavg
        self.vplot_avg_nat = vplot_avg_nat
        self.resplot = resplot
        self.parplot = parplot
        self.allplot = allplot
        self.tsep = tsep
        self.dplot_IFsep = dplot_IFsep
        self.filetype = filetype
        
        self.aipslog = aipslog
        self.difmaplog = difmaplog
        
        self.aipstime = 0.
        self.difmaptime = 0.
        self.gpcaltime = 0.
        

        # Create a list of colors
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf'] + \
                       ['blue', 'red', 'orange', 'steelblue', 'green', 'slategrey', 'cyan', 'royalblue', 'purple', 'blueviolet', 'darkcyan', 'darkkhaki', 'magenta', 'navajowhite', 'darkred'] + \
                           ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf'] + \
                                          ['blue', 'red', 'orange', 'steelblue', 'green', 'slategrey', 'cyan', 'royalblue', 'purple', 'blueviolet', 'darkcyan', 'darkkhaki', 'magenta', 'navajowhite', 'darkred']
        # Define a list of markers
        self.markerarr = ['o', '^', 's', '<', 'p', '*', 'X', 'P', 'D', 'v', 'd', 'x'] * 5
        
        
        # If direc does not finish with a slash, then append it.
        if(self.direc[-1] != '/'): self.direc = self.direc + '/'

        # Create a folder named 'gpcal' in the working directory if it does not exist.
        if(os.path.exists(self.direc+'gpcal') == False):
            os.system('mkdir ' + self.direc+'gpcal') 
            
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
        
                
#        # Check the input parameters
#        for i in range(len(self.calsour)):
#            if(self.calsour[i] not in self.source):
#                self.source.append(self.calsour[i])
#                self.logger.info('{:s} is not in SOURCE. It will be added.'.format(self.calsour[i]))
#        
#        for i in range(len(self.polcalsour)):
#            if(self.polcalsour[i] not in self.source):
#                self.source.append(self.polcalsour[i])
#                self.logger.info('{:s} is not in SOURCE. It will be added.'.format(self.polcalsour[i]))
        
#        if(len(self.calsour) != len(self.cnum)):
#            raise Exception("The number of CALSOUR and CNUM do nat match!")
#        
#        if(len(self.calsour) != len(self.autoccedt)):
#            raise Exception("The number of CALSOUR and AUTOCCEDT do nat match!")
#        
#        for i in range(len(self.calsour)):
#            if not path.exists(self.direc + self.dataname + self.calsour[i] + '.uvf'):
#                raise Exception("{:s} does not exist in {:s}.".format(self.dataname + self.calsour[i] + '.uvf', self.direc))
#            
#            if not path.exists(self.direc + self.dataname + self.calsour[i] + '.fits'):
#                raise Exception("{:s} does not exist in {:s}.".format(self.dataname + self.calsour[i] + '.fits', self.direc))
#            
#            if(self.autoccedt[i] == False):
#                if (cnum[i] != 0) & (not path.exists(self.direc + 'gpcal/' + self.dataname + self.calsour[i] + '.box')):
#                    message = "It was requested to split the total intensity CLEAN components of {:s} into sub-models (autoccedt = False) but there is no '.box' file in '{:s}'.".format(self.calsour[i], self.direc + 'gpcal/')
#                    raise Exception(message)
#        
#        for i in range(len(self.polcalsour)):
#            if not path.exists(self.direc + self.dataname + self.polcalsour[i] + '.uvf'):
#                raise Exception("{:s} does not exist in {:s}.".format(self.dataname + self.polcalsour[i] + '.uvf', self.direc))
#            
#            if not path.exists(self.direc + self.dataname + self.polcalsour[i] + '.fits'):
#                raise Exception("{:s} does not exist in {:s}.".format(self.dataname + self.polcalsour[i] + '.fits', self.direc))


    def get_zbl_data(self):
        """
        Make a pandas dataframe containing UV data and models for the zero-baseline D-term estimation
        
        """      
        
        
        AIPS.userno = self.aips_userno
        
        if self.aipslog:
            AIPS.log = open(self.logfile, 'a')
        AIPSTask.msgkill = -1
        
        
        ifarr, timearr, sourcearr, ant1arr, ant2arr, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        pdkey = ["IF", "year", "month", "day", "time", "source", "ant1", "ant2", "u", "v", "pang1", "pang2", "rrreal", "rrimag", "rrsigma", "llreal", "llimag", "llsigma", "rlreal", "rlimag", "rlsigma", "lrreal", "lrimag", "lrsigma", \
             "rramp", "rrphas", "rramp_sigma", "rrphas_sigma", "llamp", "llphas", "llamp_sigma", "llphas_sigma", "rlamp", "rlphas", "rlamp_sigma", "rlphas_sigma", "lramp", "lrphas", "lramp_sigma", "lrphas_sigma", \
             "qamp", "qphas", "qamp_sigma", "qphas_sigma", "qsigma", "uamp", "uphas", "uamp_sigma", "uphas_sigma", "usigma"]

        info = oh.basic_info(self.zblcalsour, self.direc, self.dataname)

        obsra = info['obsra']
        obsdec = info['obsdec']
        year = info['year']
        month = info['month']
        day = info['day']
        antname = info['antname']
        antx = info['antx']
        anty = info['anty']
        antz = info['antz']
        antmount = info['antmount']
        ifnum = info['ifnum']
        freq = info['freq']
        f_par = info['f_par']
        f_el = info['f_el']
        phi_off = info['phi_off']
        
        self.logger.info('\nGetting data for {:d} sources for {:d} IFs...\n'.format(len(self.zblcalsour), ifnum))

        lonarr, latarr, heightarr = oh.coord(antname, antx, anty, antz)

        nant = len(antname)
        
        self.ifnum = ifnum
        self.year = year
        self.month = month
        self.day = day
        self.antname = antname
        self.antmount = antmount
        self.antx = antx
        self.anty = anty
        self.antz = antz
        self.zbl_obsra = obsra
        self.zbl_obsdec = obsdec


        zbl_antname = []
        for i in range(len(self.zblant)):
            zbl_antname.append(self.zblant[i][0])
            zbl_antname.append(self.zblant[i][1])   
            

        ifarr, timearr, sourcearr, ant1arr, ant2arr, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        
        self.zbl_data = pd.DataFrame(columns = pdkey)
        
        
        for l in range(len(self.zblcalsour)):
        
            inname = str(self.zblcalsour[l])
        
            data = AIPSUVData(inname, 'EDIT', 1, 1)
            if(data.exists() == True):
                data.clrstat()
                data.zap()
            
            # Load UVFITS file.
            au.runfitld(inname, 'EDIT', self.direc+self.dataname+self.zblcalsour[l]+'.uvf')
                
            data = AIPSUVData(inname, 'EDIT', 1, 1)
                        
            # Perform additional self-calibration with CALIB if requested.
            if self.selfcal:            
                
                cmap = AIPSImage(inname, 'CMAP', 1, 1)
                if(cmap.exists() == True):
                    cmap.clrstat()
                    cmap.zap()
                    
                au.runfitld(inname, 'CMAP', self.direc+self.dataname+self.zblcalsour[l]+'.fits')
                
                
                calib = AIPSUVData(inname, 'CALIB', 1, 1)
                if(calib.exists() == True):
                    calib.clrstat()
                    calib.zap()
                
                au.runcalib(inname, 'EDIT', inname, 'CMAP', 'CALIB', self.solint, self.soltype, self.solmode, self.weightit)
                
                calib = WAIPSUVData(inname, 'CALIB', 1, 1)
                
                dumu, dumv, ifname, time, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight = \
                    oh.uvprt(calib, "all")
                
                cmap.zap()                
                calib.zap()
            
            else:
                
                data = WAIPSUVData(inname, 'EDIT', 1, 1)
                
                dumu, dumv, ifname, time, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight = \
                    oh.uvprt(data, "all")
                    
                
            data.zap()
            
            
            uarr = uarr + dumu
            varr = varr + dumv
            sourcearr = sourcearr + [self.zblcalsour[l]] * len(time)
            ifarr = ifarr + ifname
            timearr = timearr + time
            ant1arr = ant1arr + ant1
            ant2arr = ant2arr + ant2
            rrrealarr = rrrealarr + rrreal
            rrimagarr = rrimagarr + rrimag
            rrweightarr = rrweightarr + rrweight
            llrealarr = llrealarr + llreal
            llimagarr = llimagarr + llimag
            llweightarr = llweightarr + llweight
            rlrealarr = rlrealarr + rlreal
            rlimagarr = rlimagarr + rlimag
            rlweightarr = rlweightarr + rlweight
            lrrealarr = lrrealarr + lrreal
            lrimagarr = lrimagarr + lrimag
            lrweightarr = lrweightarr + lrweight

        
        # Convert the lists to numpy arrays.
        ifarr, timearr, sourcearr, ant1arr, ant2arr, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr = \
            np.array(ifarr), np.array(timearr), np.array(sourcearr), np.array(ant1arr), np.array(ant2arr), np.array(uarr), np.array(varr), np.array(rrrealarr), np.array(rrimagarr), np.array(rrweightarr), np.array(llrealarr), np.array(llimagarr), np.array(llweightarr), \
            np.array(rlrealarr), np.array(rlimagarr), np.array(rlweightarr), np.array(lrrealarr), np.array(lrimagarr), np.array(lrweightarr)
        
        rlweightarr[rlweightarr > self.remove_weight_outlier_threshold * np.median(rlweightarr)] = np.median(rlweightarr)
        lrweightarr[lrweightarr > self.remove_weight_outlier_threshold * np.median(lrweightarr)] = np.median(lrweightarr)
        
        # Combine the numpy arrays into a single pandas dataframe.
        self.zbl_data.loc[:,"IF"], self.zbl_data.loc[:,"time"], self.zbl_data.loc[:,"source"], self.zbl_data.loc[:,"ant1"], self.zbl_data.loc[:,"ant2"], self.zbl_data.loc[:,"u"], self.zbl_data.loc[:,"v"], \
        self.zbl_data.loc[:,"rrreal"], self.zbl_data.loc[:,"rrimag"], self.zbl_data.loc[:,"rrweight"], self.zbl_data.loc[:,"llreal"], self.zbl_data.loc[:,"llimag"], self.zbl_data.loc[:,"llweight"], \
        self.zbl_data.loc[:,"rlreal"], self.zbl_data.loc[:,"rlimag"], self.zbl_data.loc[:,"rlweight"], self.zbl_data.loc[:,"lrreal"], self.zbl_data.loc[:,"lrimag"], self.zbl_data.loc[:,"lrweight"] = \
                ifarr, timearr * 24., sourcearr, ant1arr - 1, ant2arr - 1, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr
        
        
        zblant = self.zblant[0]
        select = (ant1arr-1 == zbl_antname.index(zblant[0])) & (ant2arr-1 == zbl_antname.index(zblant[1]))
        for j in range(len(self.zblant)):
            zblant = self.zblant[j]
            dumselect = (ant1arr-1 == zbl_antname.index(zblant[0])) & (ant2arr-1 == zbl_antname.index(zblant[1]))
            select = np.logical_or(select, dumselect)
            
        # Filter bad data points.
        select = select & (rrweightarr > 0.) & (llweightarr > 0.) & (rlweightarr > 0.) & (lrweightarr > 0.) & (~np.isnan(rrweightarr)) & (~np.isnan(llweightarr)) & (~np.isnan(rlweightarr)) & (~np.isnan(lrweightarr))
        
        
        self.zbl_data = self.zbl_data.loc[select].reset_index(drop=True)
        
        dumant1 = self.zbl_data.loc[:,"ant1"]
        dumant2 = self.zbl_data.loc[:,"ant2"]
        dumsource = self.zbl_data.loc[:,"source"]

        longarr1, latarr1, f_el1, f_par1, phi_off1 = oh.coordarr(lonarr, latarr, f_el, f_par, phi_off, dumant1)
        longarr2, latarr2, f_el2, f_par2, phi_off2 = oh.coordarr(lonarr, latarr, f_el, f_par, phi_off, dumant2)
        
        yeararr, montharr, dayarr, raarr, decarr = oh.calendar(dumsource, self.zblcalsour, year, month, day, obsra, obsdec)
        
        timearr = np.array(self.zbl_data.loc[:,"time"])
                
        for i in range(10):
            dayarr[timearr>=24.] += 1 
            timearr[timearr>=24.] -= 24. 
                
        self.zbl_data.loc[:,"year"] = yeararr
        self.zbl_data.loc[:,"month"] = montharr
        self.zbl_data.loc[:,"day"] = dayarr
        
        self.zbl_data.loc[:,"pang1"] = oh.get_parang(yeararr, montharr, dayarr, timearr, raarr, decarr, longarr1, latarr1, f_el1, f_par1, phi_off1)
        self.zbl_data.loc[:,"pang2"] = oh.get_parang(yeararr, montharr, dayarr, timearr, raarr, decarr, longarr2, latarr2, f_el2, f_par2, phi_off2)          
        
        self.zbl_data = oh.pd_modifier(self.zbl_data)

    

    def get_data(self):
        """
        Make a pandas dataframe containing UV data and models for the D-term estimation using the similarity assumption.
        
        """    
        
        AIPS.userno = self.aips_userno
        
        if self.aipslog:
            AIPS.log = open(self.logfile, 'a')
        AIPSTask.msgkill = -1
        
        self.obsra, self.obsdec, self.year, self.month, self.day = [], [], [], [], []


        ifarr, timearr, sourcearr, ant1arr, ant2arr, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        pdkey = ["IF", "year", "month", "day", "time", "source", "ant1", "ant2", "u", "v", "pang1", "pang2", "rrreal", "rrimag", "rrsigma", "llreal", "llimag", "llsigma", "rlreal", "rlimag", "rlsigma", "lrreal", "lrimag", "lrsigma", \
                 "rramp", "rrphas", "rramp_sigma", "rrphas_sigma", "llamp", "llphas", "llamp_sigma", "llphas_sigma", "rlamp", "rlphas", "rlamp_sigma", "rlphas_sigma", "lramp", "lrphas", "lramp_sigma", "lrphas_sigma", \
                 "qamp", "qphas", "qamp_sigma", "qphas_sigma", "qsigma", "uamp", "uphas", "uamp_sigma", "uphas_sigma", "usigma"]

        orig_cnum = copy.deepcopy(self.cnum)
        
        for i in range(max(self.cnum)):
            pdkey.append("model"+str(i+1)+"_amp")
            pdkey.append("model"+str(i+1)+"_phas")
            
        self.data = pd.DataFrame(columns = pdkey)


        info = oh.basic_info(self.calsour, self.direc, self.dataname)
        
        self.obsra = info['obsra']
        self.obsdec = info['obsdec']
        self.year = info['year']
        self.month = info['month']
        self.day = info['day']
        self.antname = info['antname']
        self.antx = info['antx']
        self.anty = info['anty']
        self.antz = info['antz']
        self.antmount = info['antmount']
        self.ifnum = info['ifnum']
        self.freq = info['freq']
        self.f_par = info['f_par']
        self.f_el = info['f_el']
        self.phi_off = info['phi_off']
        
        self.lonarr, self.latarr, self.heightarr = oh.coord(self.antname, self.antx, self.anty, self.antz)
        
        self.nant = len(self.antname)
        

            
        self.logger.info('\nGetting data for {:d} sources for {:d} IFs...\n'.format(len(self.calsour), self.ifnum))
        
        
        # Add source sub-model columns to the pandas dataframe.
        index = []
        if self.zblcal: index.append("zbl")
        index.append("simil")
        if self.selfpol:
            for it in range(self.selfpoliter):
                index.append("pol_iter"+str(it+1))
        self.chisq = pd.DataFrame(index = index, columns = ['IF'+str(it+1) for it in np.arange(self.ifnum)])
        
        
        for l in range(len(self.calsour)):
            
            dumdata = pd.DataFrame(columns = pdkey)
        
            inname = str(self.calsour[l])
        
            data = AIPSUVData(inname, 'EDIT', 1, 1)
            if(data.exists() == True):
                data.clrstat()
                data.zap()
            
            cmap = AIPSImage(inname, 'CMAP', 1, 1)
            if(cmap.exists() == True):
                cmap.clrstat()
                cmap.zap()
                
            # Load UVFITS and image fits files.
            au.runfitld(inname, 'EDIT', self.direc+self.dataname+self.calsour[l]+'.uvf')
            au.runfitld(inname, 'CMAP', self.direc+self.dataname+self.calsour[l]+'.fits')
        
                
            data = AIPSUVData(inname, 'EDIT', 1, 1)
            
            # Perform additional self-calibration with CALIB if requested.
            if self.selfcal:            
                if(self.cnum[l] >= 2):
                    au.runccedt(inname, self.direc+'gpcal/'+self.dataname+self.calsour[l]+'.box', self.cnum[l], self.autoccedt[l])
                
                # If automatic CCEDT is performed, then sometimes the output number of sub-models is not the same as the input number. Correct for cnum if this is the case.
                if self.autoccedt[l]:
                    if(self.cnum[l] != 1):
                        dumcnum = 0
                        
                        for cn in cmap.tables:
                            if(cn[1] == 'AIPS CC'):
                                dumcnum += 1
                        
                        self.cnum[l] = dumcnum - 1
                    
                
                calib = AIPSUVData(inname, 'CALIB', 1, 1)
                if(calib.exists() == True):
                    calib.clrstat()
                    calib.zap()
                
                au.runcalib(inname, 'EDIT', inname, 'CMAP', 'CALIB', self.solint, self.soltype, self.solmode, self.weightit)
                
                # Export the self-calibrated UVFITS files to the working directory.
                if path.exists(self.direc + self.dataname + self.calsour[l] + '.calib'):
                    os.system('rm ' + self.direc + self.dataname + self.calsour[l] + '.calib')
                au.runfittp(inname, 'CALIB', self.direc + self.dataname + self.calsour[l] + '.calib')
                
                calib = AIPSUVData(inname, 'CALIB', 1, 1)
                
                # Obtain model visibilities for each visibility measurement for each sub-model.
                if(self.cnum[l] >= 2):
                    for m in range(self.cnum[l]):
                        
                        moddata = AIPSUVData(inname, 'UVSUB', 1, m+1)
                        if(moddata.exists() == True):
                            moddata.clrstat()
                            moddata.zap()
                            
                        au.runuvsub(inname, 'CALIB', 'CMAP', m+2, m+1)
                        
                elif (self.cnum[l] == 1):
                    moddata = AIPSUVData(inname, 'UVSUB', 1, 1)
                    if(moddata.exists() == True):
                        moddata.clrstat()
                        moddata.zap()
                    
                    au.runuvsub(inname, 'CALIB', 'CMAP', 1, 1)
                
                
                calib = WAIPSUVData(inname, 'CALIB', 1, 1)
                
                
                # Extract UV data.
                dumu, dumv, ifname, time, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight = \
                    oh.uvprt(calib, "all")
                
                # Stack the data lists.
                uarr = uarr + dumu
                varr = varr + dumv
                sourcearr = sourcearr + [self.calsour[l]] * len(time)
                ifarr = ifarr + ifname
                timearr = timearr + time
                ant1arr = ant1arr + ant1
                ant2arr = ant2arr + ant2
                rrrealarr = rrrealarr + rrreal
                rrimagarr = rrimagarr + rrimag
                rrweightarr = rrweightarr + rrweight
                llrealarr = llrealarr + llreal
                llimagarr = llimagarr + llimag
                llweightarr = llweightarr + llweight
                rlrealarr = rlrealarr + rlreal
                rlimagarr = rlimagarr + rlimag
                rlweightarr = rlweightarr + rlweight
                lrrealarr = lrrealarr + lrreal
                lrimagarr = lrimagarr + lrimag
                lrweightarr = lrweightarr + lrweight


                # Append the source model arrays to the dataframe.
                if(self.cnum[l] == 0):
                    nanarr = np.empty(len(time))
                    nanarr[:] = np.NaN
                    dumdata.loc[:,"model1_amp"] = nanarr
                else:
                    for m in range(self.cnum[l]):
                        
                        moddata = WAIPSUVData(inname, 'UVSUB', 1, m+1)
                        
                        modreal = np.array(oh.uvprt(moddata, "rrreal"), dtype = 'float64')
                        modimag = np.array(oh.uvprt(moddata, "rrimag"), dtype = 'float64')
                            
                        dumdata.loc[:,"model"+str(m+1)+"_amp"] = np.absolute(modreal + 1j*modimag)
                        dumdata.loc[:,"model"+str(m+1)+"_phas"] = np.angle(modreal + 1j*modimag)


                # Run LPCAL and export the antenna files if requested.
                if self.lpcal:
                    au.runlpcal(inname, 'CALIB', 'CMAP', self.cnum[l])
                    
                    au.runprtab(calib, self.direc+'gpcal/'+self.dataname+self.calsour[l]+'.an')
            
            else:
                # Repeat the same procedure except for additional self-calibration.
                if(self.cnum[l] >= 2):
                    au.runccedt(inname, self.direc+'gpcal/' + self.dataname+self.calsour[l]+'.box', self.cnum[l], self.autoccedt[l])
                
                if self.autoccedt[l]:
                    
                    if(self.cnum[l] != 1):
                        dumcnum = 0
                        
                        for cn in cmap.tables:
                            if(cn[1] == 'AIPS CC'):
                                dumcnum += 1
                        
                        self.cnum[l] = dumcnum - 1
                    
                
                if(self.cnum[l] >= 2):
                    for m in range(self.cnum[l]):
                        
                        moddata = AIPSUVData(inname, 'UVSUB', 1, m+1)
                        if(moddata.exists() == True):
                            moddata.clrstat()
                            moddata.zap()
                            
                        au.runuvsub(inname, 'EDIT', 'CMAP', m+2, m+1)
                elif (self.cnum[l] == 1):
                    moddata = AIPSUVData(inname, 'UVSUB', 1, 1)
                    if(moddata.exists() == True):
                        moddata.clrstat()
                        moddata.zap()
                        
                    au.runuvsub(inname, 'EDIT', 'CMAP', 1, 1)
                    
                    
                data = WAIPSUVData(inname, 'EDIT', 1, 1)
                
                dumu, dumv, ifname, time, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight = \
                    oh.uvprt(data, "all")
                
                uarr = uarr + dumu
                varr = varr + dumv
                sourcearr = sourcearr + [self.calsour[l]] * len(time)
                ifarr = ifarr + ifname
                timearr = timearr + time
                ant1arr = ant1arr + ant1
                ant2arr = ant2arr + ant2
                rrrealarr = rrrealarr + rrreal
                rrimagarr = rrimagarr + rrimag
                rrweightarr = rrweightarr + rrweight
                llrealarr = llrealarr + llreal
                llimagarr = llimagarr + llimag
                llweightarr = llweightarr + llweight
                rlrealarr = rlrealarr + rlreal
                rlimagarr = rlimagarr + rlimag
                rlweightarr = rlweightarr + rlweight
                lrrealarr = lrrealarr + lrreal
                lrimagarr = lrimagarr + lrimag
                lrweightarr = lrweightarr + lrweight
                
                
                if(self.cnum[l] == 0):
                    nanarr = np.empty(len(time))
                    nanarr[:] = np.NaN
                    dumdata.loc[:,"model1_amp"] = nanarr
                else:
                    for m in range(self.cnum[l]):
                        moddata = WAIPSUVData(inname, 'UVSUB', 1, m+1)
                        
                        modreal = np.array(oh.uvprt(moddata, "rrreal"), dtype = 'float64')
                        modimag = np.array(oh.uvprt(moddata, "rrimag"), dtype = 'float64')
                        
                        dumdata.loc[:,"model"+str(m+1)+"_amp"] = np.absolute(modreal + 1j*modimag)
                        dumdata.loc[:,"model"+str(m+1)+"_phas"] = np.angle(modreal + 1j*modimag)

                
                if self.lpcal:
                    au.runlpcal(inname, 'EDIT', 'CMAP', self.cnum[l])
                    lpcal = AIPSUVData(inname, 'EDIT', 1, 1)
                    au.runprtab(lpcal, self.direc+'gpcal/'+self.dataname+self.calsour[l]+'.an')
            
            if(self.cnum[l] >= 2):
                for m in range(self.cnum[l]):
                    moddata = AIPSUVData(inname, 'UVSUB', 1, m+1)
                    moddata.zap()
            elif (self.cnum[l] == 1):
                moddata = AIPSUVData(inname, 'UVSUB', 1, 1)
                moddata.zap()
                
            data.zap()
            
            cmap = AIPSImage(inname, 'CMAP', 1, 1)
            cmap.zap()
            
            if self.selfcal:
                calib.zap()
            
            
            self.data = self.data.append(dumdata, ignore_index=True, sort=False)
        
        
        if(max(orig_cnum) > max(self.cnum)):
            for i in range(max(self.cnum), max(orig_cnum)):
                del self.data["model"+str(i+1)+"_amp"]
                del self.data["model"+str(i+1)+"_phas"]
                
        
        # Convert the lists to numpy arrays.
        ifarr, timearr, sourcearr, ant1arr, ant2arr, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr = \
            np.array(ifarr), np.array(timearr), np.array(sourcearr), np.array(ant1arr), np.array(ant2arr), np.array(uarr), np.array(varr), np.array(rrrealarr), np.array(rrimagarr), np.array(rrweightarr), np.array(llrealarr), np.array(llimagarr), np.array(llweightarr), \
            np.array(rlrealarr), np.array(rlimagarr), np.array(rlweightarr), np.array(lrrealarr), np.array(lrimagarr), np.array(lrweightarr)
        
        rlweightarr[rlweightarr > self.remove_weight_outlier_threshold * np.median(rlweightarr)] = np.median(rlweightarr)
        lrweightarr[lrweightarr > self.remove_weight_outlier_threshold * np.median(lrweightarr)] = np.median(lrweightarr)
        
        # Combine the numpy arrays into a single pandas dataframe.
        self.data.loc[:,"IF"], self.data.loc[:,"time"], self.data.loc[:,"source"], self.data.loc[:,"ant1"], self.data.loc[:,"ant2"], self.data.loc[:,"u"], self.data.loc[:,"v"], \
        self.data.loc[:,"rrreal"], self.data.loc[:,"rrimag"], self.data.loc[:,"rrweight"], self.data.loc[:,"llreal"], self.data.loc[:,"llimag"], self.data.loc[:,"llweight"], \
        self.data.loc[:,"rlreal"], self.data.loc[:,"rlimag"], self.data.loc[:,"rlweight"], self.data.loc[:,"lrreal"], self.data.loc[:,"lrimag"], self.data.loc[:,"lrweight"] = \
                ifarr, timearr * 24., sourcearr, ant1arr - 1, ant2arr - 1, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr
        
        # Filter bad data points.
        select = (rrweightarr > 0.) & (llweightarr > 0.) & (rlweightarr > 0.) & (lrweightarr > 0.) & (~np.isnan(rrweightarr)) & (~np.isnan(llweightarr)) & (~np.isnan(rlweightarr)) & (~np.isnan(lrweightarr))
        
        self.data = self.data.loc[select].reset_index(drop=True)
   
        dumant1 = self.data.loc[:,"ant1"]
        dumant2 = self.data.loc[:,"ant2"]
        dumsource = self.data.loc[:,"source"]
        
        longarr1, latarr1, f_el1, f_par1, phi_off1 = oh.coordarr(self.lonarr, self.latarr, self.f_el, self.f_par, self.phi_off, dumant1)
        longarr2, latarr2, f_el2, f_par2, phi_off2 = oh.coordarr(self.lonarr, self.latarr, self.f_el, self.f_par, self.phi_off, dumant2)
        
        yeararr, montharr, dayarr, raarr, decarr = oh.calendar(dumsource, self.calsour, self.year, self.month, self.day, self.obsra, self.obsdec)
        
        timearr = np.array(self.data.loc[:,"time"])
        
        for i in range(10):
            dayarr[timearr>=24.] += 1 
            timearr[timearr>=24.] -= 24. 
        
        self.data.loc[:,"time"] = timearr
        self.data.loc[:,"year"] = yeararr
        self.data.loc[:,"month"] = montharr
        self.data.loc[:,"day"] = dayarr
        
        self.data.loc[:,"pang1"] = oh.get_parang(yeararr, montharr, dayarr, timearr, raarr, decarr, longarr1, latarr1, f_el1, f_par1, phi_off1)
        self.data.loc[:,"pang2"] = oh.get_parang(yeararr, montharr, dayarr, timearr, raarr, decarr, longarr2, latarr2, f_el2, f_par2, phi_off2)          
        
        self.data = oh.pd_modifier(self.data)

        for i in range(max(self.cnum)):
            self.data["model"+str(i+1)+"_amp"] = self.data["model"+str(i+1)+"_amp"].astype('float64')
            self.data["model"+str(i+1)+"_phas"] = self.data["model"+str(i+1)+"_phas"].astype('float64')
            
        
    def get_pol_data(self):
        """
        Make a pandas dataframe containing UV data and models for the D-term estimation with instrumental polarization self-calibration.
        
        """
        
        # If polcalsour == calsour, then don't repeat the procedure and just copy the data array that was already made.
        if(np.sort(self.calsour).tolist() == np.sort(self.polcalsour).tolist()):
            self.logger.info('calsour and polcalsour are the same. Will not load the data for polcalsour again.\n')
            
            self.pol_obsra = self.obsra
            self.pol_obsdec = self.obsdec
            
            self.pol_data = copy.deepcopy(self.data)
            
            for i in range(max(self.cnum)):
                del self.pol_data["model"+str(i+1)+"_amp"]
                del self.pol_data["model"+str(i+1)+"_phas"]
                        
            return
        
        
        info = oh.basic_info(self.polcalsour, self.direc, self.dataname)
        
        self.pol_obsra = info['obsra']
        self.pol_obsdec = info['obsdec']
        self.pol_year = info['year']
        self.pol_month = info['month']
        self.pol_day = info['day']
        self.antname = info['antname']
        self.antx = info['antx']
        self.anty = info['anty']
        self.antz = info['antz']
        self.antmount = info['antmount']
        self.ifnum = info['ifnum']
        self.freq = info['freq']
        self.f_par = info['f_par']
        self.f_el = info['f_el']
        self.phi_off = info['phi_off']
        
        self.lonarr, self.latarr, self.heightarr = oh.coord(self.antname, self.antx, self.anty, self.antz)
        
        self.nant = len(self.antname)
        
        self.logger.info('\nGetting data for {:d} sources for {:d} IFs...'.format(len(self.polcalsour), self.ifnum))
            
        AIPS.userno = self.aips_userno
        
        if self.aipslog:
            AIPS.log = open(self.logfile, 'a')
        AIPSTask.msgkill = -1
        
        pdkey = ["IF", "year", "month", "day", "time", "source", "ant1", "ant2", "u", "v", "pang1", "pang2", \
                 "rrreal", "rrimag", "rrsigma", "llreal", "llimag", "llsigma", "rlreal", "rlimag", "rlsigma", "lrreal", "lrimag", "lrsigma", \
                 "rramp", "rrphas", "rramp_sigma", "rrphas_sigma", "llamp", "llphas", "llamp_sigma", "llphas_sigma", "rlamp", "rlphas", "rlamp_sigma", "rlphas_sigma", "lramp", "lrphas", "lramp_sigma", "lrphas_sigma", \
                 "qamp", "qphas", "qamp_sigma", "qphas_sigma", "qsigma", "uamp", "uphas", "uamp_sigma", "uphas_sigma", "usigma"]

            
        self.pol_data = pd.DataFrame(columns = pdkey)       
        
        
        ifarr, timearr, sourcearr, ant1arr, ant2arr, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []


        for l in range(len(self.polcalsour)):
        
            inname = str(self.polcalsour[l])

            # Perform additional self-calibration with CALIB if requested.
            if self.selfcal:
                if not self.polcalsour[l] in self.calsour:
        
                    data = AIPSUVData(inname, 'EDIT', 1, 1)
                    if(data.exists() == True):
                        data.clrstat()
                        data.zap()
                
                    au.runfitld(inname, 'EDIT', self.direc+self.dataname+self.polcalsour[l]+'.uvf')
                    
                    cmap = AIPSImage(inname, 'CMAP', 1, 1)
                    if(cmap.exists() == True):
                        cmap.clrstat()
                        cmap.zap()
                        
                    au.runfitld(inname, 'CMAP', self.direc+self.dataname+self.polcalsour[l]+'.fits')
                        
                    calib = AIPSUVData(inname, 'CALIB', 1, 1)
                    if(calib.exists() == True):
                        calib.clrstat()
                        calib.zap()
                    
                    au.runcalib(inname, 'EDIT', inname, 'CMAP', 'CALIB', self.solint, self.soltype, self.solmode, self.weightit)
                    
                    if path.exists(self.direc + self.dataname + self.polcalsour[l] + '.calib'):
                        os.system('rm ' + self.direc + self.dataname + self.polcalsour[l] + '.calib')
                    au.runfittp(inname, 'CALIB', self.direc + self.dataname + self.polcalsour[l] + '.calib')
                    
                    data.zap()
                    cmap.zap()
                
                else:
                    calib = AIPSUVData(inname, 'CALIB', 1, 1)
                    if(calib.exists() == True):
                        calib.clrstat()
                        calib.zap()
                        
                    au.runfitld(inname, 'CALIB', self.direc+self.dataname+self.polcalsour[l]+'.calib')
                
            else:                    
                calib = AIPSUVData(inname, 'CALIB', 1, 1)
                if(calib.exists() == True):
                    calib.clrstat()
                    calib.zap()
                au.runfitld(inname, 'CALIB', self.direc+self.dataname+self.polcalsour[l]+'.uvf')
                
                    
            calib = WAIPSUVData(inname, 'CALIB', 1, 1)
            
            dumu, dumv, ifname, time, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight = \
                oh.uvprt(calib, "all")
            
            uarr = uarr + dumu
            varr = varr + dumv
            sourcearr = sourcearr + [self.polcalsour[l]] * len(time)
            ifarr = ifarr + ifname
            timearr = timearr + time
            ant1arr = ant1arr + ant1
            ant2arr = ant2arr + ant2
            rrrealarr = rrrealarr + rrreal
            rrimagarr = rrimagarr + rrimag
            rrweightarr = rrweightarr + rrweight
            llrealarr = llrealarr + llreal
            llimagarr = llimagarr + llimag
            llweightarr = llweightarr + llweight
            rlrealarr = rlrealarr + rlreal
            rlimagarr = rlimagarr + rlimag
            rlweightarr = rlweightarr + rlweight
            lrrealarr = lrrealarr + lrreal
            lrimagarr = lrimagarr + lrimag
            lrweightarr = lrweightarr + lrweight
                        
            calib.zap()


        # Convert the lists to numpy arrays.
        ifarr, timearr, sourcearr, ant1arr, ant2arr, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr = \
            np.array(ifarr), np.array(timearr), np.array(sourcearr), np.array(ant1arr), np.array(ant2arr), np.array(uarr), np.array(varr), np.array(rrrealarr), np.array(rrimagarr), np.array(rrweightarr), np.array(llrealarr), np.array(llimagarr), np.array(llweightarr), \
            np.array(rlrealarr), np.array(rlimagarr), np.array(rlweightarr), np.array(lrrealarr), np.array(lrimagarr), np.array(lrweightarr)
        
        rlweightarr[rlweightarr > self.remove_weight_outlier_threshold * np.median(rlweightarr)] = np.median(rlweightarr)
        lrweightarr[lrweightarr > self.remove_weight_outlier_threshold * np.median(lrweightarr)] = np.median(lrweightarr)
        
        # Combine the numpy arrays into a single pandas dataframe.
        self.pol_data.loc[:,"IF"], self.pol_data.loc[:,"time"], self.pol_data.loc[:,"source"], self.pol_data.loc[:,"ant1"], self.pol_data.loc[:,"ant2"], self.pol_data.loc[:,"u"], self.pol_data.loc[:,"v"], \
        self.pol_data.loc[:,"rrreal"], self.pol_data.loc[:,"rrimag"], self.pol_data.loc[:,"rrweight"], self.pol_data.loc[:,"llreal"], self.pol_data.loc[:,"llimag"], self.pol_data.loc[:,"llweight"], \
        self.pol_data.loc[:,"rlreal"], self.pol_data.loc[:,"rlimag"], self.pol_data.loc[:,"rlweight"], self.pol_data.loc[:,"lrreal"], self.pol_data.loc[:,"lrimag"], self.pol_data.loc[:,"lrweight"] = \
                ifarr, timearr * 24., sourcearr, ant1arr - 1, ant2arr - 1, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr
        
        # Filter bad data points.
        select = (rrweightarr > 0.) & (llweightarr > 0.) & (rlweightarr > 0.) & (lrweightarr > 0.) & (~np.isnan(rrweightarr)) & (~np.isnan(llweightarr)) & (~np.isnan(rlweightarr)) & (~np.isnan(lrweightarr))
        
        self.pol_data = self.pol_data.loc[select].reset_index(drop=True)
        
        dumant1 = self.pol_data.loc[:,"ant1"]
        dumant2 = self.pol_data.loc[:,"ant2"]
        dumsource = self.pol_data.loc[:,"source"]
                
        longarr1, latarr1, f_el1, f_par1, phi_off1 = oh.coordarr(self.lonarr, self.latarr, self.f_el, self.f_par, self.phi_off, dumant1)
        longarr2, latarr2, f_el2, f_par2, phi_off2 = oh.coordarr(self.lonarr, self.latarr, self.f_el, self.f_par, self.phi_off, dumant2)
        
        yeararr, montharr, dayarr, raarr, decarr = oh.calendar(dumsource, self.polcalsour, self.pol_year, self.pol_month, self.pol_day, self.pol_obsra, self.pol_obsdec)
        
        timearr = np.array(self.pol_data.loc[:,"time"])
        
        for i in range(10):
            dayarr[timearr>=24.] += 1
            timearr[timearr>=24.] -= 24.
        
        self.pol_data.loc[:,"time"] = timearr
        self.pol_data.loc[:,"year"] = yeararr
        self.pol_data.loc[:,"month"] = montharr
        self.pol_data.loc[:,"day"] = dayarr
        
        self.pol_data.loc[:,"pang1"] = oh.get_parang(yeararr, montharr, dayarr, timearr, raarr, decarr, longarr1, latarr1, f_el1, f_par1, phi_off1)
        self.pol_data.loc[:,"pang2"] = oh.get_parang(yeararr, montharr, dayarr, timearr, raarr, decarr, longarr2, latarr2, f_el2, f_par2, phi_off2)

        self.pol_data = oh.pd_modifier(self.pol_data)
        



    def parangplot(self, k, nant, antname, source, time, ant1, ant2, sourcearr, pang1, pang2, filename):
        """
        Draw field-rotation angle plots.
        
        Args:
            k (int): IF number - 1.
            nant (int): the number of all antennas.
            antname (list): a list of the antenna names.
            source (list): a list of calibrators to be plotted.
            time (numpy array): a numpy array of time of visibilities.
            ant1 (numpy array): a numpy array of the first antenna number of visibilities.
            ant2 (numpy array): a numpy array of the second antenna number of visibilities.
            sourcearr (numpy array): a numpy array of the sources of visibilities.
            pang1 (numpy array): a numpy array of the field-rotation angles of the first antenna of visibilities.
            pang2 (numpy array): a numpy array of the field-rotation angles of the second antenna of visibilities.
            filename (str): the name of the output figure.
        """
        
        for m in range(nant):
            
            if(sum(ant1 == m) == 0) & (sum(ant2 == m) == 0): continue
                
            fig, ax = plt.subplots(figsize=(10, 8))
            
            ax.tick_params(length=6, width=2,which = 'major')
            ax.tick_params(length=4, width=1.5,which = 'minor')
            
            ax.set_xlim(np.min(time) - (np.max(time) - np.min(time)) * 0.35, np.max(time) + (np.max(time) - np.min(time)) * 0.2)
        
            ax.set(xlabel = 'Time (UT)')
            ax.set(ylabel = 'Field rotation angle (deg)')
            
            ax.annotate(self.antname[m], xy=(1, 1), xycoords = 'axes fraction', xytext = (-25, -25), size = 24, textcoords='offset pixels', horizontalalignment='right', verticalalignment='top')
            ax.annotate('IF {:d}'.format(k+1), xy = (0, 0), xycoords = 'axes fraction', xytext = (25, 25), size = 24, textcoords='offset pixels', horizontalalignment='left', verticalalignment='bottom')
            
            for l in range(len(source)):
                
                select = (ant1 == m) & (sourcearr == source[l])
                            
                dumx = time[select]
                dumy = np.degrees(pang1[select])
                        
                ax.scatter(dumx, dumy, s = 30, marker = 'o', facecolor = 'None', edgecolor = self.colors[l], label = source[l].upper(), zorder = 0)
                
                select = (ant2 == m)
                select = select & (sourcearr == source[l])
                            
                dumx = time[select]
                dumy = np.degrees(pang2[select])
                        
                ax.scatter(dumx, dumy, s = 30, marker = 'o', facecolor = 'None', edgecolor = self.colors[l], zorder = 0)
                
            ax.legend(loc='upper left', fontsize = 18 - int(len(source)/2.), frameon=False, markerfirst=True, handlelength = 1.0)
    
            fig.savefig(filename + '.' + antname[m]+'.'+self.filetype, bbox_inches = 'tight')
            
            plt.close('all')


    def deq(self, x, *p):
        """
        The D-term models for the initial D-term estimation using the similarity assumption.
        
        Args:
            x: dummy parameters (not to be used).
            *p (args): the best-fit parameters args.
        """
        
        RiLj_Real = np.zeros(len(self.pang1))
        RiLj_Imag = np.zeros(len(self.pang1))
        LiRj_Real = np.zeros(len(self.pang1))
        LiRj_Imag = np.zeros(len(self.pang1))
        
        dump = np.array(p)
        
        dumreal = dump[2*self.ant1]
        dumimag = dump[2*self.ant1 + 1]
        Tot_D_iR_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_iR_phas = np.angle(dumreal + 1j*dumimag)

        dumreal = dump[2*self.nant + 2*self.ant2]
        dumimag = dump[2*self.nant + 2*self.ant2 + 1]
        Tot_D_jL_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_jL_phas = np.angle(dumreal + 1j*dumimag)
        
        dumreal = dump[2*self.nant + 2*self.ant1]
        dumimag = dump[2*self.nant + 2*self.ant1 + 1]
        Tot_D_iL_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_iL_phas = np.angle(dumreal + 1j*dumimag)
        
        dumreal = dump[2*self.ant2]
        dumimag = dump[2*self.ant2 + 1]
        Tot_D_jR_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_jR_phas = np.angle(dumreal + 1j*dumimag)


        P_ij_amp = [np.absolute(it1 + 1j*it2) for it1, it2 in zip([p[self.nant * 4 + s * 2] for s in range(sum(self.cnum))], [p[self.nant * 4 + s * 2 + 1] for s in range(sum(self.cnum))])]
        P_ij_phas = [np.angle(it1 + 1j*it2) for it1, it2 in zip([p[self.nant * 4 + s * 2] for s in range(sum(self.cnum))], [p[self.nant * 4 + s * 2 + 1] for s in range(sum(self.cnum))])]
        
        for l in range(len(self.calsour)):
            select = (self.sourcearr == self.calsour[l])
            
            if(self.cnum[l] != 0.):
                for t in range(self.cnum[l]):
                    if(l==0):
                        dummodamp = np.array(self.modamp[t])
                        dummodphas = np.array(self.modphas[t])
                        Pick = t
                    else:
                        dummodamp = np.array(self.modamp[sum(self.cnum[0:l])+t])
                        dummodphas = np.array(self.modphas[sum(self.cnum[0:l])+t])
                        Pick = sum(self.cnum[0:l]) + t
                        
                    submodamp = dummodamp[select]
                    submodphas = dummodphas[select]
                    
            
                    Pamp = P_ij_amp[Pick]
                    Pphas = P_ij_phas[Pick]
                    
        
                    RiLj_Real[select] += Pamp * submodamp * np.cos(submodphas + Pphas) + \
                      Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * Pamp * submodamp * np.cos(submodphas + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] - Pphas + 2. * (self.pang1[select] + self.pang2[select]))
            
                    RiLj_Imag[select] += Pamp * submodamp * np.sin(submodphas + Pphas) + \
                      Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * Pamp * submodamp * np.sin(submodphas + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] - Pphas + 2. * (self.pang1[select] + self.pang2[select]))
            
                    LiRj_Real[select] += Pamp * submodamp * np.cos(submodphas - Pphas) + \
                      Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * Pamp * submodamp * np.cos(submodphas + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] + Pphas - 2. * (self.pang1[select] + self.pang2[select]))
            
                    LiRj_Imag[select] += Pamp * submodamp * np.sin(submodphas - Pphas) + \
                      Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * Pamp * submodamp * np.sin(submodphas + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] + Pphas - 2. * (self.pang1[select] + self.pang2[select]))     
        
        
        RiLj_Real += \
          Tot_D_iR_amp * self.llamp * np.cos(Tot_D_iR_phas + self.llphas + 2. * self.pang1) + \
          Tot_D_jL_amp * self.rramp * np.cos(-Tot_D_jL_phas + self.rrphas + 2. * self.pang2)
        
        RiLj_Imag += \
          Tot_D_iR_amp * self.llamp * np.sin(Tot_D_iR_phas + self.llphas + 2. * self.pang1) + \
          Tot_D_jL_amp * self.rramp * np.sin(-Tot_D_jL_phas + self.rrphas + 2. * self.pang2)
    
        LiRj_Real += \
          Tot_D_iL_amp * self.rramp * np.cos(Tot_D_iL_phas + self.rrphas - 2. * self.pang1) + \
          Tot_D_jR_amp * self.llamp * np.cos(-Tot_D_jR_phas + self.llphas - 2. * self.pang2)
    
        LiRj_Imag += \
          Tot_D_iL_amp * self.rramp * np.sin(Tot_D_iL_phas + self.rrphas - 2. * self.pang1) + \
          Tot_D_jR_amp * self.llamp * np.sin(-Tot_D_jR_phas + self.llphas - 2. * self.pang2)
       
    
        compv = np.concatenate([LiRj_Real, LiRj_Imag, RiLj_Real, RiLj_Imag])
            
        return compv


    def deq_comp(self, comp, *p):
        """
        Extract component-wise best-fit models using the similarity assumption.
        
        Args:
            comp (str): the name of the component to be extracted.
            *p (args): the best-fit parameters args.
        """
        
        pol_RiLj_Real = np.zeros(len(self.pang1))
        pol_RiLj_Imag = np.zeros(len(self.pang1))
        pol_LiRj_Real = np.zeros(len(self.pang1))
        pol_LiRj_Imag = np.zeros(len(self.pang1))
        
        dterm_RiLj_Real = np.zeros(len(self.pang1))
        dterm_RiLj_Imag = np.zeros(len(self.pang1))
        dterm_LiRj_Real = np.zeros(len(self.pang1))
        dterm_LiRj_Imag = np.zeros(len(self.pang1))
        
        second_RiLj_Real = np.zeros(len(self.pang1))
        second_RiLj_Imag = np.zeros(len(self.pang1))
        second_LiRj_Real = np.zeros(len(self.pang1))
        second_LiRj_Imag = np.zeros(len(self.pang1))        
        
        
        dump = np.array(p)
        

        dumreal = dump[2*self.ant1]
        dumimag = dump[2*self.ant1 + 1]
        Tot_D_iR_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_iR_phas = np.angle(dumreal + 1j*dumimag)

        dumreal = dump[2*self.nant + 2*self.ant2]
        dumimag = dump[2*self.nant + 2*self.ant2 + 1]
        Tot_D_jL_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_jL_phas = np.angle(dumreal + 1j*dumimag)
        
        dumreal = dump[2*self.nant + 2*self.ant1]
        dumimag = dump[2*self.nant + 2*self.ant1 + 1]
        Tot_D_iL_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_iL_phas = np.angle(dumreal + 1j*dumimag)
        
        dumreal = dump[2*self.ant2]
        dumimag = dump[2*self.ant2 + 1]
        Tot_D_jR_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_jR_phas = np.angle(dumreal + 1j*dumimag)

    
        P_ij_amp = [np.absolute(it1 + 1j*it2) for it1, it2 in zip([p[self.nant * 4 + s * 2] for s in range(sum(self.cnum))], [p[self.nant * 4 + s * 2 + 1] for s in range(sum(self.cnum))])]
        P_ij_phas = [np.angle(it1 + 1j*it2) for it1, it2 in zip([p[self.nant * 4 + s * 2] for s in range(sum(self.cnum))], [p[self.nant * 4 + s * 2 + 1] for s in range(sum(self.cnum))])]    
    
    
        for l in range(len(self.calsour)):
            select = (self.sourcearr == self.calsour[l])
            
            if(self.cnum[l] != 0.):
                for t in range(self.cnum[l]):
                    if(l==0):
                        dummodamp = np.array(self.modamp[t])
                        dummodphas = np.array(self.modphas[t])
                        Pick = t
                    else:
                        dummodamp = np.array(self.modamp[sum(self.cnum[0:l])+t])
                        dummodphas = np.array(self.modphas[sum(self.cnum[0:l])+t])
                        Pick = sum(self.cnum[0:l]) + t
            
                    submodamp = dummodamp[select]
                    submodphas = dummodphas[select]
                    
                
                    Pamp = P_ij_amp[Pick]
                    Pphas = P_ij_phas[Pick]
                    
        
                    pol_RiLj_Real[select] += Pamp * submodamp * np.cos(submodphas + Pphas)
                    pol_RiLj_Imag[select] += Pamp * submodamp * np.sin(submodphas + Pphas)
                    pol_LiRj_Real[select] += Pamp * submodamp * np.cos(submodphas - Pphas)
                    pol_LiRj_Imag[select] += Pamp * submodamp * np.sin(submodphas - Pphas)
                    
                    second_RiLj_Imag[select] += Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * Pamp * submodamp * np.sin(submodphas + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] - Pphas + 2. * (self.pang1[select] + self.pang2[select]))
                    second_RiLj_Real[select] += Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * Pamp * submodamp * np.cos(submodphas + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] - Pphas + 2. * (self.pang1[select] + self.pang2[select]))
                    second_LiRj_Real[select] += Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * Pamp * submodamp * np.cos(submodphas + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] + Pphas - 2. * (self.pang1[select] + self.pang2[select]))
                    second_LiRj_Imag[select] += Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * Pamp * submodamp * np.sin(submodphas + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] + Pphas - 2. * (self.pang1[select] + self.pang2[select]))     
            
                  
        dterm_RiLj_Real += Tot_D_iR_amp * self.llamp * np.cos(Tot_D_iR_phas + self.llphas + 2. * self.pang1) + Tot_D_jL_amp * self.rramp * np.cos(-Tot_D_jL_phas + self.rrphas + 2. * self.pang2)
        dterm_RiLj_Imag += Tot_D_iR_amp * self.llamp * np.sin(Tot_D_iR_phas + self.llphas + 2. * self.pang1) + Tot_D_jL_amp * self.rramp * np.sin(-Tot_D_jL_phas + self.rrphas + 2. * self.pang2)
        dterm_LiRj_Real += Tot_D_iL_amp * self.rramp * np.cos(Tot_D_iL_phas + self.rrphas - 2. * self.pang1) + Tot_D_jR_amp * self.llamp * np.cos(-Tot_D_jR_phas + self.llphas - 2. * self.pang2)
        dterm_LiRj_Imag += Tot_D_iL_amp * self.rramp * np.sin(Tot_D_iL_phas + self.rrphas - 2. * self.pang1) + Tot_D_jR_amp * self.llamp * np.sin(-Tot_D_jR_phas + self.llphas - 2. * self.pang2)


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



    def zbl_deq(self, x, *p):
        """
        The D-term models for the zero-baseline D-term estimation.
        
        Args:
            x: dummy parameters (not to be used).
            *p (args): the best-fit parameters args.
        """
        
        RiLj_Real = np.zeros(len(self.pang1))
        RiLj_Imag = np.zeros(len(self.pang1))
        LiRj_Real = np.zeros(len(self.pang1))
        LiRj_Imag = np.zeros(len(self.pang1))
        
        dump = np.array(p)

        dumreal = dump[2*self.ant1]
        dumimag = dump[2*self.ant1 + 1]
        Tot_D_iR_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_iR_phas = np.angle(dumreal + 1j*dumimag)

        dumreal = dump[2*self.zbl_nant + 2*self.ant2]
        dumimag = dump[2*self.zbl_nant + 2*self.ant2 + 1]
        Tot_D_jL_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_jL_phas = np.angle(dumreal + 1j*dumimag)
        
        dumreal = dump[2*self.zbl_nant + 2*self.ant1]
        dumimag = dump[2*self.zbl_nant + 2*self.ant1 + 1]
        Tot_D_iL_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_iL_phas = np.angle(dumreal + 1j*dumimag)
        
        dumreal = dump[2*self.ant2]
        dumimag = dump[2*self.ant2 + 1]
        Tot_D_jR_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_jR_phas = np.angle(dumreal + 1j*dumimag)
        
    
        if(len(self.zblcalsour) == 1):
            P_ij_amp = [np.absolute(p[self.zbl_nant * 4] + 1j * p[self.zbl_nant * 4 + 1])]
            P_ij_phas = [np.angle(p[self.zbl_nant * 4] + 1j*p[self.zbl_nant * 4 + 1])]
        else:
            P_ij_amp = [np.absolute(it1 + 1j*it2) for it1, it2 in zip([p[self.zbl_nant * 4 + s * 2] for s in range(len(self.zblcalsour))], [p[self.zbl_nant * 4 + s * 2 + 1] for s in range(len(self.zblcalsour))])]
            P_ij_phas = [np.angle(it1 + 1j*it2) for it1, it2 in zip([p[self.zbl_nant * 4 + s * 2] for s in range(len(self.zblcalsour))], [p[self.zbl_nant * 4 + s * 2 + 1] for s in range(len(self.zblcalsour))])]
    
    
        for l in range(len(self.zblcalsour)):
            
            select = (self.sourcearr == self.zblcalsour[l])
            
            Pamp = P_ij_amp[l]
            Pphas = P_ij_phas[l]
            

            RiLj_Real[select] += Pamp * np.cos(Pphas) + \
              Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * Pamp * np.cos(Tot_D_iR_phas[select] - Tot_D_jL_phas[select] - Pphas + 2. * (self.pang1[select] + self.pang2[select]))
    
            RiLj_Imag[select] += Pamp * np.sin(Pphas) + \
              Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * Pamp * np.sin(Tot_D_iR_phas[select] - Tot_D_jL_phas[select] - Pphas + 2. * (self.pang1[select] + self.pang2[select]))
    
            LiRj_Real[select] += Pamp * np.cos(-Pphas) + \
              Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * Pamp * np.cos(Tot_D_iL_phas[select] - Tot_D_jR_phas[select] + Pphas - 2. * (self.pang1[select] + self.pang2[select]))
    
            LiRj_Imag[select] += Pamp * np.sin(-Pphas) + \
              Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * Pamp * np.sin(Tot_D_iL_phas[select] - Tot_D_jR_phas[select] + Pphas - 2. * (self.pang1[select] + self.pang2[select]))     
  
        
        RiLj_Real += \
          Tot_D_iR_amp * self.llamp * np.cos(Tot_D_iR_phas + self.llphas + 2. * self.pang1) + \
          Tot_D_jL_amp * self.rramp * np.cos(-Tot_D_jL_phas + self.rrphas + 2. * self.pang2)
        
        RiLj_Imag += \
          Tot_D_iR_amp * self.llamp * np.sin(Tot_D_iR_phas + self.llphas + 2. * self.pang1) + \
          Tot_D_jL_amp * self.rramp * np.sin(-Tot_D_jL_phas + self.rrphas + 2. * self.pang2)
    
        LiRj_Real += \
          Tot_D_iL_amp * self.rramp * np.cos(Tot_D_iL_phas + self.rrphas - 2. * self.pang1) + \
          Tot_D_jR_amp * self.llamp * np.cos(-Tot_D_jR_phas + self.llphas - 2. * self.pang2)
    
        LiRj_Imag += \
          Tot_D_iL_amp * self.rramp * np.sin(Tot_D_iL_phas + self.rrphas - 2. * self.pang1) + \
          Tot_D_jR_amp * self.llamp * np.sin(-Tot_D_jR_phas + self.llphas - 2. * self.pang2)
       
    
        compv = np.concatenate([LiRj_Real, LiRj_Imag, RiLj_Real, RiLj_Imag])
            
        return compv


    def zbl_deq_comp(self, comp, *p):
        """
        Extract component-wise best-fit models for zero-baseline D-term estimation.
        
        Args:
            comp (str): the name of the component to be extracted.
            *p (args): the best-fit parameters args.
        """
        
        pol_RiLj_Real = np.zeros(len(self.pang1))
        pol_RiLj_Imag = np.zeros(len(self.pang1))
        pol_LiRj_Real = np.zeros(len(self.pang1))
        pol_LiRj_Imag = np.zeros(len(self.pang1))
        
        dterm_RiLj_Real = np.zeros(len(self.pang1))
        dterm_RiLj_Imag = np.zeros(len(self.pang1))
        dterm_LiRj_Real = np.zeros(len(self.pang1))
        dterm_LiRj_Imag = np.zeros(len(self.pang1))
        
        second_RiLj_Real = np.zeros(len(self.pang1))
        second_RiLj_Imag = np.zeros(len(self.pang1))
        second_LiRj_Real = np.zeros(len(self.pang1))
        second_LiRj_Imag = np.zeros(len(self.pang1))

        
        dump = np.array(p)
        
        dumreal = dump[2*self.ant1]
        dumimag = dump[2*self.ant1 + 1]
        Tot_D_iR_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_iR_phas = np.angle(dumreal + 1j*dumimag)

        dumreal = dump[2*self.zbl_nant + 2*self.ant2]
        dumimag = dump[2*self.zbl_nant + 2*self.ant2 + 1]
        Tot_D_jL_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_jL_phas = np.angle(dumreal + 1j*dumimag)
        
        dumreal = dump[2*self.zbl_nant + 2*self.ant1]
        dumimag = dump[2*self.zbl_nant + 2*self.ant1 + 1]
        Tot_D_iL_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_iL_phas = np.angle(dumreal + 1j*dumimag)
        
        dumreal = dump[2*self.ant2]
        dumimag = dump[2*self.ant2 + 1]
        Tot_D_jR_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_jR_phas = np.angle(dumreal + 1j*dumimag)
    
    
        if(len(self.zblcalsour) == 1):
            P_ij_amp = [np.absolute(p[self.zbl_nant * 4] + 1j * p[self.zbl_nant * 4 + 1])]
            P_ij_phas = [np.angle(p[self.zbl_nant * 4] + 1j*p[self.zbl_nant * 4 + 1])]
        else:
            P_ij_amp = [np.absolute(it1 + 1j*it2) for it1, it2 in zip([p[self.zbl_nant * 4 + s * 2] for s in range(len(self.zblcalsour))], [p[self.zbl_nant * 4 + s * 2 + 1] for s in range(len(self.zblcalsour))])]
            P_ij_phas = [np.angle(it1 + 1j*it2) for it1, it2 in zip([p[self.zbl_nant * 4 + s * 2] for s in range(len(self.zblcalsour))], [p[self.zbl_nant * 4 + s * 2 + 1] for s in range(len(self.zblcalsour))])]
    
    
        for l in range(len(self.zblcalsour)):
            
            select = (self.sourcearr == self.zblcalsour[l])
            
            Pamp = P_ij_amp[l]
            Pphas = P_ij_phas[l]
            

            pol_RiLj_Real[select] += Pamp * np.cos(Pphas)
            pol_RiLj_Imag[select] += Pamp * np.sin(Pphas)
            pol_LiRj_Real[select] += Pamp * np.cos(-Pphas)
            pol_LiRj_Imag[select] += Pamp * np.sin(-Pphas)
            
            second_RiLj_Real[select] += Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * Pamp * np.cos(Tot_D_iR_phas[select] - Tot_D_jL_phas[select] - Pphas + 2. * (self.pang1[select] + self.pang2[select]))
            second_RiLj_Imag[select] += Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * Pamp * np.sin(Tot_D_iR_phas[select] - Tot_D_jL_phas[select] - Pphas + 2. * (self.pang1[select] + self.pang2[select]))
            second_LiRj_Real[select] += Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * Pamp * np.cos(Tot_D_iL_phas[select] - Tot_D_jR_phas[select] + Pphas - 2. * (self.pang1[select] + self.pang2[select]))
            second_LiRj_Imag[select] += Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * Pamp * np.sin(Tot_D_iL_phas[select] - Tot_D_jR_phas[select] + Pphas - 2. * (self.pang1[select] + self.pang2[select]))     
  
        
        dterm_RiLj_Real += Tot_D_iR_amp * self.llamp * np.cos(Tot_D_iR_phas + self.llphas + 2. * self.pang1) + Tot_D_jL_amp * self.rramp * np.cos(-Tot_D_jL_phas + self.rrphas + 2. * self.pang2)
        dterm_RiLj_Imag += Tot_D_iR_amp * self.llamp * np.sin(Tot_D_iR_phas + self.llphas + 2. * self.pang1) + Tot_D_jL_amp * self.rramp * np.sin(-Tot_D_jL_phas + self.rrphas + 2. * self.pang2)
        dterm_LiRj_Real += Tot_D_iL_amp * self.rramp * np.cos(Tot_D_iL_phas + self.rrphas - 2. * self.pang1) + Tot_D_jR_amp * self.llamp * np.cos(-Tot_D_jR_phas + self.llphas - 2. * self.pang2)
        dterm_LiRj_Imag += Tot_D_iL_amp * self.rramp * np.sin(Tot_D_iL_phas + self.rrphas - 2. * self.pang1) + Tot_D_jR_amp * self.llamp * np.sin(-Tot_D_jR_phas + self.llphas - 2. * self.pang2)
       
        
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
        


    def pol_deq(self, parmset, *p):
        """
        The D-term models for the D-term estimation with instrumental polarization self-calibration.
        
        Args:
            x: dummy parameters (not to be used).
            *p (args): the best-fit parameters args.
        """
        
        (nant, polcalsour, sourcearr, pang1, pang2, ant1, ant2, model_rlreal, model_rlimag, model_lrreal, model_lrimag, rramp, rrphas, llamp, llphas) = parmset
        
        model_rlamp = np.abs(model_rlreal + 1j * model_rlimag)
        model_rlphas = np.angle(model_rlreal + 1j * model_rlimag)
        model_lramp = np.abs(model_lrreal + 1j * model_lrimag)
        model_lrphas = np.angle(model_lrreal + 1j * model_lrimag)
        
        
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
        
        
        for l in range(len(polcalsour)):
        
            select = (sourcearr == polcalsour[l])
            
            RiLj_Real[select] += model_rlreal[select] + \
              Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * model_lramp[select] * np.cos(model_lrphas[select] + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] + 2. * (pang1[select] + pang2[select]))
    
            RiLj_Imag[select] += model_rlimag[select] + \
              Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * model_lramp[select] * np.sin(model_lrphas[select] + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] + 2. * (pang1[select] + pang2[select]))
    
            LiRj_Real[select] += model_lrreal[select] + \
              Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * model_rlamp[select] * np.cos(model_rlphas[select] + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] - 2. * (pang1[select] + pang2[select]))
    
            LiRj_Imag[select] += model_lrimag[select] + \
              Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * model_rlamp[select] * np.sin(model_rlphas[select] + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] - 2. * (pang1[select] + pang2[select]))
            
            
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


    def pol_deq_comp(self, comp, parmset, *p):
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


    def dum_deq(self, x, *p):
        """
        The D-term models for the D-term estimation with instrumental polarization self-calibration.
        
        Args:
            x: dummy parameters (not to be used).
            *p (args): the best-fit parameters args.
        """
        
        RiLj_Real = np.zeros(len(self.pang1))
        RiLj_Imag = np.zeros(len(self.pang1))
        LiRj_Real = np.zeros(len(self.pang1))
        LiRj_Imag = np.zeros(len(self.pang1))
        
        
        dump = np.array(p)
        
        dumreal = dump[2*self.ant1]
        dumimag = dump[2*self.ant1 + 1]
        Tot_D_iR_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_iR_phas = np.angle(dumreal + 1j*dumimag)

        dumreal = dump[2*self.nant + 2*self.ant2]
        dumimag = dump[2*self.nant + 2*self.ant2 + 1]
        Tot_D_jL_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_jL_phas = np.angle(dumreal + 1j*dumimag)
        
        dumreal = dump[2*self.nant + 2*self.ant1]
        dumimag = dump[2*self.nant + 2*self.ant1 + 1]
        Tot_D_iL_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_iL_phas = np.angle(dumreal + 1j*dumimag)
        
        dumreal = dump[2*self.ant2]
        dumimag = dump[2*self.ant2 + 1]
        Tot_D_jR_amp = np.absolute(dumreal + 1j * dumimag)
        Tot_D_jR_phas = np.angle(dumreal + 1j*dumimag)
        
        
        for l in range(len(self.timecalsour)):
        
            select = (self.sourcearr == self.timecalsour[l])
            
            RiLj_Real[select] += self.model_rlreal[select] + \
              Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * self.model_lramp[select] * np.cos(self.model_lrphas[select] + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] + 2. * (self.pang1[select] + self.pang2[select]))
    
            RiLj_Imag[select] += self.model_rlimag[select] + \
              Tot_D_iR_amp[select] * Tot_D_jL_amp[select] * self.model_lramp[select] * np.sin(self.model_lrphas[select] + Tot_D_iR_phas[select] - Tot_D_jL_phas[select] + 2. * (self.pang1[select] + self.pang2[select]))
    
            LiRj_Real[select] += self.model_lrreal[select] + \
              Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * self.model_rlamp[select] * np.cos(self.model_rlphas[select] + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] - 2. * (self.pang1[select] + self.pang2[select]))
    
            LiRj_Imag[select] += self.model_lrimag[select] + \
              Tot_D_iL_amp[select] * Tot_D_jR_amp[select] * self.model_rlamp[select] * np.sin(self.model_rlphas[select] + Tot_D_iL_phas[select] - Tot_D_jR_phas[select] - 2. * (self.pang1[select] + self.pang2[select]))
            
            
        RiLj_Real += \
          Tot_D_iR_amp * self.llamp * np.cos(Tot_D_iR_phas + self.llphas + 2. * self.pang1) + \
          Tot_D_jL_amp * self.rramp * np.cos(-Tot_D_jL_phas + self.rrphas + 2. * self.pang2)
        
        RiLj_Imag += \
          Tot_D_iR_amp * self.llamp * np.sin(Tot_D_iR_phas + self.llphas + 2. * self.pang1) + \
          Tot_D_jL_amp * self.rramp * np.sin(-Tot_D_jL_phas + self.rrphas + 2. * self.pang2)
    
        LiRj_Real += \
          Tot_D_iL_amp * self.rramp * np.cos(Tot_D_iL_phas + self.rrphas - 2. * self.pang1) + \
          Tot_D_jR_amp * self.llamp * np.cos(-Tot_D_jR_phas + self.llphas - 2. * self.pang2)
    
        LiRj_Imag += \
          Tot_D_iL_amp * self.rramp * np.sin(Tot_D_iL_phas + self.rrphas - 2. * self.pang1) + \
          Tot_D_jR_amp * self.llamp * np.sin(-Tot_D_jR_phas + self.llphas - 2. * self.pang2)  
       
    
        compv = np.concatenate([LiRj_Real, LiRj_Imag, RiLj_Real, RiLj_Imag])
        
        return compv
    

    

    
    def residualplot(self, k, nant, antname, source, dumfit, time, ant1, ant2, sourcearr, qamp, qphas, uamp, uphas, qsigma, usigma, tsep, filename):
        """
        Draw fitting residual plots.
        """  
        
        qreal, qimag, ureal, uimag = qamp * np.cos(qphas), qamp * np.sin(qphas), uamp * np.cos(uphas), uamp * np.sin(uphas)
        data_q = qreal + 1j*qimag
        data_u = ureal + 1j*uimag
        
        mod_lr, mod_rl = dumfit[0:len(time)] + 1j*dumfit[len(time):len(time)*2], dumfit[len(time)*2:len(time)*3] + 1j*dumfit[len(time)*3:len(time)*4]
        mod_q, mod_u = (mod_rl + mod_lr) / 2., -1j * (mod_rl - mod_lr) / 2.
        
        
        for m in range(nant):
            
            if(sum(ant1 == m) == 0) & (sum(ant2 == m) == 0): continue
            
            figure, axes = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0}, figsize=(8, 8))
    
            for ax in axes.flat:
                ax.tick_params(length=6, width=2,which = 'major')
                ax.tick_params(length=4, width=1.5,which = 'minor')
                
            axes[0].set_xlim(np.min(time) - (np.max(time) - np.min(time)) * 0.35, np.max(time) + (np.max(time) - np.min(time)) * 0.1)
            axes[1].set_xlim(np.min(time) - (np.max(time) - np.min(time)) * 0.35, np.max(time) + (np.max(time) - np.min(time)) * 0.1)
            
#            axes[0].set(title = 'BL229AE')
            
            axes[1].set(xlabel = 'Time (UT)')
            
            axes[0].set(ylabel = 'Stokes Q (sigma)')
            axes[1].set(ylabel = 'Stokes U (sigma)')
            
            axes[0].set_title(self.dataname)
            
            axes[1].annotate(antname[m], xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25), size = 24, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
            axes[1].annotate('IF {:d}'.format(k+1), xy = (0, 0), xycoords = 'axes fraction', xytext = (25, 25), size = 24, textcoords='offset pixels', horizontalalignment='left', verticalalignment='bottom')
            
            
            for l in range(len(source)):
                
                select = (ant1 == m) | (ant2 == m)
                select = select & (sourcearr == source[l])
                            
                dumx = time[select]
                dumy = np.abs(((data_q[select] - mod_q[select]) / qsigma[select]))
                
                if(len(dumx) <= 1.): continue
            
                argsort = np.argsort(dumx)
                dumx = dumx[argsort]
                dumy = dumy[argsort]
                
                boundary_left = [np.min(dumx)]
                boundary_right = []
                
                for j in range(len(dumx)-1):
                    if(dumx[j+1] - dumx[j]) > tsep:
                        boundary_left.append(dumx[j+1])
                        boundary_right.append(dumx[j])
                
                boundary_right.append(np.max(dumx))
                
                binx = np.zeros(len(boundary_left))
                biny = np.zeros(len(boundary_left))
                binyerr = np.zeros(len(boundary_left))
                
                for j in range(len(boundary_left)):
                    binx[j] = (boundary_left[j] + boundary_right[j]) / 2.
                    biny[j] = np.mean(dumy[(dumx >= boundary_left[j]) & (dumx <= boundary_right[j])])
                    binyerr[j] = np.std(dumy[(dumx >= boundary_left[j]) & (dumx <= boundary_right[j])])
                    
                axes[0].scatter(binx, biny, s = 30, marker = self.markerarr[l], facecolor = 'None', edgecolor = self.colors[l], label = source[l].upper(), zorder = 0)
                
                dumy = np.abs(((data_u[select] - mod_u[select]) / usigma[select]))
    
                dumy = dumy[argsort]
                
                binx = np.zeros(len(boundary_left))
                biny = np.zeros(len(boundary_left))
                binyerr = np.zeros(len(boundary_left))
                
                for j in range(len(boundary_left)):
                    binx[j] = (boundary_left[j] + boundary_right[j]) / 2.
                    biny[j] = np.mean(dumy[(dumx >= boundary_left[j]) & (dumx <= boundary_right[j])])
                    binyerr[j] = np.std(dumy[(dumx >= boundary_left[j]) & (dumx <= boundary_right[j])])
                    
                axes[1].scatter(binx, biny, s = 30, marker = self.markerarr[l], facecolor = 'None', edgecolor = self.colors[l], zorder = 0)
                
                       
            axes[0].legend(loc='upper left', fontsize = 18 - int(len(source)/2.), frameon=False, markerfirst=True, handlelength=1.0)


            dumxticks = axes[1].get_xticks()
            dumxticklabel = []
            for it in dumxticks:
                if(it % 1) == 0.: dumxticklabel.append(str(int(it)))
                if(it % 1) != 0.: dumxticklabel.append(str(it))
            if(np.max(dumxticks) > 24.):
                dumit = 0
                for it in range(len(dumxticks)):
                    if(dumxticks[it] > 24.): 
                        if(dumxticks[it] % 1) == 0.: dumxticklabel[it] = str(int(dumxticks[it] - 24.))
                        if(dumxticks[it] % 1) != 0.: dumxticklabel[it] = str(dumxticks[it] - 24.)
                        if(dumit == 0): dumxticklabel[it] = '{:02d}d/'.format(np.min(self.day) + 1) + dumxticklabel[it]
                        dumit += 1
            
            axes[1].set_xticklabels(dumxticklabel)
    
    
            figure.savefig(filename+'.'+antname[m]+'.'+self.filetype, bbox_inches = 'tight')
            
            plt.close('all')



    def dplot(self, pol, filename, antname, DRArr, DLArr, source, lpcal):
        """
        Draw D-term plots on the complex plane.
        """  
        if not self.dplot_IFsep:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            ax.tick_params(length=6, width=2,which = 'major')
            ax.tick_params(length=4, width=1.5,which = 'minor')
            
            plt.grid()
                
            for m in range(len(antname)):
                
                drreal = DRArr.dropna().applymap(np.real).loc[:,antname[m]].to_numpy() * 1e2
                drimag = DRArr.dropna().applymap(np.imag).loc[:,antname[m]].to_numpy() * 1e2
                
                dlreal = DLArr.dropna().applymap(np.real).loc[:,antname[m]].to_numpy() * 1e2
                dlimag = DLArr.dropna().applymap(np.imag).loc[:,antname[m]].to_numpy() * 1e2
                
                ax.scatter(drreal, drimag, s = 180, facecolor = self.colors[m], edgecolor = self.colors[m], marker = self.markerarr[m], label = antname[m])
                ax.scatter(dlreal, dlimag, s = 180, facecolor = 'None', edgecolor = self.colors[m], marker = self.markerarr[m])
                
            if lpcal:
                for l in range(len(source)):
                    read = pd.read_csv(self.direc+'gpcal/'+self.dataname+source[l]+'.an', header = None, skiprows=1, delimiter = '\t')
                    anname, lpcaldr, lpcaldl = read[1].to_numpy(), read[2].to_numpy(), read[3].to_numpy()
                    dumant, dumdrreal, dumdrimag, dumdlreal, dumdlimag = anname[::2], lpcaldr[::2] * 1e2, lpcaldr[1::2] * 1e2, lpcaldl[::2] * 1e2, lpcaldl[1::2] * 1e2
                    for m in range(len(antname)):
                        ax.scatter(dumdrreal[dumant == antname[m]], dumdrimag[dumant == antname[m]], s = 20, facecolor = self.colors[m], edgecolor = self.colors[m], marker = self.markerarr[m], \
                                   alpha = 0.2)
                        ax.scatter(dumdlreal[dumant == antname[m]], dumdlimag[dumant == antname[m]], s = 20, facecolor = 'None', edgecolor = self.colors[m], marker = self.markerarr[m], \
                                   alpha = 0.2)
                        
    
            if(self.drange == None):
                dumbound = np.max([DRArr.applymap(np.absolute).max().max(), DLArr.applymap(np.absolute).max().max()]) * 1e2 * 1.2
            else:
                dumbound = self.drange
            
            plt.xlim(-dumbound, dumbound)
            plt.ylim(-dumbound, dumbound)
            plt.xlabel('Real (\%)')
            plt.ylabel('Imaginary (\%)')
            plt.title(self.dataname)
            
            
            ax.annotate('Calibrators:', xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
            for i in range(len(source)):
                ax.annotate(source[i], xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25*(i+2)), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
            
            
            if(pol >= 0):
                ax.annotate('Pol-selfcal, Iteration = {:d}/{:d}'.format(pol, self.selfpoliter), xy=(1, 1), xycoords = 'axes fraction', xytext = (-25, -25), size = 22, textcoords='offset pixels', horizontalalignment='right', verticalalignment='top')
    
            
            leg1 = ax.legend(loc='lower left', fontsize = 18, frameon=False, markerfirst=True, handlelength=0.3, labelspacing=0.3)
            
            rcp = ax.scatter(1000, 1000, s = 120, facecolor = 'black', edgecolor = 'black', marker = 'o')
            lcp = ax.scatter(1000, 1000, s = 120, facecolor = 'none', edgecolor = 'black', marker = 'o')
            
            leg2 = ax.legend([rcp, lcp], ['Filled - RCP', 'Open - LCP'], loc = 'lower right', frameon=False, fontsize = 24, handlelength=0.3)
            
            if lpcal:
                gpcal = ax.scatter(1000, 1000, s = 180, facecolor = 'black', edgecolor = 'black', marker = 'o')
                lpcal = ax.scatter(1000, 1000, s = 20, facecolor = 'black', edgecolor = 'black', marker = 'o')
                leg3 = ax.legend([gpcal, lpcal], ['GPCAL', 'LPCAL'], loc = 'upper right', frameon=False, fontsize = 24, handlelength=0.3)
            
            ax.add_artist(leg1)
            
            if lpcal:
                ax.add_artist(leg2)
            
            
            plt.savefig(self.direc + 'gpcal/' + filename +'.'+self.filetype, bbox_inches = 'tight')
            plt.close('all')


        # If dplot_IFsep == True, then plot the D-terms for each IF separately.
        else:

            for k in range(self.ifnum):
                
                if(np.sum(DRArr.loc["IF"+str(k+1)]) == 0.):
                    continue
                fig, ax = plt.subplots(figsize=(8, 8))
                
                ax.tick_params(length=6, width=2,which = 'major')
                ax.tick_params(length=4, width=1.5,which = 'minor')
                
                plt.grid()
        
                
                for m in range(len(antname)):
                    
                    drreal = DRArr.dropna().applymap(np.real).loc["IF"+str(k+1),antname[m]] * 1e2
                    drimag = DRArr.dropna().applymap(np.imag).loc["IF"+str(k+1),antname[m]] * 1e2
                    
                    dlreal = DLArr.dropna().applymap(np.real).loc["IF"+str(k+1),antname[m]] * 1e2
                    dlimag = DLArr.dropna().applymap(np.imag).loc["IF"+str(k+1),antname[m]] * 1e2
                    
                    if(drreal == 0.) & (drimag == 0.) & (dlreal == 0.) & (dlimag == 0.): continue
                    
                    ax.scatter(drreal[drreal != 0.], drimag[drimag != 0.], s = 180, facecolor = self.colors[m], edgecolor = self.colors[m], marker = self.markerarr[m], label = antname[m])
                    ax.scatter(dlreal[drreal != 0.], dlimag[drimag != 0.], s = 180, facecolor = 'None', edgecolor = self.colors[m], marker = self.markerarr[m])
                    
                if lpcal:
                    for l in range(len(source)):
                        read = pd.read_csv(self.direc+'gpcal/'+self.dataname+source[l]+'.an', header = None, skiprows=1, delimiter = '\t')
                        anname, lpcaldr, lpcaldl = read[1].to_numpy(), read[2].to_numpy(), read[3].to_numpy()
                        dumant, dumdrreal, dumdrimag, dumdlreal, dumdlimag = anname[::2], lpcaldr[::2] * 1e2, lpcaldr[1::2] * 1e2, lpcaldl[::2] * 1e2, lpcaldl[1::2] * 1e2
                        for m in range(len(antname)):
                            dumx = dumdrreal[dumant == antname[m]]
                            dumy = dumdrimag[dumant == antname[m]]
                            if(dumx[k] == 0.) & (dumy[k] == 0.): continue
                            ax.scatter(dumx[k], dumy[k], s = 20, facecolor = self.colors[m], edgecolor = self.colors[m], marker = self.markerarr[m], \
                                       alpha = 0.2)
                            dumx = dumdlreal[dumant == antname[m]]
                            dumy = dumdlimag[dumant == antname[m]]
                            ax.scatter(dumx[k], dumy[k], s = 20, facecolor = 'None', edgecolor = self.colors[m], marker = self.markerarr[m], \
                                       alpha = 0.2)
                            
        
                if(self.drange == None):
                    dumbound = np.max([DRArr.applymap(np.absolute).max().max(), DLArr.applymap(np.absolute).max().max()]) * 1e2 * 1.2
                else:
                    dumbound = self.drange
                
                plt.xlim(-dumbound, dumbound)
                plt.ylim(-dumbound, dumbound)
                plt.xlabel('Real (\%)')
                plt.ylabel('Imaginary (\%)')
                plt.title(self.dataname + ', IF' + str(k+1))
                
                
                sourcename = ''
                for i in range(len(source)):
                    sourcename = sourcename + source[i]+'-'
                
                ax.annotate('Calibrators:', xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
                for i in range(len(source)):
                    ax.annotate(source[i], xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25*(i+2)), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
                
                if(pol >= 0):
                    ax.annotate('Pol-selfcal, Iteration = {:d}/{:d}'.format(pol, self.selfpoliter), xy=(1, 1), xycoords = 'axes fraction', xytext = (-25, -25), size = 22, textcoords='offset pixels', horizontalalignment='right', verticalalignment='top')
        
                
                leg1 = ax.legend(loc='lower left', fontsize = 18, frameon=False, markerfirst=True, handlelength=0.3, labelspacing=0.3)
                
                rcp = ax.scatter(1000, 1000, s = 120, facecolor = 'black', edgecolor = 'black', marker = 'o')
                lcp = ax.scatter(1000, 1000, s = 120, facecolor = 'none', edgecolor = 'black', marker = 'o')
                
                leg2 = ax.legend([rcp, lcp], ['Filled - RCP', 'Open - LCP'], loc = 'lower right', frameon=False, fontsize = 24, handlelength=0.3)
                
                if lpcal:
                    gpcal = ax.scatter(1000, 1000, s = 180, facecolor = 'black', edgecolor = 'black', marker = 'o')
                    lpcal = ax.scatter(1000, 1000, s = 20, facecolor = 'black', edgecolor = 'black', marker = 'o')
                    leg3 = ax.legend([gpcal, lpcal], ['GPCAL', 'LPCAL'], loc = 'upper right', frameon=False, fontsize = 24, handlelength=0.3)
                
                ax.add_artist(leg1)
                
                if lpcal:
                    ax.add_artist(leg2)
                
                
                plt.savefig(self.direc + 'gpcal/' + filename + '.IF'+str(k+1) + '.'+self.filetype, bbox_inches = 'tight')
                plt.close('all')



    def dplot_new(self, pol, filename, antname, IFarr, DRArr, DLArr, source = None, lpcal = True, IFsep = False):
        """
        Draw D-term plots on the complex plane.
        """  
        
        uniqant = np.unique(antname)
        
        if not IFsep:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            ax.tick_params(length=6, width=2,which = 'major')
            ax.tick_params(length=4, width=1.5,which = 'minor')
            
            plt.grid()
    
            for m in range(len(uniqant)):
                
                select = (antname == uniqant[m])
                
                drreal = np.real(DRArr[select]) * 1e2
                drimag = np.imag(DRArr[select]) * 1e2
                
                dlreal = np.real(DLArr[select]) * 1e2
                dlimag = np.imag(DLArr[select]) * 1e2
                
                ax.scatter(drreal, drimag, s = 180, facecolor = self.colors[m], edgecolor = self.colors[m], marker = self.markerarr[m], label = antname[m])
                ax.scatter(dlreal, dlimag, s = 180, facecolor = 'None', edgecolor = self.colors[m], marker = self.markerarr[m])
                
            if lpcal:
                for l in range(len(source)):
                    read = pd.read_csv(self.direc+'gpcal/'+self.dataname+source[l]+'.an', header = None, skiprows=1, delimiter = '\t')
                    anname, lpcaldr, lpcaldl = read[1].to_numpy(), read[2].to_numpy(), read[3].to_numpy()
                    dumant, dumdrreal, dumdrimag, dumdlreal, dumdlimag = anname[::2], lpcaldr[::2] * 1e2, lpcaldr[1::2] * 1e2, lpcaldl[::2] * 1e2, lpcaldl[1::2] * 1e2
                    for m in range(len(antname)):
                        ax.scatter(dumdrreal[dumant == antname[m]], dumdrimag[dumant == antname[m]], s = 20, facecolor = self.colors[m], edgecolor = self.colors[m], marker = self.markerarr[m], \
                                   alpha = 0.2)
                        ax.scatter(dumdlreal[dumant == antname[m]], dumdlimag[dumant == antname[m]], s = 20, facecolor = 'None', edgecolor = self.colors[m], marker = self.markerarr[m], \
                                   alpha = 0.2)
                        
    
            if(self.drange == None):
                dumbound = np.max([np.abs(DRArr), np.abs(DLArr)]) * 1e2 * 1.2
            else:
                dumbound = self.drange
                
            
            plt.xlim(-dumbound, dumbound)
            plt.ylim(-dumbound, dumbound)
            plt.xlabel('Real (\%)')
            plt.ylabel('Imaginary (\%)')
            plt.title(self.dataname)
            
            
            ax.annotate('Calibrators:', xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
            for i in range(len(source)):
                ax.annotate(source[i], xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25*(i+2)), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
            
            
            if(pol >= 0):
                ax.annotate('Pol-selfcal, Iteration = {:d}/{:d}'.format(pol, self.selfpoliter), xy=(1, 1), xycoords = 'axes fraction', xytext = (-25, -25), size = 22, textcoords='offset pixels', horizontalalignment='right', verticalalignment='top')
    
            
            leg1 = ax.legend(loc='lower left', fontsize = 18, frameon=False, markerfirst=True, handlelength=0.3, labelspacing=0.3)
            
            rcp = ax.scatter([], [], s = 120, facecolor = 'black', edgecolor = 'black', marker = 'o')
            lcp = ax.scatter([], [], s = 120, facecolor = 'none', edgecolor = 'black', marker = 'o')
            
            leg2 = ax.legend([rcp, lcp], ['Filled - RCP', 'Open - LCP'], loc = 'lower right', frameon=False, fontsize = 24, handlelength=0.3)
            
            if lpcal:
                gpcal = ax.scatter([], [], s = 180, facecolor = 'black', edgecolor = 'black', marker = 'o')
                lpcal = ax.scatter([], [], s = 20, facecolor = 'black', edgecolor = 'black', marker = 'o')
                leg3 = ax.legend([gpcal, lpcal], ['GPCAL', 'LPCAL'], loc = 'upper right', frameon=False, fontsize = 24, handlelength=0.3)
            
            ax.add_artist(leg1)
            
            if lpcal:
                ax.add_artist(leg2)
            
            
            plt.savefig(self.direc + 'gpcal/' + filename +'.'+self.filetype, bbox_inches = 'tight')
            plt.close('all')


        # If dplot_IFsep == True, then plot the D-terms for each IF separately.
        else:

            for k in range(self.ifnum):
                
                if(np.sum((IFarr == k+1)) == 0.):
                    continue
                
                fig, ax = plt.subplots(figsize=(8, 8))
                
                ax.tick_params(length=6, width=2,which = 'major')
                ax.tick_params(length=4, width=1.5,which = 'minor')
                
                plt.grid()
        
                
                for m in range(len(uniqant)):
                    
                    select = (antname == uniqant[m]) & (IFarr == k+1)
                    
                    drreal = np.real(DRArr[select]) * 1e2
                    drimag = np.imag(DRArr[select]) * 1e2
                    
                    dlreal = np.real(DLArr[select]) * 1e2
                    dlimag = np.imag(DLArr[select]) * 1e2
                    
                    
                    if(drreal == 0.) & (drimag == 0.) & (dlreal == 0.) & (dlimag == 0.): continue
                    
                    ax.scatter(drreal[drreal != 0.], drimag[drimag != 0.], s = 180, facecolor = self.colors[m], edgecolor = self.colors[m], marker = self.markerarr[m], label = antname[m])
                    ax.scatter(dlreal[drreal != 0.], dlimag[drimag != 0.], s = 180, facecolor = 'None', edgecolor = self.colors[m], marker = self.markerarr[m])
                    
                if lpcal:
                    for l in range(len(source)):
                        read = pd.read_csv(self.direc+'gpcal/'+self.dataname+source[l]+'.an', header = None, skiprows=1, delimiter = '\t')
                        anname, lpcaldr, lpcaldl = read[1].to_numpy(), read[2].to_numpy(), read[3].to_numpy()
                        dumant, dumdrreal, dumdrimag, dumdlreal, dumdlimag = anname[::2], lpcaldr[::2] * 1e2, lpcaldr[1::2] * 1e2, lpcaldl[::2] * 1e2, lpcaldl[1::2] * 1e2
                        for m in range(len(antname)):
                            dumx = dumdrreal[dumant == antname[m]]
                            dumy = dumdrimag[dumant == antname[m]]
                            if(dumx[k] == 0.) & (dumy[k] == 0.): continue
                            ax.scatter(dumx[k], dumy[k], s = 20, facecolor = self.colors[m], edgecolor = self.colors[m], marker = self.markerarr[m], \
                                       alpha = 0.2)
                            dumx = dumdlreal[dumant == antname[m]]
                            dumy = dumdlimag[dumant == antname[m]]
                            ax.scatter(dumx[k], dumy[k], s = 20, facecolor = 'None', edgecolor = self.colors[m], marker = self.markerarr[m], \
                                       alpha = 0.2)
                            
        
                if(self.drange == None):
                    dumbound = np.max([np.abs(DRArr), np.abs(DLArr)]) * 1e2 * 1.2
                else:
                    dumbound = self.drange
                
                plt.xlim(-dumbound, dumbound)
                plt.ylim(-dumbound, dumbound)
                plt.xlabel('Real (\%)')
                plt.ylabel('Imaginary (\%)')
                plt.title(self.dataname + ', IF' + str(k+1))
                
                
                sourcename = ''
                for i in range(len(source)):
                    sourcename = sourcename + source[i]+'-'
                
                ax.annotate('Calibrators:', xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
                for i in range(len(source)):
                    ax.annotate(source[i], xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25*(i+2)), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
                
                if(pol >= 0):
                    ax.annotate('Pol-selfcal, Iteration = {:d}/{:d}'.format(pol, self.selfpoliter), xy=(1, 1), xycoords = 'axes fraction', xytext = (-25, -25), size = 22, textcoords='offset pixels', horizontalalignment='right', verticalalignment='top')
        
                
                leg1 = ax.legend(loc='lower left', fontsize = 18, frameon=False, markerfirst=True, handlelength=0.3, labelspacing=0.3)
                
                rcp = ax.scatter([], [], s = 120, facecolor = 'black', edgecolor = 'black', marker = 'o')
                lcp = ax.scatter([], [], s = 120, facecolor = 'none', edgecolor = 'black', marker = 'o')
                
                leg2 = ax.legend([rcp, lcp], ['Filled - RCP', 'Open - LCP'], loc = 'lower right', frameon=False, fontsize = 24, handlelength=0.3)
                
                if lpcal:
                    gpcal = ax.scatter([], [], s = 180, facecolor = 'black', edgecolor = 'black', marker = 'o')
                    lpcal = ax.scatter([], [], s = 20, facecolor = 'black', edgecolor = 'black', marker = 'o')
                    leg3 = ax.legend([gpcal, lpcal], ['GPCAL', 'LPCAL'], loc = 'upper right', frameon=False, fontsize = 24, handlelength=0.3)
                
                ax.add_artist(leg1)
                
                if lpcal:
                    ax.add_artist(leg2)
                
                
                plt.savefig(self.direc + 'gpcal/' + filename + '.IF'+str(k+1) + '.'+self.filetype, bbox_inches = 'tight')
                plt.close('all')
                                


    def chisqplot(self, chisq):
        """
        Draw the fitting chi-square plots.
        """  
        fig, ax = plt.subplots(figsize=(9, 7))
        
        ax.tick_params(length=6, width=2,which = 'major')
        ax.tick_params(length=4, width=1.5,which = 'minor')
        
        chisq = copy.deepcopy(self.chisq)
        if self.zblcal:
            chisq = chisq.drop(["zbl"], axis = 0)

        for k in range(self.ifnum):
            ax.scatter(np.arange(0, len(chisq["IF1"])), chisq["IF"+str(k+1)], s = 180, facecolor = 'None', edgecolor = self.colors[k], marker = self.markerarr[k], label = "IF"+str(k+1))
        
        plt.xlim(-1, len(chisq["IF1"])+2)
        plt.xlabel('Iteration')
        plt.ylabel('Reduced chi-square')
        plt.title(self.dataname)
                
        ax.legend(loc='upper right', fontsize = 18, frameon=False, markerfirst=True, handlelength=0.3, labelspacing=0.3)
        
        
        plt.savefig(self.direc + 'gpcal/' + self.outputname+'chisq.'+self.filetype, bbox_inches = 'tight')
        plt.close('all')
        

    def applydterm(self, source, DRArr, DLArr):
        """
        Apply the best-fit D-terms estimated by using the similarity assumption to UVFITS data.
        
        Args:
            source (list): a list of the sources for which D-terms will be corrected.
            DRArr (list): a list of the best-fit RCP D-terms.
            DLArr (list): a list of the best-fit LCP D-terms.
        """
        
        self.logger.info('Applying the estimated D-Terms to {:d} sources...\n'.format(len(self.source)))
        
        dtermname = self.dataname[0:10]        
        
        data = AIPSUVData(dtermname, 'DTERM', 1, 1)
        if(data.exists() == True):
            data.clrstat()
            data.zap()
        
        cmap = AIPSImage(dtermname, 'CMAP', 1, 1)
        if(cmap.exists() == True):
            cmap.clrstat()
            cmap.zap()
            
        au.runfitld(dtermname, 'DTERM', self.direc+self.dataname+source[0]+'.uvf')
        au.runfitld(dtermname, 'CMAP', self.direc+self.dataname+source[0]+'.fits')
    
        au.runlpcal(dtermname, 'DTERM', 'CMAP', 1)
        
        cmap.zap()
        
        
        dterm = WAIPSUVData(dtermname, 'DTERM', 1, 1)
        
        tabed = dterm.table('AN', 1)
        
        
        # Update the antenna tables using the input D-terms.
        i = 0
        for row in tabed:
            
            dumdr = DRArr.loc[:,row.anname.replace(' ', '')]
            dumreal = np.real(np.array([complex(it) for it in dumdr]))
            dumimag = np.imag(np.array([complex(it) for it in dumdr]))
            row.polcala = list(np.ravel([dumreal, dumimag], 'F'))

            dumdl = DLArr.loc[:,row.anname.replace(' ', '')]
            dumreal = np.real(np.array([complex(it) for it in dumdl]))
            dumimag = np.imag(np.array([complex(it) for it in dumdl]))
            row.polcalb = list(np.ravel([dumreal, dumimag], 'F'))
            
            row.update()
            row.update()
            
            i += 1
                
        if path.exists(self.direc+'gpcal/'+self.outputname+'dterm.tbout'):
            os.system('rm ' + self.direc+'gpcal/'+self.outputname+'dterm.tbout')
        au.runtbout(dtermname, 'DTERM', self.direc+'gpcal/'+self.outputname+'dterm.tbout')
        
        # Apply the D-terms and export the D-term corrected UVFITS files to the working directory.
        for l in range(len(source)):
            inname = str(source[l])
            
            if self.selfcal:
                if (not source[l] in self.calsour) | (not source[l] in self.polcalsour):
                    
                    data = AIPSUVData(inname, 'EDIT', 1, 1)
                    if(data.exists() == True):
                        data.clrstat()
                        data.zap()
                        
                    au.runfitld(inname, 'EDIT', self.direc+self.dataname+source[l]+'.uvf')
                    
                            
                    cmap = AIPSImage(inname, 'CMAP', 1, 1)
                    if(cmap.exists() == True):
                        cmap.clrstat()
                        cmap.zap()
                        
                    au.runfitld(inname, 'CMAP', self.direc+self.dataname+source[l]+'.fits')
        
                    calib = AIPSUVData(inname, 'CALIB', 1, 1)
                    if(calib.exists() == True):
                        calib.clrstat()
                        calib.zap()
                            
                    au.runcalib(inname, 'EDIT', inname, 'CMAP', 'CALIB', self.solint, self.soltype, self.solmode, self.weightit)
                    calib = AIPSUVData(inname, 'CALIB', 1, 1)
    
                    if path.exists(self.direc + self.dataname + source[l] + '.calib'):
                        os.system('rm ' + self.direc + self.dataname + source[l] + '.calib')
                    au.runfittp(inname, 'CALIB', self.direc + self.dataname + source[l] + '.calib')
                    
                    data.zap()
                    cmap.zap()
                
                else:
                    calib = AIPSUVData(inname, 'CALIB', 1, 1)
                    if(calib.exists() == True):
                        calib.clrstat()
                        calib.zap()
                    
                    au.runfitld(inname, 'CALIB', self.direc+self.dataname+source[l]+'.calib')
                    
                
                calib.zap_table('AN', 1)
                
                au.runtacop(dtermname, 'DTERM', inname, 'CALIB')
                
                
                split = AIPSUVData(inname, 'SPLIT', 1, 1)
                if(split.exists() == True):
                    split.clrstat()
                    split.zap()
                
                au.runsplit(inname, 'CALIB')
                
                if path.exists(self.direc+self.outputname+source[l]+'.dtcal.uvf'):
                    os.system('rm ' + self.direc+self.outputname+source[l]+'.dtcal.uvf')
                au.runfittp(inname, 'SPLIT', self.direc+self.outputname+source[l]+'.dtcal.uvf')
                
                split = AIPSUVData(inname, 'SPLIT', 1, 1)
                split.zap()
                
                calib.zap()
                
            
            else:
                data = AIPSUVData(inname, 'EDIT', 1, 1)
                if(data.exists() == True):
                    data.clrstat()
                    data.zap()
                    
                au.runfitld(inname, 'EDIT', self.direc+self.dataname+source[l]+'.uvf')
                
                data = AIPSUVData(inname, 'EDIT', 1, 1)
                
                data.zap_table('AN', 1)
                
                au.runtacop(dtermname, 'DTERM', inname, 'EDIT')
                
                
                split = AIPSUVData(inname, 'SPLIT', 1, 1)
                
                if(split.exists() == True):
                    split.clrstat()
                    split.zap()
                
                au.runsplit(inname, 'EDIT')
                
                if path.exists(self.direc+self.outputname+source[l]+'.dtcal.uvf'):
                    os.system('rm ' + self.direc+self.outputname+source[l]+'.dtcal.uvf')
                au.runfittp(inname, 'SPLIT', self.direc+self.outputname+source[l]+'.dtcal.uvf')
                
                split = AIPSUVData(inname, 'SPLIT', 1, 1)
                split.zap()
            
                data.zap()
        
        dterm.zap()


    def pol_applydterm(self, source, read, filename):
        """
        Apply the best-fit D-terms estimated by using instrumental polarization self-calibration to UVFITS data.
        
        Args:
            source (list): a list of the sources for which D-terms will be corrected.
            DRArr (list): a list of the best-fit RCP D-terms.
            DLArr (list): a list of the best-fit LCP D-terms.
            filename (str): the output file name.
        """
        
        self.logger.info('Applying the estimated D-Terms to {:d} sources...\n'.format(len(self.source)))
        
        
        dtermread = pd.read_csv(read, header = 0, skiprows=0, delimiter = '\t', index_col = 0)
            
        pol_ant = np.array(dtermread['antennas'])
        pol_IF = np.array(dtermread['IF'])
        dum_DRArr = np.array(dtermread['DRArr'])
        dum_DLArr = np.array(dtermread['DLArr'])
        
        pol_DRArr = np.array([complex(it) for it in dum_DRArr])
        pol_DLArr = np.array([complex(it) for it in dum_DLArr])
        
    
        inname = 'APPLY'
        
        data = AIPSUVData(inname, 'DTERM', 1, 1)
        if(data.exists() == True):
            data.clrstat()
            data.zap()
        
        cmap = AIPSImage(inname, 'CMAP', 1, 1)
        if(cmap.exists() == True):
            cmap.clrstat()
            cmap.zap()
        
        
        if self.selfcal:
            au.runfitld(inname, 'DTERM', self.direc+self.dataname+source[0]+'.calib')
        else:
            au.runfitld(inname, 'DTERM', self.direc+self.dataname+source[0]+'.uvf')
        
        au.runfitld(inname, 'CMAP', self.direc+self.dataname+source[0]+'.fits')
        
        au.runlpcal(inname, 'DTERM', 'CMAP', 1)
        
        
        image = AIPSImage(inname, 'CMAP', 1, 1)
        image.zap()
        
        
        dterm = WAIPSUVData(inname, 'DTERM', 1, 1)
        
        tabed = dterm.table('AN', 1)
        
        
        fqtable = data.table('FQ', 1)
        if(isinstance(fqtable[0].if_freq, float) == True):
            ifnum = 1
        else:
            ifnum = len(fqtable[0].if_freq)
            
        
        # Update the D-terms in the antenna table.
        for row in tabed:
            anname = row.anname.replace(' ', '')
            
            select = (pol_ant == anname)
            
            dumif = pol_IF[select]
            dum_DRArr = pol_DRArr[select]
            dum_DLArr = pol_DLArr[select]
            
            dumreal, dumimag = [], []
            
            for i in range(ifnum):
                dumselect = (dumif == i+1)
                if(np.sum(dumselect) > 0):
                    dumreal.append(np.real(dum_DRArr[dumselect]))
                    dumimag.append(np.imag(dum_DRArr[dumselect]))
                else:
                    dumreal.append(0.)
                    dumimag.append(0.)
                        
            row.polcala = list(np.ravel([dumreal, dumimag], 'F'))
            
            
            dumreal, dumimag = [], []
            
            for i in range(ifnum):
                dumselect = (dumif == i+1)
                if(np.sum(dumselect) > 0):
                    dumreal.append(np.real(dum_DLArr[dumselect]))
                    dumimag.append(np.imag(dum_DLArr[dumselect]))
                else:
                    dumreal.append(0.)
                    dumimag.append(0.)
                    
            row.polcalb = list(np.ravel([dumreal, dumimag], 'F'))
                        
            row.update()
            row.update()
                
            
            
        if path.exists(filename+'dterm.tbout'):
            os.system('rm ' + filename+'dterm.tbout')
        au.runtbout(inname, 'DTERM', filename+'dterm.tbout')
            
            
        for l in range(len(source)):
            inname = str(source[l])
            
            calib = AIPSUVData(inname, 'CALIB', 1, 1)
            if(calib.exists() == True):
                calib.clrstat()
                calib.zap()
            
            if self.selfcal:
                au.runfitld(inname, 'CALIB', self.direc + self.dataname + source[l]+'.calib')
            else:
                au.runfitld(inname, 'CALIB', self.direc + self.dataname + source[l]+'.uvf')
        
            calib = AIPSUVData(inname, 'CALIB', 1, 1)
        
            calib.zap_table('AN', 1)
            
            au.runtacop('APPLY', 'DTERM', inname, 'CALIB')
        
        
            split = AIPSUVData(inname, 'SPLIT', 1, 1)
            if(split.exists() == True):
                split.clrstat()
                split.zap()
                    
            au.runsplit(inname, 'CALIB')
            
        
            if path.exists(filename+source[l]+'.dtcal.uvf'):
                os.system('rm ' + filename+source[l]+'.dtcal.uvf')
            au.runfittp(inname, 'SPLIT', filename+source[l]+'.dtcal.uvf')
            
            split = AIPSUVData(inname, 'SPLIT', 1, 1)
            split.zap()
            
        
            calib.zap()
            
        
        dterm.zap()
        


    def evpacal(self, datain, dataout, clcorprm):
        
        self.logger.info('Correcting EVPAs... \n Input file: {:} \n Output file: {:}'.format(datain, dataout))
        
        pinal = AIPSUVData('EVPA', 'PINAL', 1, 1)
        if(pinal.exists() == True):
            pinal.zap()
            
        au.runfitld('EVPA', 'PINAL', datain)
        
        pinal = AIPSUVData('EVPA', 'PINAL', 1, 1)
        
        aipssource = pinal.header.object
        
        multi = AIPSUVData('EVPA', 'MULTI', 1, 1)
        if(multi.exists() == True):
            multi.zap()
            
        au.runmulti('EVPA', 'PINAL', 1, 1)
        au.runclcor('EVPA', 'MULTI', 1, 1, clcorprm)
        
        pang = AIPSUVData(aipssource, 'PANG', 1, 1)
        if(pang.exists() == True):
            pang.zap()
            
        au.runsplitpang('EVPA')
                
        au.runfittp(aipssource, 'PANG', dataout)
        
        pinal.zap()
        multi.zap()
        pang.zap()
    

        
    def zbl_dtermsolve(self):
        """
        Estimate the D-terms using the zero baselines.
        """        
                
        self.logger.info('\n####################################################################')
        self.logger.info('Zero-baseline D-Term estimation mode...\n')

        # Get the zero baseline data.
        self.get_zbl_data()
        
        # Print the basic information of the data.
        zblblname = ''
        for i in range(len(self.zblant)):
            zblblname = zblblname+('{:s}-{:s}'.format(self.zblant[i][0], self.zblant[i][1]))+','
        zblblname = zblblname[:-1]
        self.logger.info('baselines to be used: ' + zblblname + '\n')
        zblsourcename = ''
        for i in range(len(self.zblcalsour)):
            zblsourcename = zblsourcename+('{:s}'.format(self.zblcalsour[i]))+','
        zblsourcename = zblsourcename[:-1]
        self.logger.info('{:d} data from {:s} source(s) will be used.'.format(len(self.zbl_data["time"]), str(len(self.zblcalsour))) + '\n')
        self.logger.info('Source coordinates:')
        for i in range(len(self.zblcalsour)):
            self.logger.info('{:s}: RA = {:5.2f} deg, Dec = {:5.2f} deg'.format(self.zblcalsour[i], self.zbl_obsra[i], self.zbl_obsdec[i]))
        self.logger.info('\nAntenna information:')
        for i in range(len(self.zblant)):
            for j in range(len(self.antname)):
                if(self.zblant[i][0] == self.antname[j]):
                    if(self.antmount[j] == 0): mount = 'Cassegrain'
                    if(self.antmount[j] == 4): mount = 'Nasmyth-Right'
                    if(self.antmount[j] == 5): mount = 'Nasmyth-Left'
                    self.logger.info('{:s}: antenna mount = {:13s}, X = {:11.2f}m, Y = {:11.2f}m, Z = {:11.2f}m'.format(self.zblant[i][0], mount, self.antx[j], self.anty[j], self.antz[j]))
                if(self.zblant[i][1] == self.antname[j]):
                    if(self.antmount[j] == 0): mount = 'Cassegrain'
                    if(self.antmount[j] == 4): mount = 'Nasmyth-Right'
                    if(self.antmount[j] == 5): mount = 'Nasmyth-Left'
                    self.logger.info('{:s}: antenna mount = {:13s}, X = {:11.2f}m, Y = {:11.2f}m, Z = {:11.2f}m'.format(self.zblant[i][1], mount, self.antx[j], self.anty[j], self.antz[j]))
        
        self.logger.info(' ')
        
        self.logger.info('Observation date = {:s}'.format(str(np.min(self.year))+'-'+str(np.min(self.month))+'-'+str(np.min(self.day))))
        
        
        self.zbl_nant = len(self.zblant)*2
        
        
        zbl_antname = []
        for i in range(len(self.zblant)):
            zbl_antname.append(self.zblant[i][0])
            zbl_antname.append(self.zblant[i][1])
        
        
        # Create pandas dataframes where the best-fit D-terms will be stored.
        self.zbl_DRArr = pd.DataFrame(index = ['IF'+str(it+1) for it in np.arange(self.ifnum)], columns = zbl_antname)
        self.zbl_DLArr = pd.DataFrame(index = ['IF'+str(it+1) for it in np.arange(self.ifnum)], columns = zbl_antname)
        
        self.zbl_DRErr = pd.DataFrame(index = ['IF'+str(it+1) for it in np.arange(self.ifnum)], columns = zbl_antname)
        self.zbl_DLErr = pd.DataFrame(index = ['IF'+str(it+1) for it in np.arange(self.ifnum)], columns = zbl_antname)
        
        
        # Create a pandas dataframe where the best-fit source-polarization terms will be stored.
        sourcepolcolumns = []
        
        for l in range(len(self.zblcalsour)):
            sourcepolcolumns.append(self.zblcalsour[l])
        
        self.zbl_sourcepol = pd.DataFrame(index = ['IF'+str(it+1) for it in np.arange(self.ifnum)], columns = sourcepolcolumns)
        
        
        for k in range(self.ifnum):
            
            self.logger.info('\n####################################################################')
            self.logger.info('Processing IF {:d}'.format(k+1) + '...')
            self.logger.info('####################################################################\n')
            
            
            ifdata = self.zbl_data.loc[self.zbl_data["IF"] == k+1]
                 
            time, dayarr, sourcearr, pang1, pang2, ant1, ant2, rrreal, rrimag, llreal, llimag, rlreal, rlimag, lrreal, lrimag, rlsigma, lrsigma, rramp, rrphas, llamp, llphas, \
            qamp, qphas, uamp, uphas, qamp_sigma, qphas_sigma, uamp_sigma, uphas_sigma, qsigma, usigma = \
                np.array(ifdata["time"]), np.array(ifdata["day"]), np.array(ifdata["source"]), np.array(ifdata["pang1"]), np.array(ifdata["pang2"]), np.array(ifdata["ant1"]), np.array(ifdata["ant2"]), \
                np.array(ifdata["rrreal"]), np.array(ifdata["rrimag"]), np.array(ifdata["llreal"]), np.array(ifdata["llimag"]), \
                np.array(ifdata["rlreal"]), np.array(ifdata["rlimag"]), np.array(ifdata["lrreal"]), np.array(ifdata["lrimag"]), \
                np.array(ifdata["rlsigma"]), np.array(ifdata["lrsigma"]), np.array(ifdata["rramp"]), np.array(ifdata["rrphas"]), np.array(ifdata["llamp"]), np.array(ifdata["llphas"]), \
                np.array(ifdata["qamp"]), np.array(ifdata["qphas"]), np.array(ifdata["uamp"]), np.array(ifdata["uphas"]), \
                np.array(ifdata["qamp_sigma"]), np.array(ifdata["qphas_sigma"]), np.array(ifdata["uamp_sigma"]), np.array(ifdata["uphas_sigma"]), np.array(ifdata["qsigma"]), np.array(ifdata["usigma"])

            
            if(len(time) == 0.):
                self.logger.info('Will skip this IF because there is no data.\n')
                continue
            
            
            # Draw the field-rotation angle plots if requested.
            if self.parplot: 
                self.logger.info('Creating field-rotation-angle plots...\n')
                self.parangplot(k, self.zbl_nant, zbl_antname, self.zblcalsour, time, ant1, ant2, sourcearr, pang1, pang2, self.direc+'gpcal/'+self.outputname+'zbl.FRA.IF'+str(k+1))
            
            
            
            inputx = np.concatenate([pang1, pang2, pang1, pang2])
            inputy = np.concatenate([lrreal, lrimag, rlreal, rlimag])
            
            inputsigma = np.concatenate([lrsigma, lrsigma, rlsigma, rlsigma])
            
            
            # The boundaries of parameters allowed for the least-square fitting.
            lbound = [-self.Dbound]*(2*2*self.zbl_nant) + [-self.Pbound / np.sqrt(2.)]*(len(self.zblcalsour)*2)
            ubound = [self.Dbound]*(2*2*self.zbl_nant) + [self.Pbound / np.sqrt(2.)]*(len(self.zblcalsour)*2)
            
            
            if(k == 0): 
                init = np.zeros(2*2*self.zbl_nant + len(self.zblcalsour)*2)
            else:
                if('Iteration' in locals()):
                    init = Iteration
                    init[2*2*self.zbl_nant:] = 0.
                else:
                    init = np.zeros(2*2*self.zbl_nant + len(self.zblcalsour)*2)
                    
            
            bounds=(lbound,ubound)
            
            
            # Define global variables to be transferred into the fitting functions.
            self.pang1, self.pang2, self.ant1, self.ant2, self.sourcearr, self.rramp, self.rrphas, self.llamp, self.llphas = \
                pang1, pang2, ant1, ant2, sourcearr, rramp, rrphas, llamp, llphas
            
            
            # Perform the least-square fitting using Scipy curve_fit.
            Iteration, pco = curve_fit(self.zbl_deq, inputx, inputy, p0=init, sigma = inputsigma, absolute_sigma = False, bounds = bounds)
            error = np.sqrt(np.diag(pco))
            
            
            # Save the best-fit D-terms in the pandas dataframes.
            for m in range(self.zbl_nant-1):
                for n in np.arange(m+1, self.zbl_nant):
                    self.zbl_DRArr.loc["IF"+str(k+1), zbl_antname[m]] = Iteration[2*m] + 1j*Iteration[2*m+1]
                    self.zbl_DRArr.loc["IF"+str(k+1), zbl_antname[n]] = Iteration[2*n] + 1j*Iteration[2*n+1]
                    self.zbl_DLArr.loc["IF"+str(k+1), zbl_antname[m]] = Iteration[2*self.zbl_nant+2*m] + 1j*Iteration[2*self.zbl_nant+2*m+1]
                    self.zbl_DLArr.loc["IF"+str(k+1), zbl_antname[n]] = Iteration[2*self.zbl_nant+2*n] + 1j*Iteration[2*self.zbl_nant+2*n+1]

                    self.zbl_DRErr.loc["IF"+str(k+1), zbl_antname[m]] = error[2*m] + 1j*error[2*m+1]
                    self.zbl_DRErr.loc["IF"+str(k+1), zbl_antname[n]] = error[2*n] + 1j*error[2*n+1]
                    self.zbl_DLErr.loc["IF"+str(k+1), zbl_antname[m]] = error[2*self.zbl_nant+2*m] + 1j*error[2*self.zbl_nant+2*m+1]
                    self.zbl_DLErr.loc["IF"+str(k+1), zbl_antname[n]] = error[2*self.zbl_nant+2*n] + 1j*error[2*self.zbl_nant+2*n+1]

    
            # Save the best-fit source-polarization terms in the pandas dataframe.
            for l in range(len(self.zblcalsour)):
                
                self.zbl_sourcepol.loc["IF"+str(k+1), self.zblcalsour[l]] = Iteration[self.zbl_nant * 4 + 2*l] + 1j*Iteration[self.zbl_nant * 4 + 2*l + 1]
                
            
            self.zbl_DRamp = pd.concat([self.zbl_DRArr.applymap(np.absolute)])
            self.zbl_DRphas = pd.concat([self.zbl_DRArr.applymap(np.angle).applymap(np.degrees)])
            self.zbl_DLamp = pd.concat([self.zbl_DLArr.applymap(np.absolute)])
            self.zbl_DLphas = pd.concat([self.zbl_DLArr.applymap(np.angle).applymap(np.degrees)])
            self.zbl_sourcepolamp = pd.concat([self.zbl_sourcepol.applymap(np.absolute)])
            self.zbl_sourcepolphas = pd.concat([self.zbl_sourcepol.applymap(np.angle).applymap(np.degrees)])
                            
 
            # Calculate the reduced chi-square of the fitting.
            dumfit = self.zbl_deq(inputx, *Iteration)
            
            ydata = np.concatenate([lrreal+1j*lrimag, rlreal+1j*rlimag])
            yfit = np.concatenate([dumfit[0:len(lrreal)]+1j*dumfit[len(lrreal):len(lrreal)*2], dumfit[len(lrreal)*2:len(lrreal)*3]+1j*dumfit[len(lrreal)*3:len(lrreal)*4]])
            ysigma = np.concatenate([lrsigma, rlsigma])
            
            chisq = np.sum(np.abs(((ydata - yfit) / ysigma) ** 2)) / (2. * len(ydata))
            
            self.zbl_chisq = chisq
            
            self.logger.info('\nThe reduced chi-square of the fitting is {:5.3f}.'.format(chisq))
        
        
            self.logger.info(' ')
            for m in range(self.zbl_nant):
                self.logger.info('{:s}: RCP - amplitude = {:6.3f} %, phase = {:7.2f} deg, LCP - amplitude = {:6.3f} %, phase = {:7.2f} deg'.format(zbl_antname[m], \
                      self.zbl_DRamp.loc["IF"+str(k+1), zbl_antname[m]] * 1e2, self.zbl_DRphas.loc["IF"+str(k+1), zbl_antname[m]], 
                      self.zbl_DLamp.loc["IF"+str(k+1), zbl_antname[m]] * 1e2, self.zbl_DLphas.loc["IF"+str(k+1), zbl_antname[m]]))
            
            
            self.logger.info(' ')
            for l in range(len(self.zblcalsour)):
            
                dumr = Iteration[self.zbl_nant * 4 + 2*l]
                dumi = Iteration[self.zbl_nant * 4 + 2*l + 1]
                
                self.logger.info(r'{:s}: Polarized Intensity = {:6.3f} Jy, EVPA = {:6.2f} deg'.format(self.zblcalsour[l], np.sqrt(dumr ** 2 + dumi ** 2), np.degrees(np.angle(dumr + 1j * dumi)) / 2.))
            
            
            mod_lr, mod_rl = dumfit[0:len(time)] + 1j*dumfit[len(time):len(time)*2], dumfit[len(time)*2:len(time)*3] + 1j*dumfit[len(time)*3:len(time)*4]
            mod_q, mod_u = (mod_rl + mod_lr) / 2., -1j * (mod_rl - mod_lr) / 2.
            mod_qamp, mod_qphas, mod_uamp, mod_uphas = np.absolute(mod_q), np.angle(mod_q), np.absolute(mod_u), np.angle(mod_u)
            
            if self.allplot:
                mod_pol_qamp, mod_pol_qphas, mod_pol_uamp, mod_pol_uphas = self.zbl_deq_comp('pol', *Iteration)
                mod_dterm_qamp, mod_dterm_qphas, mod_dterm_uamp, mod_dterm_uphas = self.zbl_deq_comp('dterm', *Iteration)
                mod_second_qamp, mod_second_qphas, mod_second_uamp, mod_second_uphas = self.zbl_deq_comp('second', *Iteration)
                allplots = (mod_pol_qamp, mod_pol_qphas, mod_pol_uamp, mod_pol_uphas, mod_dterm_qamp, mod_dterm_qphas, mod_dterm_uamp, mod_dterm_uphas)
            else:
                allplots = None
            
            
            if (self.vplot_title == None):
                self.vplot_title = self.dataname[:-1]
                
            # Create vplots if requested.
            if self.vplot:
                self.logger.info('\nCreating vplots for all baselines... It may take some time.')
                
                parmset = []

                for l in range(len(self.zblcalsour)):
                    for m in range(self.zbl_nant):
                        for n in range(self.zbl_nant):
                            if(m==n):
                                continue
                            dumm = m
                            dumn = n
                            if(m>n):
                                dumm = n
                                dumn = m
                                
                                select = (ant1 == dumm) & (ant2 == dumn) & (sourcearr == self.zblcalsour[l])
                            
                                selected_time, selected_day, selected_qamp, selected_qphas, selected_qsigma, selected_uamp, selected_uphas, selected_usigma, selected_mod_qamp, selected_mod_qphas, selected_mod_uamp, selected_mod_uphas = \
                                    time[select], dayarr[select], qamp[select], qphas[select], qsigma[select], uamp[select], uphas[select], usigma[select], \
                                    mod_qamp[select], mod_qphas[select], mod_uamp[select], mod_uphas[select]
                                    
                                if self.allplot:
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
                                
                                    
                                if self.multiproc:
                                    parmset.append((self.colors[l], self.zblcalsour[l], k+1, zbl_antname[dumm], zbl_antname[dumn], \
                                                   selected_day, selected_time, selected_qamp, selected_qphas, selected_qsigma, selected_uamp, selected_uphas, selected_usigma, \
                                                   selected_mod_qamp, selected_mod_qphas, selected_mod_uamp, selected_mod_uphas, self.direc+'gpcal/'+self.outputname+'zbl.vplot.IF'+str(k+1), self.filetype, \
                                                   allplots, self.vplot_title, self.vplot_scanavg, self.vplot_avg_nat, self.tsep))
                                else:
                                    ph.visualplot(self.colors[l], self.zblcalsour[l], k+1, zbl_antname[dumm], zbl_antname[dumn], \
                                                   selected_day, selected_time, selected_qamp, selected_qphas, selected_qsigma, selected_uamp, selected_uphas, selected_usigma, \
                                                   selected_mod_qamp, selected_mod_qphas, selected_mod_uamp, selected_mod_uphas, self.direc+'gpcal/'+self.outputname+'zbl.vplot.IF'+str(k+1), self.filetype, \
                                                   allplots = allplots, title = self.dataname, scanavg = self.vplot_scanavg, avg_nat = self.vplot_avg_nat, tsep = self.tsep)


                if self.multiproc:
                    ph.visualplot_run(parmset, nproc = self.nproc)
                    
                self.logger.info('...completed.\n')
            
            
            # Create fitting residual plots if requested.
            if self.resplot:
                self.logger.info('Creating fitting residual plots for all stations... It may take some time.')
                self.residualplot(k, self.zbl_nant, zbl_antname, self.zblcalsour, dumfit, time, ant1, ant2, sourcearr, qamp, qphas, uamp, uphas, qsigma, usigma, self.tsep, \
                                  self.direc+'gpcal/'+self.outputname+'zbl.res.IF'+str(k+1)) 
                   
                self.logger.info('...completed.\n')
            
            
            del self.pang1, self.pang2, self.ant1, self.ant2, self.sourcearr, self.rramp, self.rrphas, self.llamp, self.llphas 
            
        
        # Create D-term plots.
        self.dplot(-1, self.outputname+'zbl.D_Terms', zbl_antname, self.zbl_DRArr, self.zbl_DLArr, self.zblcalsour, False)
        
        self.zbl_sourcepol.to_csv(self.direc+'gpcal/'+self.outputname+'sourcepol.txt', sep = "\t")
        
        
        f = open(self.direc+'GPCAL_Difmap_v1','w')
        
        f.write('observe %1\nmapcolor rainbow, 1, 0.5\nselect %13, %2, %3\nmapsize %4, %5\nuvweight %6, %7\nrwin %8\nshift %9,%10\ndo i=1,100\nclean 100, 0.02, imstat(rms)*%11\nend do\nselect i\nsave %12.%13\nexit')
            
        f.close()
        
    
    def dtermsolve(self):    
        """
        A main function.
        """        
        
        if self.zblcal: self.zbl_dtermsolve()
        
        self.get_data()
        
        self.simil_dtermsolve()
                
        if self.selfpol: self.pol_dtermsolve()
        
        
        
    def simil_dtermsolve(self):
        """
        Estimate the D-terms using the similarity assumption.
        """  
        
        # Print the basic information of the data.
        self.logger.info('\n####################################################################')
        self.logger.info('Estimating the D-Terms by using the similarity assumption...\n')
        sourcename = ''
        for i in range(len(self.calsour)):
            sourcename = sourcename+('{:s}'.format(self.calsour[i]))+','
        sourcename = sourcename[:-1]
        self.logger.info('{:d} data from {:s} source(s) will be used:'.format(len(self.data["time"]), str(len(self.calsour))) + '\n')
        self.logger.info('Source coordinates:')
        for i in range(len(self.calsour)):
            self.logger.info('{:s}: RA = {:5.2f} deg, Dec = {:5.2f} deg'.format(self.calsour[i], self.obsra[i], self.obsdec[i]))
        self.logger.info('\nAntenna information:')
        for i in range(self.nant):
            if(self.antmount[i] == 0): mount = 'Cassegrain'
            if(self.antmount[i] == 4): mount = 'Nasmyth-Right'
            if(self.antmount[i] == 5): mount = 'Nasmyth-Left'
            self.logger.info('{:s}: antenna mount = {:13s}, X = {:11.2f}m, Y = {:11.2f}m, Z = {:11.2f}m'.format(self.antname[i], mount, self.antx[i], self.anty[i], self.antz[i]))
                
        
        self.logger.info(' ')
        
        self.logger.info('Observation date = {:s}'.format(str(np.min(self.year))+'-'+str(np.min(self.month))+'-'+str(np.min(self.day))))
        
        
        # Create pandas dataframes where the best-fit D-terms will be stored.
        self.DRArr = pd.DataFrame(index = ['IF'+str(it+1) for it in np.arange(self.ifnum)], columns = self.antname)
        self.DLArr = pd.DataFrame(index = ['IF'+str(it+1) for it in np.arange(self.ifnum)], columns = self.antname)
        
        self.DRErr = pd.DataFrame(index = ['IF'+str(it+1) for it in np.arange(self.ifnum)], columns = self.antname)
        self.DLErr = pd.DataFrame(index = ['IF'+str(it+1) for it in np.arange(self.ifnum)], columns = self.antname)
        
        
        simil_IF, simil_ant, simil_DRArr, simil_DLArr = [], [], [], []
        
        # Create a pandas dataframe where the best-fit source-polarization terms will be stored.
        sourcepolcolumns = []
        
        for l in range(len(self.calsour)):
            if(self.cnum[l] == 0):
                sourcepolcolumns.append(self.calsour[l])
            else:
                for i in range(self.cnum[l]):
                    sourcepolcolumns.append(self.calsour[l] + ', R' + str(i+1))
        
        self.sourcepol = pd.DataFrame(index = ['IF'+str(it+1) for it in np.arange(self.ifnum)], columns = sourcepolcolumns)
        
        
        
        for k in range(self.ifnum):
            
            self.logger.info('\n####################################################################')
            self.logger.info('Processing IF {:d}'.format(k+1) + '...')
            self.logger.info('####################################################################\n')
            
            ifdata = self.data.loc[self.data["IF"] == k+1]
            
            if(np.sum(self.data["IF"] == k+1) == 0.):
                self.logger.info('Will skip this IF because there is no data.\n')
                continue
                  
            time, dayarr, sourcearr, pang1, pang2, ant1, ant2, rrreal, rrimag, llreal, llimag, rlreal, rlimag, lrreal, lrimag, rrsigma, llsigma, rlsigma, lrsigma, rramp, rrphas, llamp, llphas, \
            qamp, qphas, uamp, uphas, qamp_sigma, qphas_sigma, uamp_sigma, uphas_sigma, qsigma, usigma = \
                np.array(ifdata["time"]), np.array(ifdata["day"]), np.array(ifdata["source"]), np.array(ifdata["pang1"]), np.array(ifdata["pang2"]), np.array(ifdata["ant1"]), np.array(ifdata["ant2"]), \
                np.array(ifdata["rrreal"]), np.array(ifdata["rrimag"]), np.array(ifdata["llreal"]), np.array(ifdata["llimag"]), \
                np.array(ifdata["rlreal"]), np.array(ifdata["rlimag"]), np.array(ifdata["lrreal"]), np.array(ifdata["lrimag"]), \
                np.array(ifdata["rrsigma"]), np.array(ifdata["llsigma"]), np.array(ifdata["rlsigma"]), np.array(ifdata["lrsigma"]), \
                np.array(ifdata["rramp"]), np.array(ifdata["rrphas"]), np.array(ifdata["llamp"]), np.array(ifdata["llphas"]), \
                np.array(ifdata["qamp"]), np.array(ifdata["qphas"]), np.array(ifdata["uamp"]), np.array(ifdata["uphas"]), \
                np.array(ifdata["qamp_sigma"]), np.array(ifdata["qphas_sigma"]), np.array(ifdata["uamp_sigma"]), np.array(ifdata["uphas_sigma"]), np.array(ifdata["qsigma"]), np.array(ifdata["usigma"])
                
            
            # Draw field-rotation angle plots if requested.
            if self.parplot: 
                self.logger.info('Creating field-rotation-angle plots...\n')
                self.parangplot(k, self.nant, self.antname, self.calsour, time, ant1, ant2, sourcearr, pang1, pang2, self.direc+'gpcal/'+self.outputname+'FRA.IF'+str(k+1))
            
            
            inputx = np.concatenate([pang1, pang2, pang1, pang2])
            inputy = np.concatenate([lrreal,lrimag,rlreal,rlimag])
            
            inputsigma = np.concatenate([lrsigma, lrsigma, rlsigma, rlsigma])
            
            
            # Rescale the visibility weights of specific stations if requested.
            if self.manualweight:
                
                sigmaant1 = np.concatenate([ant1, ant1, ant1, ant1])
                sigmaant2 = np.concatenate([ant2, ant2, ant2, ant2])
                
                outputweight = 1. / inputsigma ** 2
                
                for m in range(self.nant):
                    if(self.antname[m] in self.weightfactors):
                        outputweight[(sigmaant1 == m) | (sigmaant2 == m)] = outputweight[(sigmaant1 == m) | (sigmaant2 == m)] * self.weightfactors.get(self.antname[m])
                        self.logger.info('The visibility weights for {:s} station are rescaled by a factor of {:4.2f}.'.format(self.antname[m], self.weightfactors.get(self.antname[m])))
                
                self.logger.info(' ')
                
                outputsigma = 1. / outputweight ** (1. / 2.)
                
            else:
                outputsigma = np.copy(inputsigma)
                self.logger.info('No visibility weight rescaling applied.\n')
            
            
            dumantname = np.copy(self.antname)
            orig_ant1 = np.copy(ant1)
            orig_ant2 = np.copy(ant2)
            
            
            orig_nant = self.nant
            
            
            #If there are antennas having no data, then rearrange the antenna numbers for fitting. This change will be reverted after the fitting.
            dum = 0
            
            removed_Index = []
            
            for m in range(orig_nant):
                if((sum(ant1 == dum) == 0) & (sum(ant2 == dum) == 0)):
                    ant1[ant1 > dum] -= 1
                    ant2[ant2 > dum] -= 1
                    self.nant -= 1
                    removed_Index.append(m)
                    self.logger.info('{:s} has no data, the fitting will not solve the D-Terms for it.'.format(self.antname[m]))
                else:
                    dum += 1
            
            
            if(len(removed_Index) != 0): dumantname = np.delete(dumantname, removed_Index)
            
            
            if(k == 0): 
                init = np.zeros(2*2*self.nant + sum(self.cnum)*2)
            else:
                if('Iteration' in locals()):
                    init = Iteration
                    if(len(removed_Index) != 0.):
                        dum = []
                        for it in removed_Index:
                            dum.append(2*it)
                            dum.append(2*it+1)
                            dum.append(2*orig_nant + 2*it)
                            dum.append(2*orig_nant + 2*it+1)
                        init = np.delete(init, dum)
                else:
                    init = np.zeros(2*2*self.nant + sum(self.cnum)*2)
            
            # The boundaries of parameters allowed for the least-square fitting.
            lbound = [-self.Dbound]*(2*2*self.nant) + [-self.Pbound / np.sqrt(2.)]*(sum(self.cnum)*2)
            ubound = [self.Dbound]*(2*2*self.nant) + [self.Pbound / np.sqrt(2.)]*(sum(self.cnum)*2)
            
            if not self.fixdterm:
                self.fixdr = None
                self.fixdl = None
            
            # If the zero-baseline D-term estimation was performed, then fix the D-terms of those stations for fitting for the rest of the array.
            if self.zblcal:
                if(self.fixdr == None):
                    self.fixdr = {}
                    self.fixdl = {}
                for i in range(len(self.zblant)):
                    if(self.zblant[i][0] in self.fixdr):
                        self.fixdr[self.zblant[i][0]] = self.zbl_DRArr.loc["IF"+str(k+1), self.zblant[i][0]]
                        self.fixdl[self.zblant[i][0]] = self.zbl_DLArr.loc["IF"+str(k+1), self.zblant[i][0]]
                    else:
                        self.fixdr.update({self.zblant[i][0]: self.zbl_DRArr.loc["IF"+str(k+1), self.zblant[i][0]]})
                        self.fixdl.update({self.zblant[i][0]: self.zbl_DLArr.loc["IF"+str(k+1), self.zblant[i][0]]})
                    
                    if(self.zblant[i][1] in self.fixdr):
                        self.fixdr[self.zblant[i][1]] = self.zbl_DRArr.loc["IF"+str(k+1), self.zblant[i][1]]
                        self.fixdl[self.zblant[i][1]] = self.zbl_DLArr.loc["IF"+str(k+1), self.zblant[i][1]]
                    else:
                        self.fixdr.update({self.zblant[i][1]: self.zbl_DRArr.loc["IF"+str(k+1), self.zblant[i][1]]})
                        self.fixdl.update({self.zblant[i][1]: self.zbl_DLArr.loc["IF"+str(k+1), self.zblant[i][1]]})


            # Fix the D-terms of specific stations to be certain values for fitting if requested.
            if (self.fixdterm == True) | (self.zblcal == True):
                
                for i in range(self.nant):
                    if dumantname[i] in self.fixdr:
                        lbound[2*i] = np.real(self.fixdr.get(dumantname[i])) - 1e-8
                        ubound[2*i] = np.real(self.fixdr.get(dumantname[i])) + 1e-8
                        lbound[2*i+1] = np.imag(self.fixdr.get(dumantname[i])) - 1e-8
                        ubound[2*i+1] = np.imag(self.fixdr.get(dumantname[i])) + 1e-8
                        lbound[2*self.nant+2*i] = np.real(self.fixdl.get(dumantname[i])) - 1e-8
                        ubound[2*self.nant+2*i] = np.real(self.fixdl.get(dumantname[i])) + 1e-8
                        lbound[2*self.nant+2*i+1] = np.imag(self.fixdl.get(dumantname[i])) - 1e-8
                        ubound[2*self.nant+2*i+1] = np.imag(self.fixdl.get(dumantname[i])) + 1e-8
                        init[2*i] = np.real(self.fixdr.get(dumantname[i]))
                        init[2*i+1] = np.imag(self.fixdr.get(dumantname[i]))
                        init[2*self.nant+2*i] = np.real(self.fixdl.get(dumantname[i]))
                        init[2*self.nant+2*i+1] = np.imag(self.fixdl.get(dumantname[i]))
            
            
            bounds=(lbound,ubound)
            
            # Define global variables to be transferred into the fitting functions.
            self.pang1, self.pang2, self.ant1, self.ant2, self.sourcearr, self.rramp, self.rrphas, self.llamp, self.llphas = \
                pang1, pang2, ant1, ant2, sourcearr, rramp, rrphas, llamp, llphas
            
            
            self.modamp = []
            self.modphas = []
            for l in range(len(self.calsour)):            
                if(self.cnum[l] != 0.):
                    for t in range(self.cnum[l]):
                        self.modamp.append(ifdata["model"+str(t+1)+"_amp"])
                        self.modphas.append(ifdata["model"+str(t+1)+"_phas"])
            
            
            # Perform the least-square fitting using Scipy curve_fit.
            time1 = timeit.default_timer()
            Iteration, pco = curve_fit(self.deq, inputx, inputy, p0=init, sigma = outputsigma, absolute_sigma = False, bounds = bounds)
            error = np.sqrt(np.diag(pco))


            self.inputy = np.copy(inputy)
            self.outputsigma = np.copy(outputsigma)
            
            
            # Restore the original antenna numbers.
            insert_index = []
            
            dum = 0
            for it in removed_Index:
                insert_index.append(2*it - 2*dum)
                insert_index.append(2*it - 2*dum)
                insert_index.append(2*self.nant + 2*it - 2*dum)
                insert_index.append(2*self.nant + 2*it - 2*dum)
                dum += 1
                
            Iteration = np.insert(Iteration, insert_index, [0.]*len(insert_index))
            error = np.insert(error, insert_index, [0.]*len(insert_index))
            time2 = timeit.default_timer()
            
            
            self.logger.info('The fitting is completed within {:d} seconds.\n'.format(int(round(time2 - time1))))
            
            
            self.nant = orig_nant
            
            ant1 = np.copy(orig_ant1)
            ant2 = np.copy(orig_ant2)
            
            self.ant1, self.ant2 = np.copy(ant1), np.copy(ant2)
                       
            
            # Calculate the reduced chi-square of the fitting.
            dumfit = self.deq(pang1, *Iteration)
            
            ydata = np.concatenate([lrreal+1j*lrimag, rlreal+1j*rlimag])
            yfit = np.concatenate([dumfit[0:len(lrreal)]+1j*dumfit[len(lrreal):len(lrreal)*2], dumfit[len(lrreal)*2:len(lrreal)*3]+1j*dumfit[len(lrreal)*3:len(lrreal)*4]])
            ysigma = np.concatenate([lrsigma, rlsigma])
            
            chisq = np.sum(np.abs(((ydata - yfit) / ysigma) ** 2)) / (2. * len(ydata))
            
            
            if self.zblcal:
                self.chisq.loc["zbl", "IF"+str(k+1)] = self.zbl_chisq
            
            self.chisq.loc["simil", "IF"+str(k+1)] = chisq
            
            self.logger.info('The reduced chi-square of the fitting is {:5.3f}.\n'.format(chisq))
            

            # Save the best-fit D-terms in the pandas dataframes.
            for m in range(self.nant-1):
                for n in np.arange(m+1, self.nant):
                    self.DRArr.loc["IF"+str(k+1), self.antname[m]] = Iteration[2*m] + 1j*Iteration[2*m+1]
                    self.DRArr.loc["IF"+str(k+1), self.antname[n]] = Iteration[2*n] + 1j*Iteration[2*n+1]
                    self.DLArr.loc["IF"+str(k+1), self.antname[m]] = Iteration[2*self.nant+2*m] + 1j*Iteration[2*self.nant+2*m+1]
                    self.DLArr.loc["IF"+str(k+1), self.antname[n]] = Iteration[2*self.nant+2*n] + 1j*Iteration[2*self.nant+2*n+1]

                    self.DRErr.loc["IF"+str(k+1), self.antname[m]] = error[2*m] + 1j*error[2*m+1]
                    self.DRErr.loc["IF"+str(k+1), self.antname[n]] = error[2*n] + 1j*error[2*n+1]
                    self.DLErr.loc["IF"+str(k+1), self.antname[m]] = error[2*self.nant+2*m] + 1j*error[2*self.nant+2*m+1]
                    self.DLErr.loc["IF"+str(k+1), self.antname[n]] = error[2*self.nant+2*n] + 1j*error[2*self.nant+2*n+1]

            self.DRamp = pd.concat([self.DRArr.applymap(np.absolute)])
            self.DRphas = pd.concat([self.DRArr.applymap(np.angle).applymap(np.degrees)])
            self.DLamp = pd.concat([self.DLArr.applymap(np.absolute)])
            self.DLphas = pd.concat([self.DLArr.applymap(np.angle).applymap(np.degrees)])            
        
            for m in range(self.nant):
                self.logger.info('{:s}: RCP - amplitude = {:6.3f} %, phase = {:7.2f} deg, LCP - amplitude = {:6.3f} %, phase = {:7.2f} deg'.format(self.antname[m], \
                      self.DRamp.loc["IF"+str(k+1), self.antname[m]] * 1e2, self.DRphas.loc["IF"+str(k+1), self.antname[m]], 
                      self.DLamp.loc["IF"+str(k+1), self.antname[m]] * 1e2, self.DLphas.loc["IF"+str(k+1), self.antname[m]]))
                
                simil_IF.append(k+1)
                simil_ant.append(self.antname[m])
                simil_DRArr.append(self.DRArr.loc["IF"+str(k+1), self.antname[m]])
                simil_DLArr.append(self.DLArr.loc["IF"+str(k+1), self.antname[m]])
            
            # Save the best-fit source-polarization terms in the pandas dataframe.
            for l in range(len(self.calsour)):
                if(self.cnum[l] == 0):
                    self.logger.info('\n{:s} is assumed to be unpolarized.'.format(self.calsour[l].replace('.','')))
                    self.sourcepol.loc["IF"+str(k+1), self.calsour[l]] = 0.
                else:
                    self.logger.info('\n{:s} total intensity CLEAN models are divided into {:d} regions.'.format(self.calsour[l].replace('.',''), self.cnum[l]))
                    for i in range(self.cnum[l]):
                        if(l == 0):
                            self.sourcepol.loc["IF"+str(k+1), self.calsour[l]+', R'+str(i+1)] = Iteration[self.nant * 4 + 2*i] + 1j*Iteration[self.nant * 4 + 2*i + 1]
                            dumr = Iteration[self.nant * 4 + 2*i]
                            dumi = Iteration[self.nant * 4 + 2*i + 1]
                        else:
                            self.sourcepol.loc["IF"+str(k+1), self.calsour[l]+', R'+str(i+1)] = Iteration[self.nant * 4 + np.sum(self.cnum[0:l])*2 + 2*i] + 1j*Iteration[self.nant * 4 + np.sum(self.cnum[0:l])*2 + 2*i + 1]
                            dumr = Iteration[self.nant * 4 + np.sum(self.cnum[0:l])*2 + 2*i]
                            dumi = Iteration[self.nant * 4 + np.sum(self.cnum[0:l])*2 + 2*i + 1]
                        self.logger.info(r'Region {:d}: Fractional polarization = {:6.2f} %, EVPA = {:6.2f} deg'.format(i+1, \
                              np.sqrt(dumr ** 2 + dumi ** 2) * 1e2, np.degrees(np.angle(dumr + 1j * dumi)) / 2.))
            
            self.sourcepolamp = pd.concat([self.sourcepol.applymap(np.absolute)])
            self.sourcepolphas = pd.concat([self.sourcepol.applymap(np.angle).applymap(np.degrees)])
            
            
            self.logger.info(' ')
            
            
            mod_lr, mod_rl = dumfit[0:len(time)] + 1j*dumfit[len(time):len(time)*2], dumfit[len(time)*2:len(time)*3] + 1j*dumfit[len(time)*3:len(time)*4]
            mod_q, mod_u = (mod_rl + mod_lr) / 2., -1j * (mod_rl - mod_lr) / 2.
            mod_qamp, mod_qphas, mod_uamp, mod_uphas = np.absolute(mod_q), np.angle(mod_q), np.absolute(mod_u), np.angle(mod_u)
            
            if self.allplot:
                mod_pol_qamp, mod_pol_qphas, mod_pol_uamp, mod_pol_uphas = self.deq_comp('pol', *Iteration)
                mod_dterm_qamp, mod_dterm_qphas, mod_dterm_uamp, mod_dterm_uphas = self.deq_comp('dterm', *Iteration)
                mod_second_qamp, mod_second_qphas, mod_second_uamp, mod_second_uphas = self.deq_comp('second', *Iteration)
                allplots = (mod_pol_qamp, mod_pol_qphas, mod_pol_uamp, mod_pol_uphas, mod_dterm_qamp, mod_dterm_qphas, mod_dterm_uamp, mod_dterm_uphas)
            else:
                allplots = None


            # Create vplots if requested.
            if self.vplot:
                self.logger.info('\nCreating vplots for all baselines... It may take some time.')
                
                parmset = []

                for l in range(len(self.calsour)):
                    for m in range(self.nant):
                        for n in range(self.nant):
                            if(m==n):
                                continue
                            dumm = m
                            dumn = n
                            if(m>n):
                                dumm = n
                                dumn = m
                                
                                select = (ant1 == dumm) & (ant2 == dumn) & (sourcearr == self.calsour[l])
                            
                                selected_time, selected_day, selected_qamp, selected_qphas, selected_qsigma, selected_uamp, selected_uphas, selected_usigma, selected_mod_qamp, selected_mod_qphas, selected_mod_uamp, selected_mod_uphas = \
                                    time[select], dayarr[select], qamp[select], qphas[select], qsigma[select], uamp[select], uphas[select], usigma[select], \
                                    mod_qamp[select], mod_qphas[select], mod_uamp[select], mod_uphas[select]
                                    
                                if self.allplot:
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
                                
                                
                                if self.multiproc:
                                    parmset.append((self.colors[l], self.calsour[l], k+1, self.antname[dumm], self.antname[dumn], \
                                                    selected_day, selected_time, selected_qamp, selected_qphas, selected_qsigma, selected_uamp, selected_uphas, selected_usigma, \
                                                    selected_mod_qamp, selected_mod_qphas, selected_mod_uamp, selected_mod_uphas, self.direc+'gpcal/'+self.outputname+'vplot.IF'+str(k+1), self.filetype, \
                                                    allplots, self.vplot_title, self.vplot_scanavg, self.vplot_avg_nat, self.tsep))
                                else:
                                    ph.visualplot(self.calsour[l], k+1, self.antname[dumm], self.antname[dumn], \
                                                    selected_day, selected_time, selected_qamp, selected_qphas, selected_qsigma, selected_uamp, selected_uphas, selected_usigma, \
                                                    selected_mod_qamp, selected_mod_qphas, selected_mod_uamp, selected_mod_uphas, self.direc+'gpcal/'+self.outputname+'vplot.IF'+str(k+1), self.filetype, \
                                                    allplots = allplots, title = self.dataname, scanavg = self.vplot_scanavg, avg_nat = self.vplot_avg_nat, tsep = self.tsep, color = self.colors[l])
                                            
                if self.multiproc:
                    ph.visualplot_run(parmset, nproc = self.nproc)
                    
                self.logger.info('...completed.\n')
            
                        
            # Create fitting residual plots if requested.
            if self.resplot:
                self.logger.info('Creating fitting residual plots for all stations... It may take some time.')
                # self.residualplot(k, self.nant, self.antname, self.calsour, dumfit, time, ant1, ant2, sourcearr, qamp, qphas, uamp, uphas, qsigma, usigma, self.tsep, \
                #    self.direc+'gpcal/'+self.outputname+'res.IF'+str(k+1))

                ph.residualplot(self.markerarr, self.colors, dayarr, k, self.nant, self.antname, self.calsour, \
                                dumfit, time, ant1, ant2, sourcearr, qamp, qphas, uamp, uphas, qsigma, usigma, self.direc+'gpcal/'+self.outputname+'res.IF'+str(k+1), tsep = 2. / 60., title = self.dataname, filetype = self.filetype)
                    
                self.logger.info('...completed.\n')
                
            del self.pang1, self.pang2, self.ant1, self.ant2, self.sourcearr, self.rramp, self.rrphas, self.llamp, self.llphas 
            
        
        simil_IF, simil_ant, simil_DRArr, simil_DLArr = np.array(simil_IF), np.array(simil_ant), np.array(simil_DRArr), np.array(simil_DLArr)
        
        
        # Create D-term plots.
        self.dplot(-1, self.outputname+'D_Terms', self.antname, self.DRArr, self.DLArr, self.calsour, lpcal=self.lpcal)
        
        
        # Produce the D-term corrected UVFITS files in the working directory.
        self.applydterm(self.source, self.DRArr, self.DLArr)
                
        # Save the fitting results into ASCII files.
        self.chisq.to_csv(self.direc+'gpcal/'+self.outputname+'chisq.txt', sep = "\t")
        self.sourcepol.to_csv(self.direc+'gpcal/'+self.outputname+'sourcepol.txt', sep = "\t")
        self.DRArr.to_csv(self.direc+'gpcal/'+self.outputname+'DRArr.txt', sep = "\t")
        self.DLArr.to_csv(self.direc+'gpcal/'+self.outputname+'DLArr.txt', sep = "\t")
        
        
        
        df = pd.DataFrame(simil_IF.transpose())
        df['antennas'] = np.array(simil_ant)
        df['IF'] = np.array(simil_IF)
        df['DRArr'] = np.array(simil_DRArr)
        df['DLArr'] = np.array(simil_DLArr)
        del df[0]
        
        df.to_csv(self.direc+'gpcal/'+self.outputname+'dterm.csv', sep = "\t")
        


    def pol_dtermsolve(self):
        """
        Estimate the D-terms using instrumental polarization self-calibration. Iterate (i) obtaining D-term solutions and (ii) modeling calibrators Stokes Q and U models with CLEAN as many times as specified by users.
        """ 
        
        # Create D-term plots.
        self.dplot(0, self.outputname+'pol.iter{:02d}'.format(0)+'.D_Terms', self.antname, self.DRArr, self.DLArr, self.calsour, lpcal=False)
                
        # Get the data to be used for the D-term estimation using instrumental polarization self-calibraiton
        self.get_pol_data()
        
        
        if(type(self.ms) == int) | (type(self.ms) == float):
            self.ms = [self.ms] * len(self.polcalsour)
        if(type(self.ps) == int) | (type(self.ps) == float):
            self.ps = [self.ps] * len(self.polcalsour)
        if(type(self.uvbin) == int) | (type(self.uvbin) == float):
            self.uvbin = [self.uvbin] * len(self.polcalsour)
        if(type(self.uvpower) == int) | (type(self.uvpower) == float):
            self.uvpower = [self.uvpower] * len(self.polcalsour)
        if(type(self.dynam) == int) | (type(self.dynam) == float):
            self.dynam = [self.dynam] * len(self.polcalsour)
        if(type(self.shift_x) == int) | (type(self.shift_x) == float):
            self.shift_x = [self.shift_x] * len(self.polcalsour)
        if(type(self.shift_y) == int) | (type(self.shift_y) == float):
            self.shift_y = [self.shift_y] * len(self.polcalsour)
        
        
        
        f = open(self.direc+'GPCAL_Difmap_v1','w')
        f.write('observe %1\nmapcolor rainbow, 1, 0.5\nselect %13, %2, %3\nmapsize %4, %5\nuvweight %6, %7\nrwin %8\nshift %9,%10\ndo i=1,100\nclean 100, 0.02, imstat(rms)*%11\nend do\nselect i\nsave %12.%13\nexit')
        f.close()
            
        
        # Iterate (i) obtaining D-term solutions and (ii) modeling calibrators Stokes Q and U models with CLEAN as many times as specified by users.
        for spoliter in range(self.selfpoliter):
            # Obtain Stokes Q and U models of the calibrators with CLEAN in Difmap.
            
            
            if self.multiproc:
                parmset = []
                
            for l in range(len(self.polcalsour)):
                if self.polcal_unpol[l]:
                    self.logger.info('Skip CLEAN for {:s} because it was assumed to be unpolarized...'.format(self.polcalsour[l]))
                    continue
                
                # self.logger.info('Making CLEAN models for Stokes Q & U maps for {:s}...'.format(self.polcalsour[l]))
                
                if(spoliter == 0): 

                    if self.pol_IF_combine:
                        bif = 1
                        eif = self.ifnum
                        ch.cleanqu(self.direc, self.outputname+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.allIF', \
                                   bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q')
                        ch.cleanqu(self.direc, self.outputname+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.allIF', \
                                   bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u')
                        
                        self.logger.info('\nMaking CLEAN models for Stokes Q & U maps using a single core...\n')
                        
                        self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}\n'.format(self.outputname+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', \
                                          self.dataname+self.polcalsour[l] + '.allIF.q,u'))
                            
                    else:
                        
                        if self.multiproc:
                            
                            for ifn in range(self.ifnum):
                                bif = ifn + 1
                                eif = ifn + 1
                                parmset.append([self.direc, self.outputname+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.IF{:}'.format(ifn+1), \
                                           bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q'])
                                parmset.append([self.direc, self.outputname+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.IF{:}'.format(ifn+1), \
                                            bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u'])
                            
                        else:
                            
                            for ifn in range(self.ifnum):
                                bif = ifn + 1
                                eif = ifn + 1
                                ch.cleanqu(self.direc, self.outputname+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.IF{:}'.format(ifn+1), \
                                            bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q')
                                ch.cleanqu(self.direc, self.outputname+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.IF{:}'.format(ifn+1), \
                                            bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u')
                                
                                self.logger.info('\nMaking CLEAN models for Stokes Q & U maps using a single core...\n')
                                
                                self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}\n'.format(self.outputname+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', \
                                                  self.dataname+self.polcalsour[l] + '.IF{:}.q,u'.format(ifn+1)))
                        
                        
                else:
                    
                    if self.pol_IF_combine:
                        
                        bif = 1
                        eif = self.ifnum
                        ch.cleanqu(self.direc, self.outputname+'pol.iter'+str(spoliter)+'.'+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.allIF', \
                                   bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q')
                        ch.cleanqu(self.direc, self.outputname+'pol.iter'+str(spoliter)+'.'+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.allIF', \
                                   bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u')
                    
                        self.logger.info('\nMaking CLEAN models for Stokes Q & U maps using a single core...\n')
                        
                        self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}\n'.format(self.outputname+'pol.iter'+str(spoliter)+'.'+self.polcalsour[l]+'.dtcal.uvf', \
                                                                                                         self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.allIF.q,u'))
                            
                    else:
                        
                        if self.multiproc:
                            for ifn in range(self.ifnum):
                                bif = ifn + 1
                                eif = ifn + 1
                                parmset.append([self.direc, self.outputname+'pol.iter'+str(spoliter)+'.'+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.IF{:}'.format(ifn+1), \
                                           bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q'])
                                parmset.append([self.direc, self.outputname+'pol.iter'+str(spoliter)+'.'+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.IF{:}'.format(ifn+1), \
                                           bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u'])
                            
                                    
                        else:
                            for ifn in range(self.ifnum):
                                bif = ifn + 1
                                eif = ifn + 1
                                ch.cleanqu(self.direc, self.outputname+'pol.iter'+str(spoliter)+'.'+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.IF{:}'.format(ifn+1), \
                                           bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q')
                                ch.cleanqu(self.direc, self.outputname+'pol.iter'+str(spoliter)+'.'+self.polcalsour[l]+'.dtcal.uvf', self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.IF{:}'.format(ifn+1), \
                                           bif, eif, self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u')
                                
                                self.logger.info('\nMaking CLEAN models for Stokes Q & U maps using a single core...\n')
                                
                                self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}\n'.format(self.outputname+'pol.iter'+str(spoliter)+'.'+self.polcalsour[l]+'.dtcal.uvf', \
                                                                                                                 self.dataname+self.polcalsour[l]+'.win', self.dataname+self.polcalsour[l] + '.IF{:}.q,u'.format(ifn+1)))
            
            
            if self.multiproc:                
                pool = Pool(processes = self.nproc)
                pool.map(ch.cleanqu2, parmset)
                pool.close()
                pool.join()
                
                self.logger.info('\nMaking CLEAN models for Stokes Q & U maps using multiple cores...\n')
                
                for i in range(len(parmset)):
                    self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}.q,u'.format(parmset[i][1], parmset[i][2], parmset[i][3]))
            
            
            # Extract visibility models using the CLEAN Stokes Q and U models.
            self.pol_data = oh.get_model(self.pol_data, self.direc, self.dataname, self.polcalsour, self.polcal_unpol, self.ifnum, self.pol_IF_combine, selfcal = self.selfcal, outputname = self.outputname)
            
            curdirec = os.getcwd()
            
            os.chdir(self.direc)
            
            os.system('rm {:}difmap.log*')
            
            os.chdir(curdirec)
            
            
            if self.pol_IF_combine:
                for l in range(len(self.polcalsour)):
                    os.system('rm {:}{:}{:}.allIF.q.*'.format(self.direc, self.dataname, self.polcalsour[l]))
                    os.system('rm {:}{:}{:}.allIF.u.*'.format(self.direc, self.dataname, self.polcalsour[l]))
            else:
                for ifn in range(self.ifnum):
                    for l in range(len(self.polcalsour)):
                        os.system('rm {:}{:}{:}.IF{:}.q.*'.format(self.direc, self.dataname, self.polcalsour[l], ifn+1))
                        os.system('rm {:}{:}{:}.IF{:}.u.*'.format(self.direc, self.dataname, self.polcalsour[l], ifn+1))
                        
            
            if(spoliter == 0):
                self.logger.info('\n####################################################################')
                self.logger.info('Instrumental polarization self-calibration mode...\n')
                sourcename = ''
                for i in range(len(self.polcalsour)):
                    sourcename = sourcename+('{:s}'.format(self.polcalsour[i]))+','
                sourcename = sourcename[:-1]
                self.logger.info('{:d} data from {:s} source(s) will be used:'.format(len(self.pol_data["time"]), str(len(self.polcalsour))) + '\n')
                self.logger.info('Source coordinates:')
                for i in range(len(self.polcalsour)):
                    self.logger.info('{:s}: RA = {:5.2f} deg, Dec = {:5.2f} deg'.format(self.polcalsour[i], self.pol_obsra[i], self.pol_obsdec[i]))
                self.logger.info('\nAntenna information:')
                for i in range(self.nant):
                    if(self.antmount[i] == 0): mount = 'Cassegrain'
                    if(self.antmount[i] == 4): mount = 'Nasmyth-Right'
                    if(self.antmount[i] == 5): mount = 'Nasmyth-Left'
                    self.logger.info('{:s}: antenna mount = {:13s}, X = {:11.2f}m, Y = {:11.2f}m, Z = {:11.2f}m'.format(self.antname[i], mount, self.antx[i], self.anty[i], self.antz[i]))
                        
                
                self.logger.info(' ')
                
                self.logger.info('Observation date = {:s}'.format(str(np.min(self.year))+'-'+str(np.min(self.month))+'-'+str(np.min(self.day))))
            
            
            pol_IF, pol_ant, pol_DRArr, pol_DLArr = [], [], [], []
            
            for ifn in range(self.ifnum):
                
                ifdata = self.pol_data.loc[self.pol_data["IF"] == ifn+1]
                
                if(np.sum(self.pol_data["IF"] == ifn+1) == 0.):
                    self.logger.info('Will skip this IF because there is no data.\n')
                    continue
                
                self.logger.info('\n####################################################################')
                self.logger.info('Processing IF {:d}'.format(ifn+1) + '...')
                self.logger.info('####################################################################\n')
                
                self.logger.info('Instrumental polarization self-calibration mode...')
                self.logger.info('Iteration: {:d}/{:d}\n'.format(spoliter+1, self.selfpoliter))
                
                
                dumantname = np.copy(self.antname)
                
                if not self.fixdterm:
                    self.fixdr = None
                    self.fixdl = None

                # If the zero-baseline D-term estimation was performed, then fix the D-terms of those stations for fitting for the rest of the array.
                if self.zblcal:
                    self.fixdterm = True
                    if(self.fixdr == None):
                        self.fixdr = {}
                        self.fixdl = {}
                    for i in range(len(self.zblant)):
                        if(self.zblant[i][0] in self.fixdr):
                            self.fixdr[self.zblant[i][0]] = self.zbl_DRArr.loc["IF"+str(ifn+1), self.zblant[i][0]]
                            self.fixdl[self.zblant[i][0]] = self.zbl_DLArr.loc["IF"+str(ifn+1), self.zblant[i][0]]
                        else:
                            self.fixdr.update({self.zblant[i][0]: self.zbl_DRArr.loc["IF"+str(ifn+1), self.zblant[i][0]]})
                            self.fixdl.update({self.zblant[i][0]: self.zbl_DLArr.loc["IF"+str(ifn+1), self.zblant[i][0]]})
                        
                        if(self.zblant[i][1] in self.fixdr):
                            self.fixdr[self.zblant[i][1]] = self.zbl_DRArr.loc["IF"+str(ifn+1), self.zblant[i][1]]
                            self.fixdl[self.zblant[i][1]] = self.zbl_DLArr.loc["IF"+str(ifn+1), self.zblant[i][1]]
                        else:
                            self.fixdr.update({self.zblant[i][1]: self.zbl_DRArr.loc["IF"+str(ifn+1), self.zblant[i][1]]})
                            self.fixdl.update({self.zblant[i][1]: self.zbl_DLArr.loc["IF"+str(ifn+1), self.zblant[i][1]]})

                
                if (spoliter == self.selfpoliter - 1) & (self.vplot):
                    dum_vplot = True
                else:
                    dum_vplot = False
                
                if (spoliter == self.selfpoliter - 1) & (self.vplot):
                    dum_resplot = True
                else:
                    dum_resplot = False
                    
                    
                fitresults, fitdterms, chisq = ps.pol_gpcal(self.direc, self.outputname, ifdata, self.polcalsour, self.antname, self.logger, Dbound = self.Dbound, Pbound = self.Pbound, \
                                                     manualweight = self.manualweight, weightfactors = self.weightfactors, fixdterm = self.fixdterm, fixdr = self.fixdr, fixdl = self.fixdl, \
                                                     multiproc = self.multiproc, nproc = self.nproc, colors = self.colors, vplot = dum_vplot, allplot = self.allplot, \
                                                     vplot_title = self.vplot_title, vplot_scanavg = self.vplot_scanavg, vplot_avg_nat = self.vplot_avg_nat, tsep = self.tsep, \
                                                     resplot = dum_resplot, markerarr = self.markerarr)
                
                    
                dum_ant, dum_DRArr, dum_DLArr = fitdterms
                dum_IF = np.repeat(ifn+1, len(dum_ant))
                
                pol_ant = pol_ant + dum_ant.tolist()
                pol_DRArr = pol_DRArr + dum_DRArr.tolist()
                pol_DLArr = pol_DLArr + dum_DLArr.tolist()
                pol_IF = pol_IF + dum_IF.tolist()
                
                    
            pol_IF, pol_ant, pol_DRArr, pol_DLArr = np.array(pol_IF), np.array(pol_ant), np.array(pol_DRArr), np.array(pol_DLArr)
                
            
            df = pd.DataFrame(pol_IF.transpose())
            df['antennas'] = np.array(pol_ant)
            df['IF'] = np.array(pol_IF)
            df['DRArr'] = np.array(pol_DRArr)
            df['DLArr'] = np.array(pol_DLArr)
            del df[0]
            
            df.to_csv(self.direc+'gpcal/'+self.outputname+'pol.iter'+str(spoliter+1)+'.dterm.csv', sep = "\t")
    
            # Create D-term plots.
            self.dplot_new(spoliter+1, self.outputname+'pol.iter{:02d}'.format(spoliter+1)+'.D_Terms', pol_ant, pol_IF, pol_DRArr, pol_DLArr, self.polcalsour, lpcal=False)
            
            # Apply the best-fit D-terms and produce the D-term corrected UVFITS files.
            self.pol_applydterm(self.source, self.direc+'gpcal/'+self.outputname+'pol.iter'+str(spoliter+1)+'.dterm.csv', self.direc+self.outputname+'pol.iter'+str(spoliter+1)+'.')

        # Save the reduced chi-squares into ASCII files.
        self.chisq.to_csv(self.direc+'gpcal/'+self.outputname+'chisq.txt', sep = "\t")
        
        # Create a reduced chi-square plot.
        self.chisqplot(self.chisq)
        

