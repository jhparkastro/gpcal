

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
        

class timecal(object):
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
    def __init__(self, aips_userno, direc, timecalname, calsour, source, outputname = None, Dbound = 1.0, Pbound = np.inf, freqavg = True, timecal_iter = 10, timecal_freepar = 2, \
                 drange = None, init_change_source = False, init_change_calsour = None, polcal_unpol = None, remove_weight_outlier_threshold = 10000, \
                 ms = None, ps = None, uvbin = None, uvpower = None, shift_x = 0, shift_y = 0, dynam = None, clean_IF_combine = False, multiproc = True, nproc = 2, \
                 manualweight = False, weightfactors = None, tsep = 2./60., filetype = 'pdf', aipslog = True):
        
        
        self.multiproc = multiproc
        self.nproc = nproc
        
        self.aips_userno = aips_userno
        
        self.direc = direc
        
        # If direc does not finish with a slash, then append it.
        if(self.direc[-1] != '/'): self.direc = self.direc + '/'

        # Create a folder named 'gpcal' in the working directory if it does not exist.
        if(os.path.exists(self.direc+'gpcal') == False):
            os.system('mkdir ' + self.direc+'gpcal') 
        
        self.timecalname = timecalname
        
        if(self.timecalname[-1] != '.'): self.timecalname = self.timecalname + '.'
        
        self.Dbound = Dbound
        self.Pbound = Pbound
        
        self.freqavg = freqavg
        self.timecal_freepar = timecal_freepar
        
        self.calsour = calsour
        self.source = source
        
        self.timecal_iter = timecal_iter
                
        self.init_change_calsour = init_change_calsour
        self.init_change_source = init_change_source
        
        
        self.remove_weight_outlier_threshold = remove_weight_outlier_threshold
        
        if(self.timecalname[-1] != '.'): self.timecalname = self.timecalname + '.'
        
        self.outputname = copy.deepcopy(outputname)
        if(self.outputname == None):
            self.outputname = copy.deepcopy(self.timecalname)
        else:
            if(self.outputname[-1] != '.'): self.outputname = self.outputname + '.'
        
        
        self.polcal_unpol = polcal_unpol
        if(self.polcal_unpol == None): 
            self.polcal_unpol = [False] * len(self.calsour)
        
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
        
        
        # Setup of logging
        if path.exists(direc+'gpcal/'+timecalname+'gpcal.log'):
            os.system('rm ' + direc+'gpcal/'+timecalname+'gpcal.log')
        self.logfile = direc+'gpcal/'+timecalname+'gpcal.log'
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        logs = logging.StreamHandler()
        logformat = logging.Formatter('%(message)s')
        logs.setFormatter(logformat)
        if not len(self.logger.handlers):
            self.logger.addHandler(logs)
            logf = logging.FileHandler(self.logfile)
            self.logger.addHandler(logf)
        

    def get_timecal_data(self, tditer):
        """
        Make a pandas dataframe containing UV data and models for the D-term estimation with instrumental polarization self-calibration.
        
        """

        l = 0
    
        inname = str(self.calsour[l])

        if(tditer == 0):    
            if self.freqavg:
                uvfname = self.direc + self.timecalname + self.calsour[l] + '.IFavg.uvf'
            else:
                uvfname = self.direc + self.timecalname + self.calsour[l] + '.uvf'
        else:
            uvfname = self.direc + self.outputname + 'timecal.iter{:}.'.format(tditer) + self.calsour[l] + '.uvf'
        
                
        au.runfitld(inname, 'EDIT', uvfname)

        data = WAIPSUVData(inname, 'EDIT', 1, 1)
        
        
        self.antname, self.antx, self.anty, self.antz, self.antmount, self.f_par, self.f_el, self.phi_off = oh.get_antcoord(data)
        
        self.ifnum, self.freq = oh.get_freqinfo(data)
        
        self.lonarr, self.latarr, self.heightarr = oh.coord(self.antname, self.antx, self.anty, self.antz)
        
        self.nant = len(self.antname)    

        data.zap()

        
        self.logger.info('\nGetting data for {:d} sources for {:d} IFs...'.format(len(self.calsour), self.ifnum))
            
        AIPS.userno = self.aips_userno
        
        if self.aipslog:
            AIPS.log = open(self.logfile, 'a')
        AIPSTask.msgkill = -1
        
        self.obsra = []
        self.obsdec = []
        
        self.year, self.month, self.day = [], [], [] # Version 1.1!
        
        
        # Define a pandas dataframe for the data array.
        pdkey = ["IF", "year", "month", "day", "time", "source", "ant1", "ant2", "u", "v", "pang1", "pang2", \
                 "rrreal", "rrimag", "rrsigma", "llreal", "llimag", "llsigma", "rlreal", "rlimag", "rlsigma", "lrreal", "lrimag", "lrsigma", \
                 "rramp", "rrphas", "rramp_sigma", "rrphas_sigma", "llamp", "llphas", "llamp_sigma", "llphas_sigma", "rlamp", "rlphas", "rlamp_sigma", "rlphas_sigma", "lramp", "lrphas", "lramp_sigma", "lrphas_sigma", \
                 "qamp", "qphas", "qamp_sigma", "qphas_sigma", "qsigma", "uamp", "uphas", "uamp_sigma", "uphas_sigma", "usigma"]

        self.data = pd.DataFrame(columns = pdkey)       
        
        
        ifarr, timearr, sourcearr, ant1arr, ant2arr, uarr, varr, rrrealarr, rrimagarr, rrweightarr, llrealarr, llimagarr, llweightarr, rlrealarr, rlimagarr, rlweightarr, lrrealarr, lrimagarr, lrweightarr = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []


        for l in range(len(self.calsour)):
        
            inname = str(self.calsour[l])


            if(tditer == 0):    
                if self.freqavg:
                    uvfname = self.direc + self.timecalname + self.calsour[l] + '.IFavg.uvf'
                else:
                    uvfname = self.direc + self.timecalname + self.calsour[l] + '.uvf'
            else:
                uvfname = self.direc + self.outputname + 'timecal.iter{:}.'.format(tditer) + self.calsour[l] + '.uvf'
            
            au.runfitld(inname, 'EDIT', uvfname)
            
            
            data = WAIPSUVData(inname, 'EDIT', 1, 1)
            
            year, month, day = oh.get_obsdate(data)
            
            self.year.append(year)
            self.month.append(month)
            self.day.append(day)
            
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
            
            obsra, obsdec = oh.get_obscoord(data)

            self.obsra.append(obsra)
            self.obsdec.append(obsdec)
            
            data.zap()


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
            dayarr[timearr>=24.] += 1 # Version 1.1!
            timearr[timearr>=24.] -= 24. # Version 1.1!
            
        self.data.loc[:,"year"] = yeararr
        self.data.loc[:,"month"] = montharr
        self.data.loc[:,"day"] = dayarr
                
        self.data.loc[:,"pang1"] = oh.get_parang(yeararr, montharr, dayarr, timearr, raarr, decarr, longarr1, latarr1, f_el1, f_par1, phi_off1)
        self.data.loc[:,"pang2"] = oh.get_parang(yeararr, montharr, dayarr, timearr, raarr, decarr, longarr2, latarr2, f_el2, f_par2, phi_off2)
        
        self.data = oh.pd_modifier(self.data)
        

    def tdplot(self, read, filename):
        """
        Draw fitting residual plots.
        """  
        
        dtermread = pd.read_csv(read, header = 0, skiprows=0, delimiter = '\t', index_col = 0)
            
        var_ant = np.array(dtermread['antennas'])
        var_IF = np.array(dtermread['IF'])
        var_scan = np.array(dtermread['scan'])
        var_scantime = np.array(dtermread['scantime'])
        var_scansource = np.array(dtermread['scansource'])
        dum_DRArr = np.array(dtermread['DRArr'])
        dum_DLArr = np.array(dtermread['DLArr'])
        
        var_DRArr = np.array([complex(it) for it in dum_DRArr])
        var_DLArr = np.array([complex(it) for it in dum_DLArr])
        
        
        uniqant = np.unique(var_ant)
        uniqIF = np.unique(var_IF)
        uniqsource = np.unique(var_scansource)
        
        for m in range(len(uniqant)):
            for k in range(len(uniqIF)):
                
                select = (var_ant == uniqant[m])
                if(np.sum(select) == 0): continue
                
                figure, axes = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0}, figsize=(8, 8))
        
                for ax in axes.flat:
                    ax.tick_params(length=6, width=2,which = 'major')
                    ax.tick_params(length=4, width=1.5,which = 'minor')
                    
                axes[0].set_xlim(np.min(var_scantime) - (np.max(var_scantime) - np.min(var_scantime)) * 0.4, np.max(var_scantime) + (np.max(var_scantime) - np.min(var_scantime)) * 0.1)
                axes[1].set_xlim(np.min(var_scantime) - (np.max(var_scantime) - np.min(var_scantime)) * 0.4, np.max(var_scantime) + (np.max(var_scantime) - np.min(var_scantime)) * 0.1)
                axes[1].set(xlabel = 'Time (UT)')
                    
                axes[0].set(ylabel = 'D-Term real (\%)')
                axes[1].set(ylabel = 'D-Term imaginary (\%)')
                
                axes[0].set_title(self.timecalname)
                
                axes[1].annotate(uniqant[m], xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25), size = 24, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
                axes[1].annotate('IF {:d}'.format(k+1), xy = (0, 0), xycoords = 'axes fraction', xytext = (25, 25), size = 24, textcoords='offset pixels', horizontalalignment='left', verticalalignment='bottom')
                
                for l in range(len(uniqsource)):
                
                    select = (var_ant == uniqant[m]) & (var_scansource == uniqsource[l]) & (var_IF == uniqIF[k])
                    
                    if(np.sum(select) == 0): continue
                
                    selected_time = var_scantime[select]
                    selected_drarr = var_DRArr[select]
                    selected_dlarr = var_DLArr[select]
                    
                    select = (np.abs(selected_drarr) != 0.) & (np.abs(selected_dlarr) != 0.)
                    
                    selected_time = selected_time[select]
                    selected_drarr = selected_drarr[select]
                    selected_dlarr = selected_dlarr[select]
                    
                    axes[0].scatter(selected_time, np.real(selected_drarr) * 1e2, s = 20, marker = 's', facecolor = self.colors[l], edgecolor = self.colors[l], label = uniqsource[l].upper())
                    axes[0].scatter(selected_time, np.real(selected_dlarr) * 1e2, s = 20, marker = '^', facecolor = 'None', edgecolor = self.colors[l])
                    axes[1].scatter(selected_time, np.imag(selected_drarr) * 1e2, s = 20, marker = 's', facecolor = self.colors[l], edgecolor = self.colors[l])
                    axes[1].scatter(selected_time, np.imag(selected_dlarr) * 1e2, s = 20, marker = '^', facecolor = 'None', edgecolor = self.colors[l])
                    
                
                leg1 = axes[0].legend(loc='upper left', fontsize = 18 - int(len(uniqsource)/2.), frameon=False, markerfirst=True, handlelength=1.0)
                
                rcp = axes[0].scatter([], [], s = 120, facecolor = 'black', edgecolor = 'black', marker = 's')
                lcp = axes[0].scatter([], [], s = 120, facecolor = 'none', edgecolor = 'black', marker = 's')
                
                leg2 = axes[0].legend([rcp, lcp], ['Filled - RCP', 'Open - LCP'], loc = 'lower left', frameon=False, fontsize = 18, handlelength=0.3)
                
                axes[0].add_artist(leg1)
                
                xticks = axes[1].get_xticks()
                dumxticks = np.copy(xticks)
                dumxticks[dumxticks > 24.] -= 24.
                
                xticklabel = [str(int(it)) for it in dumxticks]
                
                axes[0].set_xticks(xticks)
                axes[1].set_xticks(xticks)
                axes[1].set_xticklabels(xticklabel)

            
                figure.savefig(filename + '.' + uniqant[m] + '.IF{:}.'.format(k+1) + self.filetype, bbox_inches = 'tight')
                
                plt.close('all')
        
    
    
    def dtermsolve(self):
        """
        Estimate the D-terms using instrumental polarization self-calibration. Iterate (i) obtaining D-term solutions and (ii) modeling calibrators Stokes Q and U models with CLEAN as many times as specified by users.
        """ 
        
        inname = 'timecal'
        
        au.runfitld(inname, 'EDIT', self.direc + self.timecalname + self.calsour[0] + '.uvf')

        data = WAIPSUVData(inname, 'EDIT', 1, 1)
        
        if self.freqavg:
            self.ifnum = 1
        else:
            self.ifnum = data.header['naxis'][3]
                
        index = []
        
        for it in range(self.timecal_iter + 1):
            index.append("timecal_iter"+str(it))
        
        self.timecal_chisq = pd.DataFrame(index = index, columns = ['IF'+str(it+1) for it in np.arange(self.ifnum)])
        
        
        if self.freqavg:
            for l in range(len(self.calsour)):
                au.freqavg(self.direc + self.timecalname + self.calsour[l] + '.uvf', self.direc + self.timecalname + self.calsour[l] + '.IFavg.uvf', logger = self.logger)
        
            for l in range(len(self.source)):
                au.freqavg(self.direc + self.timecalname + self.source[l] + '.uvf', self.direc + self.timecalname + self.source[l] + '.IFavg.uvf', logger = self.logger)
                
            if self.init_change_source:
                for l in range(len(self.init_change_calsour)):
                    au.freqavg(self.direc + self.timecalname + self.init_change_calsour[l] + '.uvf', self.direc + self.timecalname + self.init_change_calsour[l] + '.IFavg.uvf', logger = self.logger)
        
        
        if(type(self.ms) == int) | (type(self.ms) == float):
            self.ms = [self.ms] * len(self.calsour)
        if(type(self.ps) == int) | (type(self.ps) == float):
            self.ps = [self.ps] * len(self.calsour)
        if(type(self.uvbin) == int) | (type(self.uvbin) == float):
            self.uvbin = [self.uvbin] * len(self.calsour)
        if(type(self.uvpower) == int) | (type(self.uvpower) == float):
            self.uvpower = [self.uvpower] * len(self.calsour)
        if(type(self.dynam) == int) | (type(self.dynam) == float):
            self.dynam = [self.dynam] * len(self.calsour)
        if(type(self.shift_x) == int) | (type(self.shift_x) == float):
            self.shift_x = [self.shift_x] * len(self.calsour)
        if(type(self.shift_y) == int) | (type(self.shift_y) == float):
            self.shift_y = [self.shift_y] * len(self.calsour)
        
        
        
        f = open(self.direc+'GPCAL_Difmap_v1','w')
        f.write('observe %1\nmapcolor rainbow, 1, 0.5\nselect %13, %2, %3\nmapsize %4, %5\nuvweight %6, %7\nrwin %8\nshift %9,%10\ndo i=1,100\nclean 100, 0.02, imstat(rms)*%11\nend do\nselect i\nsave %12.%13\nexit')
        f.close()
            
        
        for tditer in range(self.timecal_iter):
            
            self.get_timecal_data(tditer)
            
            self.antname = np.array(self.antname)
            
            # Obtain Stokes Q and U models of the calibrators with CLEAN in Difmap.
            
            if self.multiproc:
                parmset = []
                
            for l in range(len(self.calsour)):
                if self.polcal_unpol[l]:
                    self.logger.info('Skip CLEAN for {:s} because it was assumed to be unpolarized...'.format(self.calsour[l]))
                    continue                
                
                if(tditer == 0):
                    if self.freqavg:
                        uvfname = self.direc + self.timecalname + self.calsour[l] + '.IFavg.uvf'
                    else:
                        uvfname = self.direc + self.timecalname + self.calsour[l] + '.uvf'
                else:
                    uvfname = self.direc + self.outputname + 'timecal.iter{:}.'.format(tditer) + self.calsour[l] + '.uvf'
                
                
                if self.clean_IF_combine:
                    
                    bif = 1
                    eif = self.ifnum
                    
                    if self.multiproc:
                        parmset.append([self.direc, uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.calsour[l] + '.allIF', bif, eif, \
                               self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q'])
                        parmset.append([self.direc, uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.calsour[l] + '.allIF', bif, eif, \
                               self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u'])
                            
                    else:
                        
                        ch.cleanqu(self.direc, uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.calsour[l] + '.allIF', bif, eif, \
                               self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q')
                        ch.cleanqu(self.direc, uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.calsour[l] + '.allIF', bif, eif, \
                               self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u')
                        
                        self.logger.info('\nMaking CLEAN models for Stokes Q & U maps using a single core...\n')
                        
                        self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}\n'.format(uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', \
                                          self.direc + self.timecalname + self.calsour[l] + '.allIF.q,u'))
                    
                else:
                    
                    if self.multiproc:
                        for ifn in range(self.ifnum):
                            bif = ifn + 1
                            eif = ifn + 1
                            parmset.append([self.direc, uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.calsour[l] + '.IF{:}'.format(ifn+1), bif, eif, \
                                   self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q'])
                            parmset.append([self.direc, uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.calsour[l] + '.IF{:}'.format(ifn+1), bif, eif, \
                                   self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u'])
                        
                    else:
                                                                
                        for ifn in range(self.ifnum):
                            bif = ifn + 1
                            eif = ifn + 1
                            
                            ch.cleanqu(self.direc, uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.calsour[l] + '.IF{:}'.format(ifn+1), bif, eif, \
                                   self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q')
                            ch.cleanqu(self.direc, uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.calsour[l] + '.IF{:}'.format(ifn+1), bif, eif, \
                                   self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u')
                            
                            self.logger.info('\nMaking CLEAN models for Stokes Q & U maps using a single core...\n')
                            
                            self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}\n'.format(uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', \
                                              self.direc + self.timecalname + self.calsour[l] + '.IF{:}.q,u'.format(ifn+1)))
                
                
                os.system('cp {:} {:}'.format(self.direc + self.timecalname + self.calsour[l]+'.uvf', self.direc + self.timecalname + self.calsour[l]+'.uvf.backup'))
                if(uvfname != self.direc + self.timecalname + self.calsour[l]+'.uvf'):
                    os.system('cp {:} {:}'.format(uvfname, self.direc + self.timecalname + self.calsour[l]+'.uvf'))
            
            
            if self.multiproc:                
                pool = Pool(processes = self.nproc)
                pool.map(ch.cleanqu2, parmset)
                pool.close()
                pool.join()
                
                self.logger.info('\nMaking CLEAN models for Stokes Q & U maps using multiple cores...\n')
                
                for i in range(len(parmset)):
                    self.logger.info('uvfits file: {:s}, CLEAN mask: {:s}, save file: {:s}.q,u'.format(parmset[i][1], parmset[i][2], parmset[i][3]))
            
                        
            if (self.init_change_source) & (tditer == 0):
                for l in range(len(self.init_change_calsour)):
                    if(self.calsour[l] == self.init_change_calsour[l]):
                        continue
                    
                    f = open(self.direc + 'GPCAL_Difmap_initchange','w')
                    f.write('@%1\nselect i\nclrmod tru\nrmodel %2\nsave %3\nrmodel %4\nsave %5\nexit')
                    f.close()
                    
                    curdirec = os.getcwd()
                
                    if self.freqavg:
                        dumuvfname = self.direc + self.timecalname + self.init_change_calsour[l] + '.IFavg.uvf'
                    else:
                        dumuvfname = self.direc + self.timecalname + self.init_change_calsour[l] + '.uvf'
                    
                    if self.clean_IF_combine:
                    
                        ch.cleanqu(self.direc, dumuvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.init_change_calsour[l] + '.allIF', bif, eif, \
                                self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q')
                        ch.cleanqu(self.direc, dumuvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.init_change_calsour[l] + '.allIF', bif, eif, \
                                self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u')
                                                
                        os.chdir(self.direc)
                        command = "echo @GPCAL_Difmap_initchange %s,%s,%s,%s,%s | difmap > /dev/null 2>&1" %(self.timecalname + self.calsour[l] + '.allIF.q.par', self.timecalname + self.init_change_calsour[l] + '.allIF.q.mod', \
                                                                                        self.timecalname + self.calsour[l] + '.allIF.q', self.timecalname + self.init_change_calsour[l] + '.allIF.u.mod', \
                                                                                        self.timecalname + self.calsour[l] + '.allIF.u')
                        
                        os.system(command)
                        
                        os.chdir(curdirec)
                    
                    else:
                        
                        for ifn in range(self.ifnum):
                            bif = ifn + 1
                            eif = ifn + 1
                            
                            ch.cleanqu(self.direc, uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.init_change_calsour[l] + '.IF{:}'.format(ifn+1), bif, eif, \
                                    self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'q')
                            ch.cleanqu(self.direc, uvfname, self.direc + self.timecalname + self.calsour[l] + '.win', self.direc + self.timecalname + self.init_change_calsour[l] + '.IF{:}'.format(ifn+1), bif, eif, \
                                    self.ms[l], self.ps[l], self.uvbin[l], self.uvpower[l], self.dynam[l], self.shift_x[l], self.shift_y[l], 'u')
                            
                            
                            os.chdir(self.direc)
                            command = "echo @GPCAL_Difmap_initchange %s,%s,%s,%s,%s | difmap > /dev/null 2>&1" %(self.timecalname + self.calsour[l] + '.IF{:}.q.par'.format(ifn+1), self.timecalname + self.init_change_calsour[l] + '.IF{:}.q.mod'.format(ifn+1), \
                                                                                            self.timecalname + self.calsour[l] + '.IF{:}.q'.format(ifn+1), self.timecalname + self.init_change_calsour[l] + '.IF{:}.u.mod'.format(ifn+1), \
                                                                                            self.timecalname + self.calsour[l] + '.IF{:}.u'.format(ifn+1))
                            os.system(command)
                            
                            os.chdir(curdirec)
                            
            
            self.data = oh.get_model(self.data, self.direc, self.timecalname, self.calsour, self.polcal_unpol, self.ifnum, self.clean_IF_combine, selfcal = False)
                                    
            for l in range(len(self.calsour)):
                os.system('mv {:} {:}'.format(self.direc + self.timecalname + self.calsour[l]+'.uvf.backup', self.direc + self.timecalname + self.calsour[l]+'.uvf'))
            
            
            curdirec = os.getcwd()
            
            os.chdir(self.direc)
            
            os.system('rm difmap.log*')
            
            os.chdir(curdirec)
            
            
            # if self.clean_IF_combine:
            #     for l in range(len(self.calsour)):
            #         os.system('rm {:}{:}{:}.allIF.q.*'.format(self.direc, self.timecalname, self.calsour[l]))
            #         os.system('rm {:}{:}{:}.allIF.u.*'.format(self.direc, self.timecalname, self.calsour[l]))
            # else:
            #     for ifn in range(self.ifnum):
            #         for l in range(len(self.calsour)):
            #             os.system('rm {:}{:}{:}.IF{:}.q.*'.format(self.direc, self.timecalname, self.calsour[l], ifn+1))
            #             os.system('rm {:}{:}{:}.IF{:}.u.*'.format(self.direc, self.timecalname, self.calsour[l], ifn+1))
                        
            
            if(tditer == 0):
                self.logger.info('\n####################################################################')
                self.logger.info('Time-dependent D-term correction mode...\n')
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
            
            
        
            time = np.array(self.data["time"])
            sourcearr = np.array(self.data["source"])
            
            boundary_left, boundary_right, boundary_source = oh.get_scans(time, sourcearr, tsep = self.tsep)
            boundary_time = [(it1 + it2) / 2. for it1, it2 in zip(boundary_left, boundary_right)]
            
            scannum = len(boundary_left)
            
            nant = len(self.antname)
            
            timecal_DRArr, timecal_DLArr, timecal_IF, timecal_ant, timecal_scan, timecal_scantime, timecal_scansource = [], [], [], [], [], [], []
            
            for ifn in range(self.ifnum):
                
                ifdata = self.data.loc[self.data["IF"] == ifn+1]
                
                if(np.sum(self.data["IF"] == ifn+1) == 0.):
                    self.logger.info('Will skip this IF because there is no data.\n')
                    continue
                
                
                self.logger.info('\n####################################################################')
                self.logger.info('Processing IF {:d}'.format(ifn+1) + '...')
                self.logger.info('####################################################################\n')
                
                self.logger.info('Time-dependent D-term correction mode...')
                self.logger.info('Iteration: {:d}/{:d}\n'.format(tditer+1, self.timecal_iter))
                
                
                iftime = np.array(ifdata["time"])
                
                orig_chisq_dev, new_chisq_dev, orig_chisq_n, new_chisq_n = [], [], [], []
                
                
                for s in range(scannum):
                    
                    start_day = int(np.floor(boundary_left[s] / 24.))
                    start_hour = int(np.floor(boundary_left[s] - start_day * 24.))
                    start_minute = int(np.floor((boundary_left[s] - start_day * 24. - start_hour) * 60.))
                    start_second = int(np.floor(((boundary_left[s] - start_day * 24. - start_hour) * 60. - start_minute) * 60.))
                    
                    end_day = int(np.floor(boundary_right[s] / 24.))
                    end_hour = int(np.floor(boundary_right[s] - end_day * 24.))
                    end_minute = int(np.floor((boundary_right[s] - end_day * 24. - end_hour) * 60.))
                    end_second = int(np.floor(((boundary_right[s] - end_day * 24. - end_hour) * 60. - end_minute) * 60.))
                    
                    self.logger.info('Processing scan {:04d}, time: {:d}/{:02d}:{:02d}:{:02d}-{:02d}/{:02d}:{:02d}:{:02d}, source: {:s}...'.format(s+1, start_day, start_hour, start_minute, start_second, \
                                                                                                                                   end_day, end_hour, end_minute, end_second, boundary_source[s]))
                    
                    dumselect = (iftime >= boundary_left[s] - 1e-5) & (iftime <= boundary_right[s] + 1e-5)
                    
                    selected_data = ifdata.loc[dumselect]
                    
                    time, sourcearr, pang1, pang2, ant1, ant2, rlreal, rlimag, lrreal, lrimag, rlsigma, lrsigma, rramp, rrphas, llamp, llphas, \
                    model_rlreal, model_rlimag, model_lrreal, model_lrimag, model_rlamp, model_rlphas, model_lramp, model_lrphas = \
                        np.array(selected_data["time"]), np.array(selected_data["source"]), np.array(selected_data["pang1"]), np.array(selected_data["pang2"]), np.array(selected_data["ant1"]), np.array(selected_data["ant2"]), \
                        np.array(selected_data["rlreal"]), np.array(selected_data["rlimag"]), np.array(selected_data["lrreal"]), np.array(selected_data["lrimag"]), np.array(selected_data["rlsigma"]), np.array(selected_data["lrsigma"]), \
                        np.array(selected_data["rramp"]), np.array(selected_data["rrphas"]), np.array(selected_data["llamp"]), np.array(selected_data["llphas"]), \
                        np.array(selected_data["model_rlreal"]), np.array(selected_data["model_rlimag"]), np.array(selected_data["model_lrreal"]), np.array(selected_data["model_lrimag"]), \
                        np.array(selected_data["model_rlamp"]), np.array(selected_data["model_rlphas"]), np.array(selected_data["model_lramp"]), np.array(selected_data["model_lrphas"])
                    
                    
                    init = [0.] * 4 * nant
    
                    
                    ant1_inputy = np.concatenate([ant1, ant1, ant1, ant1])
                    ant2_inputy = np.concatenate([ant2, ant2, ant2, ant2])
                    
                    
                    inputz = (nant, pang1, pang2, ant1, ant2, rramp, rrphas, llamp, llphas, model_rlreal, model_rlimag, model_rlamp, model_rlphas, model_lrreal, model_lrimag, model_lramp, model_lrphas)
                    
                    model_inputy = ps.pol_deq_inputz(inputz, *init)
                    
                    inputy = np.concatenate([lrreal, lrimag, rlreal, rlimag])
                    inputsigma = np.concatenate([lrsigma, lrsigma, rlsigma, rlsigma])
                    
                    delta_inputy = np.abs((inputy - model_inputy) / inputsigma) ** 2
                    
                    
                    orig_chisq_dev.append(np.sum(delta_inputy))
                    orig_chisq_n.append(float(len(inputy)))
                    
                    
                    delta_ant = []
                    
                    for i in range(len(self.antname)):
                        delta_ant.append(np.sum(delta_inputy[(ant1_inputy == i) | (ant2_inputy == i)]) / float(len(delta_inputy)))
                        # delta_ant.append(np.median(delta_inputy[(ant1_inputy == i) | (ant2_inputy == i)]))
                    
                    for i in range(len(self.antname)):
                        if np.isnan(delta_ant[i]):
                            delta_ant[i] = 0.
                        
                    
                    fliparr = []
                    for i in range(len(delta_ant)):
                        fliparr.append(np.argsort(delta_ant)[len(delta_ant)-1-i])
                    fliparr = np.array(fliparr)
                    
                    sort_delta_ant = self.antname[fliparr][0:self.timecal_freepar]
                    
                    fixdr = {}
                    fixdl = {}
                    for i in range(len(self.antname)):
                        if (self.antname[i] not in sort_delta_ant):
                            fixdr[self.antname[i]] = 0j
                            fixdl[self.antname[i]] = 0j
                    

                    fitresults, fitdterms, fitchisq = ps.pol_gpcal(self.direc, self.outputname, selected_data, self.calsour, self.antname, self.logger, printmessage = False, fixdterm = True, fixdr = fixdr, fixdl = fixdl, \
                                                         Dbound = self.Dbound, Pbound = self.Pbound, manualweight = self.manualweight, weightfactors = self.weightfactors)
                   
                    dum_ant, dum_DRArr, dum_DLArr = fitdterms
                    dum_chisq_num, dum_chisq_den, dum_chisq = fitchisq
                    
                    new_chisq_dev.append(dum_chisq_num)
                    new_chisq_n.append(dum_chisq_den)
                   
                    for m in range(len(dum_ant)):
                        timecal_ant.append(dum_ant[m])
                        timecal_IF.append(ifn+1)
                        timecal_scan.append(s+1)
                        timecal_scantime.append(boundary_time[s])
                        timecal_scansource.append(boundary_source[s])
                        timecal_DRArr.append(dum_DRArr[m])
                        timecal_DLArr.append(dum_DLArr[m])            
                
                
                if(tditer == 0):
                    self.timecal_chisq.loc["timecal_iter0", "IF"+str(ifn+1)] = np.sum(orig_chisq_dev) / np.sum(orig_chisq_n)
                
                self.timecal_chisq.loc["timecal_iter{:}".format(tditer+1), "IF"+str(ifn+1)] = np.sum(new_chisq_dev) / np.sum(new_chisq_n)
                        
                self.logger.info('\nThe reduced chi-square before and after the time-dependent D-term correction is {:5.3f} and {:5.3f}, respectively.'.\
                                 format(np.sum(orig_chisq_dev) / np.sum(orig_chisq_n), np.sum(new_chisq_dev) / np.sum(new_chisq_n)))            
                
    
            timecal_ant = np.array(timecal_ant)
            timecal_IF = np.array(timecal_IF)
            timecal_scan = np.array(timecal_scan)
            timecal_scantime = np.array(timecal_scantime)
            timecal_scansource = np.array(timecal_scansource)
            timecal_DRArr = np.array(timecal_DRArr)
            timecal_DLArr = np.array(timecal_DLArr)
            
            timecal_DRArr[np.abs(timecal_DRArr) < 2e-8] = 0j
            timecal_DLArr[np.abs(timecal_DLArr) < 2e-8] = 0j
            
            df = pd.DataFrame(timecal_ant.transpose())
            df['antennas'] = np.array(timecal_ant)
            df['IF'] = np.array(timecal_IF)
            df['scan'] = np.array(timecal_scan)
            df['scantime'] = np.array(timecal_scantime)
            df['scansource'] = np.array(timecal_scansource)
            df['DRArr'] = np.array(timecal_DRArr)
            df['DLArr'] = np.array(timecal_DLArr)
            del df[0]
            
            df.to_csv(self.direc+'gpcal/'+self.outputname+'timecal.iter{:}.dterm.csv'.format(tditer+1), sep = "\t")
                    
            dtermread = self.direc+'gpcal/'+self.outputname+'timecal.iter{:}.dterm.csv'.format(tditer+1)
            
            
            for l in range(len(self.calsour)):
                dataout = self.direc + self.outputname + 'timecal.iter{:}.'.format(tditer + 1) + self.calsour[l] + '.uvf'

                if(tditer == 0):    
                    if self.freqavg:
                        uvfname = self.direc + self.timecalname + self.calsour[l] + '.IFavg.uvf'
                    else:
                        uvfname = self.direc + self.timecalname + self.calsour[l] + '.uvf'
                else:
                    uvfname = self.direc + self.outputname + 'timecal.iter{:}.'.format(tditer) + self.calsour[l] + '.uvf'
                
                self.dtermcorrect(self.calsour[l], uvfname, dataout, dtermread, boundary_left, boundary_right)
                
        
        self.dtermcombine(self.direc, self.outputname, self.timecal_iter)
                
        dtermread = self.direc+'gpcal/'+self.outputname+'timecal.final.dterm.csv'
    
        
        self.tdplot(dtermread, self.direc+'gpcal/'+self.outputname+'timecal.final.dterm')
        
        for l in range(len(self.source)):
            if self.freqavg:
                uvfname = self.direc + self.timecalname + self.source[l] + '.IFavg.uvf'
            else:
                uvfname = self.direc + self.timecalname + self.source[l] + '.uvf'
            
            dataout = self.direc + self.outputname + 'timecal.final.' + self.source[l] + '.uvf'
            
            self.dtermcorrect(self.source[l], uvfname, dataout, dtermread, boundary_left, boundary_right)
        

    
    def dtermcombine(self, direc, outputname, tdcaliter):
        
        timecal_ant, timecal_IF, timecal_scan, timecal_scantime, timecal_scansource, timecal_DRArr, timecal_DLArr = [], [], [], [], [], [], []
        for tditer in range(tdcaliter):
            read = pd.read_csv(direc + 'gpcal/' + outputname + 'timecal.iter{:}.dterm.csv'.format(tditer+1), header = 0, skiprows=0, delimiter = '\t', index_col = 0)
            
            timecal_ant = timecal_ant + list(read['antennas'])
            timecal_IF = timecal_IF + list(read['IF'])
            timecal_scan = timecal_scan + list(read['scan'])
            timecal_scantime = timecal_scantime + list(read['scantime'])
            timecal_scansource = timecal_scansource + list(read['scansource'])
            timecal_DRArr = timecal_DRArr + list(read['DRArr'])
            timecal_DLArr = timecal_DLArr + list(read['DLArr'])
        
        timecal_ant, timecal_IF, timecal_scan, timecal_scantime, timecal_scansource, timecal_DRArr, timecal_DLArr = \
            np.array(timecal_ant), np.array(timecal_IF), np.array(timecal_scan), np.array(timecal_scantime), np.array(timecal_scansource), np.array(timecal_DRArr).astype('complex128'), np.array(timecal_DLArr).astype('complex128')
        
        new_timecal_ant = np.copy(timecal_ant)
        new_timecal_IF = np.copy(timecal_IF)
        new_timecal_scan = np.copy(timecal_scan)
        new_timecal_scantime = np.copy(timecal_scantime)
        new_timecal_scansource = np.copy(timecal_scansource)
        new_timecal_DRArr = np.copy(timecal_DRArr)
        new_timecal_DLArr = np.copy(timecal_DLArr)
        
        for j in range(len(np.unique(timecal_IF))):
            for i in range(len(np.unique(timecal_scan))):
                dumselect = (timecal_IF == np.unique(timecal_IF)[j]) & (timecal_scan == np.unique(timecal_scan)[i])
                dumant = timecal_ant[dumselect]
                dum_DRArr = timecal_DRArr[dumselect]
                dum_DLArr = timecal_DLArr[dumselect]
                for k in range(len(np.unique(dumant))):
                    dumdumselect = (dumant == np.unique(dumant)[k])
                    if(np.sum(dumdumselect) > 1):
                        index = np.where((new_timecal_ant == np.unique(dumant)[k]) & (new_timecal_IF == np.unique(timecal_IF)[j]) & (new_timecal_scan == np.unique(timecal_scan)[i]))[0]
                        
                        new_timecal_ant = np.delete(new_timecal_ant, index)
                        new_timecal_IF = np.delete(new_timecal_IF, index)
                        new_timecal_scan = np.delete(new_timecal_scan, index)
                        new_timecal_scantime = np.delete(new_timecal_scantime, index)
                        new_timecal_scansource = np.delete(new_timecal_scansource, index)
                        new_timecal_DRArr = np.delete(new_timecal_DRArr, index)
                        new_timecal_DLArr = np.delete(new_timecal_DLArr, index)
                        
                        new_DRArr = np.sum(dum_DRArr[dumdumselect])
                        new_DLArr = np.sum(dum_DLArr[dumdumselect])
                        
                        new_timecal_ant = np.append(new_timecal_ant, np.unique(dumant)[k])
                        new_timecal_IF = np.append(new_timecal_IF, np.unique(timecal_IF)[j])
                        new_timecal_scan = np.append(new_timecal_scan, np.unique(timecal_scan)[i])
                        new_timecal_scantime = np.append(new_timecal_scantime, np.unique(timecal_scantime)[i])
                        new_timecal_scansource = np.append(new_timecal_scansource, timecal_scansource[np.unique(timecal_scan, return_index = True)[1]][i])
                        new_timecal_DRArr = np.append(new_timecal_DRArr, new_DRArr)
                        new_timecal_DLArr = np.append(new_timecal_DLArr, new_DLArr)
                                            
        
        argsort = np.argsort(new_timecal_scan)
        
        new_timecal_ant, new_timecal_IF, new_timecal_scan, new_timecal_scantime, new_timecal_scansource, new_timecal_DRArr, new_timecal_DLArr = \
            new_timecal_ant[argsort], new_timecal_IF[argsort], new_timecal_scan[argsort], new_timecal_scantime[argsort], new_timecal_scansource[argsort], new_timecal_DRArr[argsort], new_timecal_DLArr[argsort]
        
        
        df = pd.DataFrame(new_timecal_ant.transpose())
        df['antennas'] = np.array(new_timecal_ant)
        df['IF'] = np.array(new_timecal_IF)
        df['scan'] = np.array(new_timecal_scan)
        df['scantime'] = np.array(new_timecal_scantime)
        df['scansource'] = np.array(new_timecal_scansource)
        df['DRArr'] = np.array(new_timecal_DRArr)
        df['DLArr'] = np.array(new_timecal_DLArr)
        del df[0]
        
        df.to_csv(direc + 'gpcal/' + outputname + 'timecal.final.dterm.csv', sep = "\t")


    def dtermcorrect(self, source, datain, dataout, read, boundary_left, boundary_right):
        """
        Apply the best-fit D-terms estimated by using the similarity assumption to UVFITS data.
        
        Args:
            source (list): a list of the sources for which D-terms will be corrected.
            DRArr (list): a list of the best-fit RCP D-terms.
            DLArr (list): a list of the best-fit LCP D-terms.
        """
        
        self.logger.info('\nApplying the estimated time-dependent D-Terms to {:s} ...'.format(source))
        
        
        dtermread = pd.read_csv(read, header = 0, skiprows=0, delimiter = '\t', index_col = 0)
            
        timecal_ant = np.array(dtermread['antennas'])
        timecal_IF = np.array(dtermread['IF'])
        timecal_scantime = np.array(dtermread['scantime'])
        timecal_scansource = np.array(dtermread['scansource'])
        dum_DRArr = np.array(dtermread['DRArr'])
        dum_DLArr = np.array(dtermread['DLArr'])
        
        timecal_DRArr = np.array([complex(it) for it in dum_DRArr])
        timecal_DLArr = np.array([complex(it) for it in dum_DLArr])
        
        inname = 'timecal'
        
        au.runfitld(inname, 'EDIT', datain)
            
        data = WAIPSUVData(inname, 'EDIT', 1, 1)
        
        year, month, day = oh.get_obsdate(data)
        
        obsra = data.header.crval[4]
        obsdec = data.header.crval[5]
        
        self.antname, self.antx, self.anty, self.antz, self.antmount, self.f_par, self.f_el, self.phi_off = oh.get_antcoord(data)
        
        self.ifnum, self.freq = oh.get_freqinfo(data)
        
        self.lonarr, self.latarr, self.heightarr = oh.coord(self.antname, self.antx, self.anty, self.antz)

                    
        for ifn in range(self.ifnum):
            
            dumu, dumv, ifarr, time, sourcearr, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight = \
                [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            
            dum = 0
            
            
            for visibility in data:
                dumu.append(visibility.uvw[0])
                dumv.append(visibility.uvw[1])
                ifarr.append(ifn+1)
                time.append(visibility.time)
                sourcearr.append(source)
                rrreal.append(visibility.visibility[ifn,0,0,0])
                rrimag.append(visibility.visibility[ifn,0,0,1])
                rrweight.append(visibility.visibility[ifn,0,0,2])
                llreal.append(visibility.visibility[ifn,0,1,0])
                llimag.append(visibility.visibility[ifn,0,1,1])
                llweight.append(visibility.visibility[ifn,0,1,2])
                rlreal.append(visibility.visibility[ifn,0,2,0])
                rlimag.append(visibility.visibility[ifn,0,2,1])
                rlweight.append(visibility.visibility[ifn,0,2,2])
                lrreal.append(visibility.visibility[ifn,0,3,0])
                lrimag.append(visibility.visibility[ifn,0,3,1])
                lrweight.append(visibility.visibility[ifn,0,3,2])
                ant1.append(visibility.baseline[0])
                ant2.append(visibility.baseline[1])
                
                dum += 1

            # Convert the lists to numpy arrays.
            dumu, dumv, ifarr, time, sourcearr, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight = \
                np.array(dumu), np.array(dumv), np.array(ifarr), np.array(time), np.array(sourcearr), np.array(ant1), np.array(ant2), np.array(rrreal), np.array(rrimag), np.array(rrweight), np.array(llreal), np.array(llimag), np.array(llweight), \
                np.array(rlreal), np.array(rlimag), np.array(rlweight), np.array(lrreal), np.array(lrimag), np.array(lrweight)
            
            if(np.sum(rrweight > 0.) == 0):
                continue
            
            time = time * 24.
            ant1 = ant1 - 1
            ant2 = ant2 - 1
            
            dumtime = np.copy(time)
                    
            longarr1, latarr1, f_el1, f_par1, phi_off1 = oh.coordarr(self.lonarr, self.latarr, self.f_el, self.f_par, self.phi_off, ant1)
            longarr2, latarr2, f_el2, f_par2, phi_off2 = oh.coordarr(self.lonarr, self.latarr, self.f_el, self.f_par, self.phi_off, ant2)
            
            yeararr, montharr, dayarr, raarr, decarr = oh.calendar(sourcearr, [source], [year], [month], [day], [obsra], [obsdec])
                        
            for i in range(10):
                dayarr[dumtime>=24.] += 1 # Version 1.1!
                dumtime[dumtime>=24.] -= 24. # Version 1.1!
                
            pang1 = oh.get_parang(yeararr, montharr, dayarr, dumtime, raarr, decarr, longarr1, latarr1, f_el1, f_par1, phi_off1) # Version 1.1!
            pang2 = oh.get_parang(yeararr, montharr, dayarr, dumtime, raarr, decarr, longarr2, latarr2, f_el2, f_par2, phi_off2)
            
            mat_Di, mat_Dj = [], []
            Vmat = []
            true_mat = []
            
            for i in range(len(time)):
                Vmat.append(np.array([[rrreal[i] + 1j * rrimag[i], rlreal[i] + 1j * rlimag[i]], [lrreal[i] + 1j * lrimag[i], llreal[i] + 1j * llimag[i]]]))
                
                if (source in self.calsour):
                    select1 = (timecal_IF == ifn+1) & (timecal_ant == self.antname[ant1[i]]) & (timecal_scansource == source)
                    select2 = (timecal_IF == ifn+1) & (timecal_ant == self.antname[ant2[i]]) & (timecal_scansource == source)
                    
                    dum_D_iR = timecal_DRArr[select1]
                    dum_D_iL = timecal_DLArr[select1]
                    dum_D_jR = timecal_DRArr[select2]
                    dum_D_jL = timecal_DLArr[select2]
                                    
                    dumt = timecal_scantime[select1]
                    tdev = np.abs(dumt - time[i])
                    
                    index = (tdev == np.min(tdev)) 
                    
                    mat_Di.append(np.array([[1., dum_D_iR[index][0] * np.exp(2j * pang1[i])], [dum_D_iL[index][0] * np.exp(-2j * pang1[i]), 1.]]))
                    mat_Dj.append(np.array([[1., dum_D_jL[index][0].conjugate() * np.exp(2j * pang2[i])], [dum_D_jR[index][0].conjugate() * np.exp(-2j * pang2[i]), 1.]]))
                    
                else:
                    
                    select1 = (timecal_IF == ifn+1) & (timecal_ant == self.antname[ant1[i]])
                    select2 = (timecal_IF == ifn+1) & (timecal_ant == self.antname[ant2[i]])
                    
                    dum_D_iR = timecal_DRArr[select1]
                    dum_D_iL = timecal_DLArr[select1]
                    dum_D_jR = timecal_DRArr[select2]
                    dum_D_jL = timecal_DLArr[select2]
                    
                    dumt = timecal_scantime[select1]
                    
                    for j in range(len(dumt)-1):
                        if(time[i] >= dumt[j]) & (time[i] < dumt[j+1]):
                            interpol_D_iR = dum_D_iR[j] + (dum_D_iR[j+1] - dum_D_iR[j]) * (time[i] - dumt[j]) / (dumt[j+1] - dumt[j])
                            interpol_D_iL = dum_D_iL[j] + (dum_D_iL[j+1] - dum_D_iL[j]) * (time[i] - dumt[j]) / (dumt[j+1] - dumt[j])
                            interpol_D_jR = dum_D_jR[j] + (dum_D_jR[j+1] - dum_D_jR[j]) * (time[i] - dumt[j]) / (dumt[j+1] - dumt[j])
                            interpol_D_jL = dum_D_jL[j] + (dum_D_jL[j+1] - dum_D_jL[j]) * (time[i] - dumt[j]) / (dumt[j+1] - dumt[j])
                        
                    if(time[i] <= np.min(dumt)):
                        interpol_D_iR = dum_D_iR[np.where(dumt == np.min(dumt))[0][0]]
                        interpol_D_iL = dum_D_iL[np.where(dumt == np.min(dumt))[0][0]]
                        interpol_D_jR = dum_D_jR[np.where(dumt == np.min(dumt))[0][0]]
                        interpol_D_jL = dum_D_jL[np.where(dumt == np.min(dumt))[0][0]]
                    
                    if(time[i] >= np.max(dumt)):
                        interpol_D_iR = dum_D_iR[np.where(dumt == np.max(dumt))[0][0]]
                        interpol_D_iL = dum_D_iL[np.where(dumt == np.max(dumt))[0][0]]
                        interpol_D_jR = dum_D_jR[np.where(dumt == np.max(dumt))[0][0]]
                        interpol_D_jL = dum_D_jL[np.where(dumt == np.max(dumt))[0][0]]                
                
                    mat_Di.append(np.array([[1., interpol_D_iR * np.exp(2j * pang1[i])], [interpol_D_iL * np.exp(-2j * pang1[i]), 1.]]))
                    mat_Dj.append(np.array([[1., interpol_D_jL.conjugate() * np.exp(2j * pang2[i])], [interpol_D_jR.conjugate() * np.exp(-2j * pang2[i]), 1.]]))
                
            mat_Di, mat_Dj = np.array(mat_Di), np.array(mat_Dj)
            
            dum = 0
            
            
            true_mat = []
            
            for i in range(len(time)):
                true_mat.append(np.matmul(np.matmul(inv(mat_Di[i]), Vmat[i]), inv(mat_Dj[i])))
                
            data = WAIPSUVData(inname, 'EDIT', 1, 1)
            
            for vis in data:
                
                vis.visibility[ifn,0,0,0] = np.real(true_mat[dum][0,0])
                vis.visibility[ifn,0,0,1] = np.imag(true_mat[dum][0,0])
                vis.visibility[ifn,0,1,0] = np.real(true_mat[dum][1,1])
                vis.visibility[ifn,0,1,1] = np.imag(true_mat[dum][1,1])
                vis.visibility[ifn,0,2,0] = np.real(true_mat[dum][0,1])
                vis.visibility[ifn,0,2,1] = np.imag(true_mat[dum][0,1])
                vis.visibility[ifn,0,3,0] = np.real(true_mat[dum][1,0])
                vis.visibility[ifn,0,3,1] = np.imag(true_mat[dum][1,0])
                vis.update()
                vis.update()
                
                dum += 1
      
        for vis in data:
            vis.update()
                
        if path.exists(dataout):
            os.system('rm ' + dataout)
            
        au.runfittp(inname, 'EDIT', dataout)
        
        data = AIPSUVData(inname, 'EDIT', 1, 1)
        
        data.zap()
    

