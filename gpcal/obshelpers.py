#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:59:06 2021

@author: jpark
"""


import numpy as np

from astropy.coordinates import EarthLocation
import astropy.time as at
import datetime as dt


from AIPS import AIPS
from AIPSTask import AIPSTask
from AIPSData import AIPSUVData, AIPSImage
from Wizardry.AIPSData import AIPSUVData as WAIPSUVData

import aipsutil as au

from os import path

from IPython import embed


def uvprt(data, select):
    """
    Extract UV data from ParselTongue UVData.
    
    Args:
        data (ParselTongue UVData): an input ParselTongue UVData.
        select (str): type of the data that will be extracted.
    
    Returns:
        list(s) of the selected UV data.
    """        
    
    ifnum = data.header.naxis[3]
    
    dumu, dumv, ifarr, time, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    
    for ifn in range(ifnum):
        for visibility in data:
            if((visibility.visibility[ifn,0,0,2] > 0.) & (visibility.visibility[ifn,0,1,2] > 0.) & (visibility.visibility[ifn,0,2,2] > 0.) & (visibility.visibility[ifn,0,3,2] > 0.)):
                dumu.append(visibility.uvw[0])
                dumv.append(visibility.uvw[1])
                ifarr.append(ifn+1)
                time.append(visibility.time)
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
        
    if(np.sum(rrreal) == 0.):
        return
    
    selectarr = ["u", "v", "ifarr", "time", "ant1", "ant2", "rrreal", "rrimag", "rrweight", "llreal", "llimag", "llweight", "rlreal", "rlimag", "rlweight", "lrreal", "lrimag", "lrweight"]
    package = [dumu, dumv, ifarr, time, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight]
    
    for i in range(len(selectarr)):
        if(select == selectarr[i]): return package[i]
    
    if(select == "all"): return dumu, dumv, ifarr, time, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight
    


def pol_model_uvprt(data, ifn, select):
    """
    Extract UV data from ParselTongue UVData for instrumental polarization self-calibration.
    
    Args:
        data (ParselTongue UVData): an input ParselTongue UVData.
        ifn (int): selected IF number
        select (str): type of the data that will be extracted.
    
    Returns:
        list(s) of the selected UV data.
    """        
    real, imag = [], []

#        (visibility.visibility[ifn,0,0,2] == np.nan) | (visibility.visibility[ifn,0,1,2] == np.nan) | (visibility.visibility[ifn,0,2,2] == np.nan) | (visibility.visibility[ifn,0,3,2] == np.nan))
    for visibility in data:
        if((visibility.visibility[ifn,0,0,2] > 0.) & (visibility.visibility[ifn,0,1,2] > 0.) & (visibility.visibility[ifn,0,2,2] > 0.) & (visibility.visibility[ifn,0,3,2] > 0.)):
            real.append(visibility.visibility[ifn,0,0,0])
            imag.append(visibility.visibility[ifn,0,0,1])
    
    if(np.sum(real) == 0.):
        return None, None
            
    selectarr = ["real", "imag"]
    package = [real, imag]
    
    for i in range(len(selectarr)):
        if(select == selectarr[i]): return package[i]
    
    if(select == "all"): return real, imag
    


#    def get_parang(self, time, ant, sourcearr, source, obsra, obsdec):
def get_parang(yeararr, montharr, dayarr, time, raarr, decarr, lonarr, latarr, f_el_arr, f_par_arr, phi_off_arr): # Version 1.1!
    """
    Calculate antenna field-rotation angles.
    
    Args:
        time (numpy.array): a numpy array of time in UTC of the visibilities.
        ant (numpy.array): a numpy array of antenna number of the visibilities.
        sourcearr (numpy.array): a numpy array of source of the visibilities.
        source (list): a list of calibrators.
        obsra (list): a list of calibrators' right ascension in units of degrees.
        obsdec (list): a list of calibrators' declination in units of degrees.
    
    Returns:
        a numpy of the field-rotation angles.
    """        
    
    
    latarr, decarr = np.radians(latarr), np.radians(decarr)

    hour = np.floor(time)
    minute = np.floor((time - hour) * 60.)
    second = (time - hour - minute / 60.) * 3600.
    
    hour = hour.astype('int')
    minute = minute.astype('int')
    second = second.astype('int')
    
    dumdatetime = [dt.datetime(a, b, c, d, e, f) for a, b, c, d, e, f in zip(yeararr, montharr, dayarr, hour, minute, second)]

    dumt = [] 

    for dum in dumdatetime:
        dumt.append("{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:f}".format(dum.year, dum.month, dum.day, dum.hour, dum.minute, dum.second + dum.microsecond * 1e-6))

    dumt = at.Time(dumt)
    gst = dumt.sidereal_time('mean','greenwich').hour

    # Obtain field-rotation angles using the known equations.
    hangle = np.radians(gst * 15. + lonarr - raarr)        
    parang = np.arctan2((np.sin(hangle) * np.cos(latarr)), (np.sin(latarr) * np.cos(decarr) - np.cos(latarr) * np.sin(decarr) * np.cos(hangle)))        
    altitude = np.arcsin(np.sin(decarr) * np.sin(latarr) + np.cos(decarr) * np.cos(latarr) * np.cos(hangle))
    pang = f_el_arr * altitude + f_par_arr * parang + phi_off_arr
    
    
    return pang
    

# #    def get_parang(self, time, ant, sourcearr, source, obsra, obsdec):
# def get_parang(time, ant, sourcearr, year, month, day, source, obsra, obsdec, longarr, latiarr, f_el, f_par, phi_off): # Version 1.1!
#     """
#     Calculate antenna field-rotation angles.
    
#     Args:
#         time (numpy.array): a numpy array of time in UTC of the visibilities.
#         ant (numpy.array): a numpy array of antenna number of the visibilities.
#         sourcearr (numpy.array): a numpy array of source of the visibilities.
#         source (list): a list of calibrators.
#         obsra (list): a list of calibrators' right ascension in units of degrees.
#         obsdec (list): a list of calibrators' declination in units of degrees.
    
#     Returns:
#         a numpy of the field-rotation angles.
#     """        
    
#     nant = np.max(ant) + 1
#     num = len(time)
    
#     lonarr, latarr, raarr, decarr, elarr, pararr, phiarr, yeararr, montharr, dayarr = \
#         np.zeros(num), np.zeros(num), np.zeros(num), np.zeros(num), np.zeros(num), np.zeros(num), np.zeros(num), np.zeros(num), np.zeros(num), np.zeros(num) # Version 1.1!
        
#     # Produce numpy arrays for antenna longitudes, latitudes, and the coefficients of the field-rotation angle equations.
#     for m in range(nant):
#         lonarr[(ant == m)] = longarr[m]
#         latarr[(ant == m)] = latiarr[m]
#         elarr[(ant == m)] = f_el[m]
#         pararr[(ant == m)] = f_par[m]
#         phiarr[(ant == m)] = phi_off[m]
    
#     # Produce numpy arrays for sources RA and Dec.
#     for l in range(len(source)):
#         raarr[sourcearr == source[l]] = obsra[l]
#         decarr[sourcearr == source[l]] = obsdec[l]
#         yeararr[sourcearr == source[l]] = year[l]
#         montharr[sourcearr == source[l]] = month[l]
#         dayarr[sourcearr == source[l]] = day[l]
        
#     latarr, decarr = np.radians(latarr), np.radians(decarr)

    
#     hour = np.floor(time)
#     minute = np.floor((time - hour) * 60.)
#     second = (time - hour - minute / 60.) * 3600.
    
#     for i in range(100):
#         dayarr[hour>=24.] += 1. # Version 1.1!
#         hour[hour>=24.] -= 24. # Version 1.1!
        
#     # Convert UTC to GST using astropy Time.
# #        dumt = at.Time(["{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:f}".format(self.year, self.month, self.day+int(dt), int(hr), int(mn), sec) for dt, hr, mn, sec in zip(date, hour, minute, second)])
    
#     dumt = at.Time(["{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:f}".format(int(yr), int(mo), int(dt), int(hr), int(mn), sec) for yr, mo, dt, hr, mn, sec in zip(yeararr, montharr, dayarr, hour, minute, second)]) # Version 1.1!
#     gst = dumt.sidereal_time('mean','greenwich').hour
    
    
#     # Obtain field-rotation angles using the known equations.
#     hangle = np.radians(gst * 15. + lonarr - raarr)        
#     parang = np.arctan2((np.sin(hangle) * np.cos(latarr)), (np.sin(latarr) * np.cos(decarr) - np.cos(latarr) * np.sin(decarr) * np.cos(hangle)))        
#     altitude = np.arcsin(np.sin(decarr) * np.sin(latarr) + np.cos(decarr) * np.cos(latarr) * np.cos(hangle))
#     pang = elarr * altitude + pararr * parang + phiarr
        
    
#     return pang
    


def coord(antname, antx, anty, antz):
    """
    Convert antenna positions from Cartesian to spherical coordinates using astropy.
    
    Returns:
        lists of antenna longitudes, latitudes, and heights.
    """      
    lonarr = []
    latarr = []
    heightarr = []
    
    for i in range(len(antname)):
        lonarr.append(EarthLocation.from_geocentric(antx[i], anty[i], antz[i], unit = 'm').to_geodetic()[0].value)
        latarr.append(EarthLocation.from_geocentric(antx[i], anty[i], antz[i], unit = 'm').to_geodetic()[1].value)
        heightarr.append(EarthLocation.from_geocentric(antx[i], anty[i], antz[i], unit = 'm').to_geodetic()[2].value)
        
    
    return lonarr, latarr, heightarr


def calendar(sourcearr, calsour, year, month, day, obsra, obsdec):
    dnum = len(sourcearr)
    yeararr, montharr, dayarr, raarr, decarr = np.zeros(dnum, dtype = 'int'), np.zeros(dnum, dtype = 'int'), np.zeros(dnum, dtype = 'int'), np.zeros(dnum, dtype = 'float'), np.zeros(dnum, dtype = 'float')
    for l in range(len(calsour)):
        yeararr[sourcearr == calsour[l]] = year[l]
        montharr[sourcearr == calsour[l]] = month[l]
        dayarr[sourcearr == calsour[l]] = day[l]
        raarr[sourcearr == calsour[l]] = obsra[l]
        decarr[sourcearr == calsour[l]] = obsdec[l]
    
    return yeararr, montharr, dayarr, raarr, decarr


def coordarr(longi, lati, f_el, f_par, phi_off, antarr):
    longi = np.array(longi)
    lati = np.array(lati)
    f_el = np.array(f_el)
    f_par = np.array(f_par)
    phi_off = np.array(phi_off)
            
    longarr = longi[antarr]
    latarr = lati[antarr]
    f_el = f_el[antarr]
    f_par = f_par[antarr]
    phi_off = phi_off[antarr]
    
    return longarr, latarr, f_el, f_par, phi_off


def basic_info(source, direc, dataname):
    obsra, obsdec, year, month, day = [], [], [], [], []
    
    for l in range(len(source)):
        inname = str(source[l])
        
        data = AIPSUVData(inname, 'EDIT', 1, 1)
        if(data.exists() == True):
            data.clrstat()
            data.zap()
        
        # Load UVFITS files.
        au.runfitld(inname, 'EDIT', direc + dataname + source[l] + '.uvf')
        
        data = AIPSUVData(inname, 'EDIT', 1, 1)
        
        dum_obsra, dum_obsdec = get_obscoord(data)
        
        # Extract source coordinates from the header.
        obsra.append(dum_obsra)
        obsdec.append(dum_obsdec)

        
        dumyear, dummonth, dumday = get_obsdate(data)
        
        year.append(dumyear)
        month.append(dummonth)
        day.append(dumday)
        
        # Extract antenna, frequency, mount information, etc, from the header.
        if(l == 0):
            
            antname, antx, anty, antz, antmount, f_par, f_el, phi_off = get_antcoord(data)
            
            ifnum, freq = get_freqinfo(data)
        
        data.zap()
        
    info = {"obsra": obsra, "obsdec": obsdec, "year": year, "month": month, "day": day, "antname": antname, "antx": antx, "anty": anty, "antz": antz, "antmount": antmount, "ifnum": ifnum, "freq": freq, \
            "f_par": f_par, "f_el": f_el, "phi_off": phi_off}
    
    return info



def get_obsdate(data):
    
    obsdate = data.header.date_obs
    year = int(obsdate[0:4])
    month = int(obsdate[5:7])
    day = int(obsdate[8:10])
    
    return year, month, day


def get_obscoord(data):
    
    obsra = data.header.crval[4]
    obsdec = data.header.crval[5]
    
    return obsra, obsdec


def get_antcoord(data):
    
    antname = []
    antx = []
    anty = []
    antz = []
    antmount = []
    
    antable = data.table('AN', 1)
    for row in antable:
        antname.append(row.anname.replace(' ', ''))
        antx.append(row.stabxyz[0])
        anty.append(row.stabxyz[1])
        antz.append(row.stabxyz[2])
        antmount.append(row.mntsta)

    f_par = []
    f_el = []
    phi_off = []
    
    for st in range(len(antname)):
        f_par.append(1.)
        if(antmount[st] == 0) | (antmount[st] == 4) | (antmount[st] == 5):
            f_el.append(0.)
        if(antmount[st] == 4):
            f_el.append(1.)
        if(antmount[st] == 5):
            f_el.append(-1.)
        phi_off.append(0.)
        
    return antname, antx, anty, antz, antmount, f_par, f_el, phi_off


def get_freqinfo(data):
        
    fqtable = data.table('FQ', 1)
    if(isinstance(fqtable[0].if_freq, float) == True):
        ifnum = 1
        IFfreq = [data.header.crval[2] / 1e9]
        freq = "{0:.3f}".format(IFfreq[0]) + ' GHz'
    else:
        ifnum = len(fqtable[0].if_freq)
        freq = [str((it + data.header.crval[2]) / 1e9) + ' GHz' for it in fqtable[0].if_freq]
    
    return ifnum, freq


def pd_modifier(data):
    
    # Make new columns for amplitudes, phases, and corresponding errors, etc.
    data.loc[:,"rrsigma"], data.loc[:,"llsigma"], data.loc[:,"rlsigma"], data.loc[:,"lrsigma"] = \
        1. / data.loc[:,"rrweight"] ** (0.5), 1. / data.loc[:,"llweight"], 1. / data.loc[:,"rlweight"] ** (0.5), 1. / data.loc[:,"lrweight"] ** (0.5)
    
    data.loc[:,"rramp"], data.loc[:,"llamp"], data.loc[:,"rlamp"], data.loc[:,"lramp"] = \
        np.absolute(data.loc[:,"rrreal"] + 1j*data.loc[:,"rrimag"]), np.absolute(data.loc[:,"llreal"] + 1j*data.loc[:,"llimag"]), \
        np.absolute(data.loc[:,"rlreal"] + 1j*data.loc[:,"rlimag"]), np.absolute(data.loc[:,"lrreal"] + 1j*data.loc[:,"lrimag"])
    
    data.loc[:,"rrphas"], data.loc[:,"llphas"], data.loc[:,"rlphas"], data.loc[:,"lrphas"] = \
        np.angle(data.loc[:,"rrreal"] + 1j*data.loc[:,"rrimag"]), np.angle(data.loc[:,"llreal"] + 1j*data.loc[:,"llimag"]), \
        np.angle(data.loc[:,"rlreal"] + 1j*data.loc[:,"rlimag"]), np.angle(data.loc[:,"lrreal"] + 1j*data.loc[:,"lrimag"])
    
    data.loc[:,"rramp_sigma"], data.loc[:,"llamp_sigma"], data.loc[:,"rlamp_sigma"], data.loc[:,"lramp_sigma"] = \
        data.loc[:,"rrsigma"], data.loc[:,"llsigma"], data.loc[:,"rlsigma"], data.loc[:,"lrsigma"]
    
    data.loc[:,"rrphas_sigma"], data.loc[:,"llphas_sigma"], data.loc[:,"rlphas_sigma"], data.loc[:,"lrphas_sigma"] = \
        data.loc[:,"rrsigma"] / np.abs(data.loc[:,"rrreal"] + 1j*data.loc[:,"rrimag"]), \
        data.loc[:,"llsigma"] / np.abs(data.loc[:,"llreal"] + 1j*data.loc[:,"llimag"]), \
        data.loc[:,"rlsigma"] / np.abs(data.loc[:,"rlreal"] + 1j*data.loc[:,"rlimag"]), \
        data.loc[:,"lrsigma"] / np.abs(data.loc[:,"lrreal"] + 1j*data.loc[:,"lrimag"])
        
        # phaserror(data.loc[:,"rrreal"] + 1j*data.loc[:,"rrimag"], data.loc[:,"rrsigma"] + 1j*data.loc[:,"rrsigma"]), \
        # phaserror(data.loc[:,"llreal"] + 1j*data.loc[:,"llimag"], data.loc[:,"llsigma"] + 1j*data.loc[:,"llsigma"]), \
        # phaserror(data.loc[:,"rlreal"] + 1j*data.loc[:,"rlimag"], data.loc[:,"rlsigma"] + 1j*data.loc[:,"rlsigma"]), \
        # phaserror(data.loc[:,"lrreal"] + 1j*data.loc[:,"lrimag"], data.loc[:,"lrsigma"] + 1j*data.loc[:,"lrsigma"])
        
    
    dumrl, dumlr = data.loc[:,"rlreal"] + 1j*data.loc[:,"rlimag"], data.loc[:,"lrreal"] + 1j*data.loc[:,"lrimag"]
    dumrlsigma, dumlrsigma = data.loc[:,"rlsigma"] + 1j*data.loc[:,"rlsigma"], data.loc[:,"lrsigma"] + 1j*data.loc[:,"lrsigma"]
    
    dumq, dumu = (dumrl + dumlr) / 2., -1j * (dumrl - dumlr) / 2.
    dumqsigma, dumusigma = np.sqrt(dumrlsigma**2 + dumlrsigma**2) / 2., np.sqrt(dumrlsigma**2 + dumlrsigma**2) / 2.
    
    data.loc[:,"qamp"], data.loc[:,"uamp"], data.loc[:,"qphas"], data.loc[:,"uphas"] = np.absolute(dumq), np.absolute(dumu), np.angle(dumq), np.angle(dumu)
    data.loc[:,"qphas_sigma"], data.loc[:,"uphas_sigma"] = np.real(dumqsigma) / np.abs(dumq), np.real(dumusigma) / np.abs(dumu)
    data.loc[:,"qsigma"], data.loc[:,"usigma"] = np.real(dumqsigma), np.real(dumusigma)

    data["IF"] = data["IF"].astype('int32')
    data["ant1"] = data["ant1"].astype('int32')
    data["ant2"] = data["ant2"].astype('int32')
    data["pang1"] = data["pang1"].astype('float64')
    data["pang2"] = data["pang2"].astype('float64')
    
    return data


def get_model(data, direc, dataname, calsour, polcal_unpol, ifnum, pol_IF_combine, outputname = None, selfcal = False):
    """
    Extract Stokes Q and U visibility models and append them to the pandas dataframe.
    
    """
    
    # self.logger.info('\nGetting source polarization models for {:d} sources for {:d} IFs...'.format(len(self.polcalsour), self.ifnum))
        
    # AIPS.userno = self.aips_userno
    
    # if self.aipslog:
    #     AIPS.log = open(self.logfile, 'a')
    # AIPSTask.msgkill = -1
            
    
    mod_qrealarr, mod_qimagarr, mod_urealarr, mod_uimagarr = [], [], [], []
    
    
    for l in range(len(calsour)):
    
        inname = str(calsour[l])
    
        calib = AIPSUVData(inname, 'EDIT', 1, 1)
        if(calib.exists() == True):
            calib.clrstat()
            calib.zap()
        
        if selfcal:
            au.runfitld(inname, 'EDIT', direc + dataname + calsour[l]+'.calib')
        else:
            au.runfitld(inname, 'EDIT', direc + dataname + calsour[l]+'.uvf')
    
        calib = AIPSUVData(inname, 'EDIT', 1, 1)
        
        if polcal_unpol[l]:
            dumdata = data
            dumsource = np.array(dumdata.loc[:,"source"])
            dumlen = np.sum(dumsource == calsour[l])
            mod_qrealarr = mod_qrealarr + [0.] * dumlen
            mod_qimagarr = mod_qimagarr + [0.] * dumlen
            mod_urealarr = mod_urealarr + [0.] * dumlen
            mod_uimagarr = mod_uimagarr + [0.] * dumlen
       
        else:
            for k in range(ifnum):
                if(np.sum(data.loc[:, "IF"] == k+1) == 0):
                    continue

                qmap = AIPSImage(inname, 'QMAP', 1, 1)
                if(qmap.exists() == True):
                    qmap.clrstat()
                    qmap.zap()

                if pol_IF_combine:
                    fitsname = direc+dataname+calsour[l]+'.allIF.q.fits'
                else:
                    fitsname = direc+dataname+calsour[l]+'.IF'+str(k+1)+'.q.fits'
                
                
                # if not path.exists(fitsname):
                #     dum = 0
                #     calib = WAIPSUVData(inname, 'EDIT', 1, 1)
                #     for visibility in calib:
                #         if((visibility.visibility[k,0,0,2] > 0) & (visibility.visibility[k,0,1,2] > 0) & (visibility.visibility[k,0,2,2] > 0) & (visibility.visibility[k,0,3,2] > 0)):
                #             dum += 1 
    
                #     mod_qrealarr = mod_qrealarr + [0.] * dum
                #     mod_qimagarr = mod_qimagarr + [0.] * dum
                    
                #     calib = AIPSUVData(inname, 'EDIT', 1, 1)
                if not path.exists(fitsname):
                    raise Exception("The requested {:} file does not exist!".format(fitsname))
                    
                else:
                    
                    au.runfitld(inname, 'QMAP', fitsname)
                    qmap = AIPSImage(inname, 'QMAP', 1, 1)     
         
                    uvsub = AIPSUVData(inname, 'UVSUB', 1, 1)
                    if(uvsub.exists() == True):
                        uvsub.clrstat()
                        uvsub.zap()

                    au.runuvsub(inname, 'EDIT', 'QMAP', 1, 1)
                    
                    moddata = WAIPSUVData(inname, 'UVSUB', 1, 1)
                    mod_qreal, mod_qimag = pol_model_uvprt(moddata, k, "all")

                    if(mod_qreal != None):
                        mod_qrealarr = mod_qrealarr + mod_qreal
                        mod_qimagarr = mod_qimagarr + mod_qimag

                    moddata.zap()
                    qmap.zap()
                
                
                
                umap = AIPSImage(inname, 'UMAP', 1, 1)
                if(umap.exists() == True):
                    umap.clrstat()
                    umap.zap()


                if pol_IF_combine:
                    fitsname = direc+dataname+calsour[l]+'.allIF.u.fits'
                else:
                    fitsname = direc+dataname+calsour[l]+'.IF'+str(k+1)+'.u.fits'
                
                
                if not path.exists(fitsname):
                    dum = 0
                    calib = WAIPSUVData(inname, 'EDIT', 1, 1)
                    for visibility in calib:
                        if((visibility.visibility[k,0,0,2] > 0) & (visibility.visibility[k,0,1,2] > 0) & (visibility.visibility[k,0,2,2] > 0) & (visibility.visibility[k,0,3,2] > 0)):
                            dum += 1 
    
                    mod_urealarr = mod_urealarr + [0.] * dum
                    mod_uimagarr = mod_uimagarr + [0.] * dum
                    
                    calib = AIPSUVData(inname, 'EDIT', 1, 1)
                
                else:
                    au.runfitld(inname, 'UMAP', fitsname)
                    umap = AIPSImage(inname, 'UMAP', 1, 1)     
         
                    uvsub = AIPSUVData(inname, 'UVSUB', 1, 1)
                    if(uvsub.exists() == True):
                        uvsub.clrstat()
                        uvsub.zap()

                    au.runuvsub(inname, 'EDIT', 'UMAP', 1, 1)
                    
                    moddata = WAIPSUVData(inname, 'UVSUB', 1, 1)
                    mod_ureal, mod_uimag = pol_model_uvprt(moddata, k, "all")

                    if(mod_ureal != None):
                        mod_urealarr = mod_urealarr + mod_ureal
                        mod_uimagarr = mod_uimagarr + mod_uimag

                    moddata.zap()
                    umap.zap()
                    

        calib.zap()

    
    mod_qreal, mod_qimag, mod_ureal, mod_uimag = np.array(mod_qrealarr), np.array(mod_qimagarr), np.array(mod_urealarr), np.array(mod_uimagarr)
    
    
    mod_q, mod_u = mod_qreal + 1j*mod_qimag, mod_ureal + 1j*mod_uimag
    mod_rlreal, mod_rlimag, mod_lrreal, mod_lrimag = np.real(mod_q + 1j*mod_u), np.imag(mod_q + 1j*mod_u), np.real(mod_q - 1j*mod_u), np.imag(mod_q - 1j*mod_u)
    mod_rlamp, mod_rlphas, mod_lramp, mod_lrphas = np.absolute(mod_rlreal + 1j*mod_rlimag), np.angle(mod_rlreal + 1j*mod_rlimag), np.absolute(mod_lrreal + 1j*mod_lrimag), np.angle(mod_lrreal + 1j*mod_lrimag)
    mod_qamp, mod_qphas, mod_uamp, mod_uphas = np.absolute(mod_q), np.angle(mod_q), np.absolute(mod_u), np.angle(mod_u)
    

    # Append the model visibilities to the existing pandas dataframe as new columns.
    data.loc[:,"model_rlreal"], data.loc[:,"model_rlimag"], data.loc[:,"model_lrreal"], data.loc[:,"model_lrimag"], \
    data.loc[:,"model_rlamp"], data.loc[:,"model_rlphas"], data.loc[:,"model_lramp"], data.loc[:,"model_lrphas"], \
    data.loc[:,"model_qamp"], data.loc[:,"model_qphas"], data.loc[:,"model_uamp"], data.loc[:,"model_uphas"] = \
    mod_rlreal, mod_rlimag, mod_lrreal, mod_lrimag, mod_rlamp, mod_rlphas, mod_lramp, mod_lrphas, mod_qamp, mod_qphas, mod_uamp, mod_uphas
    
    return data


def evpacal(datain, dataout, clcorprm, logger = None): # Version 1.1
    
    if (logger != None):
        logger.info('Correcting EVPAs... \n Input file: {:} \n Output file: {:}'.format(datain, dataout))
    
    pinal = AIPSUVData('EVPA', 'PINAL', 1, 1)
    if(pinal.exists() == True):
        pinal.zap()
        
    au.runfitld('EVPA', 'PINAL', datain)
    
    pinal = AIPSUVData('EVPA', 'PINAL', 1, 1)
    
    aipssource = pinal.header.object
    
    multi = AIPSUVData('EVPA', 'MULTI', 1, 1)
    if(multi.exists() == True):
        multi.zap()
        
    au.runmulti('EVPA', 'PINAL')
    au.runclcor('EVPA', 'MULTI', 1, 1, clcorprm)
    
    pang = AIPSUVData(aipssource, 'PANG', 1, 1)
    if(pang.exists() == True):
        pang.zap()
        
    au.runsplitpang('EVPA')
            
    au.runfittp(aipssource, 'PANG', dataout)
    
    pinal.zap()
    multi.zap()
    pang.zap()



def get_scans(time, sourcearr, tsep = 2. / 60.):
    argsort = np.argsort(time)
    dum_sourcearr = sourcearr[argsort]
    dumx = np.sort(time)
    
    boundary_left = [np.min(dumx)]
    boundary_right = []
    boundary_source = [dum_sourcearr[0]]
    
    for j in range(len(dumx)-1):
        if(dumx[j+1] - dumx[j]) > tsep:
            boundary_left.append(dumx[j+1])
            boundary_right.append(dumx[j])
            boundary_source.append(dum_sourcearr[j+1])
            continue
        if(dum_sourcearr[j+1] != dum_sourcearr[j]):
            boundary_left.append(dumx[j+1])
            boundary_right.append(dumx[j])
            boundary_source.append(dum_sourcearr[j+1])
    
    boundary_right.append(np.max(dumx))
    
    return boundary_left, boundary_right, boundary_source


def bin_data(time, data, boundary_left, boundary_right, error = None, avg_nat = False):
    
    binx, biny, binsigma = [], [], []
    for i in range(len(boundary_left)):
        select = (time >= boundary_left[i]) & (time < boundary_right[i])
        if(np.sum(select) == 0): continue
        binx.append((boundary_left[i] + boundary_right[i]) / 2.)
        if (error is not None) & (avg_nat == True):
            biny.append(np.average(data[select], weights = 1. / error[select] ** 2, returned = True)[0])
            binsigma.append(np.abs(1. / np.average(data[select], weights = 1. / error[select] ** 2, returned = True)[1] ** 0.5))
        else:
            biny.append(np.mean(data[select]))
            binsigma.append(np.std(data[select]) / np.sqrt(float(np.sum(select))))
    
    binx, biny, binsigma = np.array(binx), np.array(biny), np.array(binsigma)
        
    return binx, biny, binsigma
    
