#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:59:06 2021

@author: jpark
"""


import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import obshelpers as oh

from multiprocessing import Pool

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


def visualplot_run(args, nproc = 2): # Version 1.1
    """
    Perform imaging of Stokes Q and U in Difmap.
    
    Args:
        data (str): the name of the UVFITS file to be CLEANed.
        mask (str): the name of the CLEAN windows to be used.
        save (str): the name of the output Difmap save file.
        log (str): the name of the Difmap log file.
    """

    pool = Pool(processes = nproc)
    pool.map(visualplot3, args)
    pool.close()
    pool.join()


def visualplot2(args): # Version 1.1
    # print(args)
    visualplot3(*args)

def visualplot3(parm): # Version 1.1
    # print(len(parm))
    # print(type(len[6]))
    (color, source, ifn, antname1, antname2, day, time, qamp, qphas, qsigma, uamp, uphas, usigma, mod_qamp, mod_qphas, mod_uamp, mod_uphas, filename, filetype, allplots, title, scanavg, avg_nat, tsep) = parm
    visualplot(source, ifn, antname1, antname2, day, time, qamp, qphas, qsigma, uamp, uphas, usigma, mod_qamp, mod_qphas, mod_uamp, mod_uphas, filename, filetype, allplots = allplots, title = title, \
               scanavg = scanavg, avg_nat = avg_nat, tsep = tsep, color = color)
    

def visualplot(source, ifn, antname1, antname2, day, time, qamp, qphas, qsigma, uamp, uphas, usigma, mod_qamp, mod_qphas, mod_uamp, mod_uphas, filename, filetype, \
               allplots = None, title = None, scanavg = None, avg_nat = False, tsep = None, color = None):
    """
    Draw vplots.
    """        
        
    if(len(time) == 0):
        return
    
    if(color == None):
        color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf'] + \
                ['blue', 'red', 'orange', 'steelblue', 'green', 'slategrey', 'cyan', 'royalblue', 'purple', 'blueviolet', 'darkcyan', 'darkkhaki', 'magenta', 'navajowhite', 'darkred']
    
    
    orig_time = np.copy(time)
    
    time2 = np.copy(time)
    day2 = np.copy(day) - np.min(day)
    
    if(np.max(day2) > 0):
        for i in range(np.max(day2)):
            time2[day2 == (i+1)] += 24. * (i+1)
        
    
    qphas = np.degrees(qphas)
    uphas = np.degrees(uphas)
    mod_qphas = np.degrees(mod_qphas)
    mod_uphas = np.degrees(mod_uphas)
    
    if allplots != None:
        (mod_pol_qamp, mod_pol_qphas, mod_pol_uamp, mod_pol_uphas, mod_dterm_qamp, mod_dterm_qphas, mod_dterm_uamp, mod_dterm_uphas) = allplots
        mod_pol_qphas = np.degrees(mod_pol_qphas)
        mod_pol_uphas = np.degrees(mod_pol_uphas)
        mod_dterm_qphas = np.degrees(mod_dterm_qphas)
        mod_dterm_uphas = np.degrees(mod_dterm_uphas)
    

    if scanavg:
        boundary_left, boundary_right, boundary_source = oh.get_scans(time2, np.repeat(source, len(time2)), tsep = tsep)
        
        q = qamp * np.exp(1j * np.radians(qphas))
        u = uamp * np.exp(1j * np.radians(uphas))
        
        bin_time, bin_q, bin_qsigma = oh.bin_data(time2, q, boundary_left, boundary_right, error = qsigma, avg_nat = avg_nat)
        bin_time, bin_u, bin_usigma = oh.bin_data(time2, u, boundary_left, boundary_right, error = usigma, avg_nat = avg_nat)
                
        bin_qamp = np.abs(bin_q)
        bin_qphas = np.angle(bin_q)
        bin_uamp = np.abs(bin_u)
        bin_uphas = np.angle(bin_u)
                
        mod_q = mod_qamp * np.exp(1j * np.radians(mod_qphas))
        mod_u = mod_uamp * np.exp(1j * np.radians(mod_uphas))
        
        bin_time, bin_mod_q, bin_mod_qsigma = oh.bin_data(time2, mod_q, boundary_left, boundary_right)
        bin_time, bin_mod_u, bin_mod_usigma = oh.bin_data(time2, mod_u, boundary_left, boundary_right)
        
        bin_mod_qamp = np.abs(bin_mod_q)
        bin_mod_qphas = np.angle(bin_mod_q)
        bin_mod_uamp = np.abs(bin_mod_u)
        bin_mod_uphas = np.angle(bin_mod_u)
                
        time, qamp, qphas, qsigma, uamp, uphas, usigma, mod_qamp, mod_qphas, mod_uamp, mod_uphas = \
            bin_time, bin_qamp, np.degrees(bin_qphas), bin_qsigma, bin_uamp, np.degrees(bin_uphas), bin_usigma, bin_mod_qamp, np.degrees(bin_mod_qphas), bin_mod_uamp, np.degrees(bin_mod_uphas)
        
        if allplots != None:
            mod_pol_q = mod_pol_qamp * np.exp(1j * np.radians(mod_pol_qphas))
            mod_pol_u = mod_pol_uamp * np.exp(1j * np.radians(mod_pol_uphas))
            
            bin_time, bin_mod_pol_q, bin_mod_pol_qsigma = oh.bin_data(time2, mod_pol_q, boundary_left, boundary_right)
            bin_time, bin_mod_pol_u, bin_mod_pol_usigma = oh.bin_data(time2, mod_pol_u, boundary_left, boundary_right)
            
            bin_mod_pol_qamp = np.abs(bin_mod_pol_q)
            bin_mod_pol_qphas = np.angle(bin_mod_pol_q)
            bin_mod_pol_uamp = np.abs(bin_mod_pol_u)
            bin_mod_pol_uphas = np.angle(bin_mod_pol_u)
            
            
            mod_dterm_q = mod_dterm_qamp * np.exp(1j * np.radians(mod_dterm_qphas))
            mod_dterm_u = mod_dterm_uamp * np.exp(1j * np.radians(mod_dterm_uphas))
            
            bin_time, bin_mod_dterm_q, bin_mod_dterm_qsigma = oh.bin_data(time2, mod_dterm_q, boundary_left, boundary_right)
            bin_time, bin_mod_dterm_u, bin_mod_dterm_usigma = oh.bin_data(time2, mod_dterm_u, boundary_left, boundary_right)
            
            bin_mod_dterm_qamp = np.abs(bin_mod_dterm_q)
            bin_mod_dterm_qphas = np.angle(bin_mod_dterm_q)
            bin_mod_dterm_uamp = np.abs(bin_mod_dterm_u)
            bin_mod_dterm_uphas = np.angle(bin_mod_dterm_u)
            
            mod_pol_qamp, mod_pol_qphas, mod_pol_uamp, mod_pol_uphas, mod_dterm_qamp, mod_dterm_qphas, mod_dterm_uamp, mod_dterm_uphas = \
                bin_mod_pol_qamp, np.degrees(bin_mod_pol_qphas), bin_mod_pol_uamp, np.degrees(bin_mod_pol_uphas), bin_mod_dterm_qamp, np.degrees(bin_mod_dterm_qphas), bin_mod_dterm_uamp, np.degrees(bin_mod_dterm_uphas)
        
    
    figure, axes = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0}, figsize=(8, 8))

    for ax in axes.flat:
        ax.tick_params(length=6, width=2,which = 'major')
        ax.tick_params(length=4, width=1.5,which = 'minor')
          
    axes[0].set_xlim(np.min(time2) - (np.max(time2) - np.min(time2)) * 0.5, np.max(time2) + (np.max(time2) - np.min(time2)) * 0.5)
    axes[1].set_xlim(np.min(time2) - (np.max(time2) - np.min(time2)) * 0.5, np.max(time2) + (np.max(time2) - np.min(time2)) * 0.5)
    axes[1].set_ylim(-180, 180)
    
    axes[1].set(xlabel = 'Time (UT)')
    
    axes[0].set(ylabel = 'Amplitude (Jy)')
    axes[1].set(ylabel = 'Phase (deg)')
    
    if title != None:
        axes[0].set_title(title)

    axes[0].annotate('Stokes Q', xy=(1, 1), xycoords = 'axes fraction', xytext = (-25, -25), size = 24, textcoords='offset pixels', horizontalalignment='right', verticalalignment='top')
    
    axes[1].annotate('{:}-{:}'.format(antname1, antname2), xy=(0, 0), xycoords = 'axes fraction', xytext = (25, 25), size = 24, textcoords='offset pixels', horizontalalignment='left', verticalalignment='bottom')
    axes[1].annotate('IF {:d}'.format(ifn), xy = (0, 1), xycoords = 'axes fraction', xytext = (25, -25), size = 24, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
                
    axes[0].errorbar(time, qamp, qsigma, fmt = 'o', markerfacecolor = 'None', markeredgecolor = color, ecolor = color, label = source.upper(), zorder = 1)
    axes[1].errorbar(time, qphas, np.degrees(qsigma / qamp), fmt = 'o', markerfacecolor = 'None', markeredgecolor = color, ecolor = color, label = source.upper(), zorder = 1)


    leg1 = axes[0].legend(loc='upper left', fontsize = 18 - int(len(source)/2.), frameon=False, markerfirst=True, handlelength=1.0)
    
    dumx = np.array(time)
    
    argsort = np.argsort(dumx)
    
    whole, = axes[0].plot(dumx[argsort], mod_qamp[argsort], color = 'grey', lw = 1.0, zorder = 0)
    axes[1].plot(dumx[argsort], mod_qphas[argsort], color = 'grey', lw = 1.0, zorder = 0)

    if (allplots != None):
        
        axes[0].plot(time[argsort], mod_pol_qamp[argsort], lw = 1.0, zorder = 0, color = 'grey', linestyle = '--', dashes = (4, 1))
        axes[1].plot(time[argsort], mod_pol_qphas[argsort], lw = 1.0, zorder = 0, color = 'grey', linestyle = '--', dashes = (4, 1))
        
        axes[0].plot(time[argsort], mod_dterm_qamp[argsort], lw = 1.0, zorder = 0, color = 'grey', linestyle = ':', dashes = (1.5, 1))
        axes[1].plot(time[argsort], mod_dterm_qphas[argsort], lw = 1.0, zorder = 0, color = 'grey', linestyle = ':', dashes = (1.5, 1))
        
        spol, = axes[0].plot(np.nan, np.nan, color = 'black', lw = 1.0, zorder = 0, linestyle = '--', dashes = (4, 1))
        dterm, = axes[0].plot(np.nan, np.nan, color = 'black', lw = 1.0, zorder = 0, linestyle = ':', dashes = (1.5, 1))

        leg2 = axes[0].legend([spol, dterm, whole], ['source-pol terms', 'instrumental-pol terms', 'whole terms'], loc = 'lower left', frameon=False, fontsize = 13, handlelength=1.0)
        axes[0].add_artist(leg1)
        
        
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
                if(dumit == 0): dumxticklabel[it] = '{:02d}d/'.format(np.min(day) + 1) + dumxticklabel[it]
                dumit += 1
    
    axes[1].set_xticklabels(dumxticklabel)

    figure.savefig("{:}.{:}.baseline_{:}_{:}.Q.{:}".format(filename, source, antname1, antname2, filetype), bbox_inches = 'tight')
    
    plt.close('all')



    figure, axes = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0}, figsize=(8, 8))

    for ax in axes.flat:
        ax.tick_params(length=6, width=2,which = 'major')
        ax.tick_params(length=4, width=1.5,which = 'minor')
          
    axes[0].set_xlim(np.min(time2) - (np.max(time2) - np.min(time2)) * 0.5, np.max(time2) + (np.max(time2) - np.min(time2)) * 0.5)
    axes[1].set_xlim(np.min(time2) - (np.max(time2) - np.min(time2)) * 0.5, np.max(time2) + (np.max(time2) - np.min(time2)) * 0.5)
    axes[1].set_ylim(-180, 180)
    
    axes[1].set(xlabel = 'Time (UT)')
    
    axes[0].set(ylabel = 'Amplitude (Jy)')
    axes[1].set(ylabel = 'Phase (deg)')
    
    if title != None:
        axes[0].set_title(title)

    axes[0].annotate('Stokes U', xy=(1, 1), xycoords = 'axes fraction', xytext = (-25, -25), size = 24, textcoords='offset pixels', horizontalalignment='right', verticalalignment='top')
    
    axes[1].annotate('{:}-{:}'.format(antname1, antname2), xy=(0, 0), xycoords = 'axes fraction', xytext = (25, 25), size = 24, textcoords='offset pixels', horizontalalignment='left', verticalalignment='bottom')
    axes[1].annotate('IF {:d}'.format(ifn), xy = (0, 1), xycoords = 'axes fraction', xytext = (25, -25), size = 24, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
                
    axes[0].errorbar(time, uamp, usigma, fmt = 'o', markerfacecolor = 'None', markeredgecolor = color, ecolor = color, label = source.upper(), zorder = 1)
    axes[1].errorbar(time, uphas, np.degrees(usigma / uamp), fmt = 'o', markerfacecolor = 'None', markeredgecolor = color, ecolor = color, label = source.upper(), zorder = 1)


    leg1 = axes[0].legend(loc='upper left', fontsize = 18 - int(len(source)/2.), frameon=False, markerfirst=True, handlelength=1.0)
    
    dumx = np.array(time)
    
    argsort = np.argsort(dumx)
    
    whole, = axes[0].plot(dumx[argsort], mod_uamp[argsort], color = 'grey', lw = 1.0, zorder = 0)
    axes[1].plot(dumx[argsort], mod_uphas[argsort], color = 'grey', lw = 1.0, zorder = 0)

    if (allplots != None):
        
        axes[0].plot(time[argsort], mod_pol_uamp[argsort], lw = 1.0, zorder = 0, color = 'grey', linestyle = '--', dashes = (4, 1))
        axes[1].plot(time[argsort], mod_pol_uphas[argsort], lw = 1.0, zorder = 0, color = 'grey', linestyle = '--', dashes = (4, 1))
        
        axes[0].plot(time[argsort], mod_dterm_uamp[argsort], lw = 1.0, zorder = 0, color = 'grey', linestyle = ':', dashes = (1.5, 1))
        axes[1].plot(time[argsort], mod_dterm_uphas[argsort], lw = 1.0, zorder = 0, color = 'grey', linestyle = ':', dashes = (1.5, 1))
        
        spol, = axes[0].plot(np.nan, np.nan, color = 'black', lw = 1.0, zorder = 0, linestyle = '--', dashes = (4, 1))
        dterm, = axes[0].plot(np.nan, np.nan, color = 'black', lw = 1.0, zorder = 0, linestyle = ':', dashes = (1.5, 1))

        leg2 = axes[0].legend([spol, dterm, whole], ['source-pol terms', 'instrumental-pol terms', 'whole terms'], loc = 'lower left', frameon=False, fontsize = 13, handlelength=1.0)
        axes[0].add_artist(leg1)
        
        
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
                if(dumit == 0): dumxticklabel[it] = '{:02d}d/'.format(np.min(day) + 1) + dumxticklabel[it]
                dumit += 1
    
    dumxticklabel[0] = '{:02d}d/'.format(np.min(day)) + dumxticklabel[0]
    
    
    axes[1].set_xticklabels(dumxticklabel)

    figure.savefig("{:}.{:}.baseline_{:}_{:}.U.{:}".format(filename, source, antname1, antname2, filetype), bbox_inches = 'tight')
    
    plt.close('all')



def residualplot(markerarr, colorarr, dayarr, k, nant, antname, source, dumfit, time, ant1, ant2, sourcearr, qamp, qphas, uamp, uphas, qsigma, usigma, filename, tsep = 2. / 60., title = None, filetype = 'pdf'):
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
            
        axes[0].set_xlim(np.min(time) - (np.max(time) - np.min(time)) * 0.5, np.max(time) + (np.max(time) - np.min(time)) * 0.5)
        axes[1].set_xlim(np.min(time) - (np.max(time) - np.min(time)) * 0.5, np.max(time) + (np.max(time) - np.min(time)) * 0.5)
        
#            axes[0].set(title = 'BL229AE')
        
        axes[1].set(xlabel = 'Time (UT)')
        
        axes[0].set(ylabel = 'Stokes Q (sigma)')
        axes[1].set(ylabel = 'Stokes U (sigma)')
        
        axes[0].set_title(title)
        
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
                
            axes[0].scatter(binx, biny, s = 30, marker = markerarr[l], facecolor = 'None', edgecolor = colorarr[l], label = source[l].upper(), zorder = 0)
            
            dumy = np.abs(((data_u[select] - mod_u[select]) / usigma[select]))

            dumy = dumy[argsort]
            
            binx = np.zeros(len(boundary_left))
            biny = np.zeros(len(boundary_left))
            binyerr = np.zeros(len(boundary_left))
            
            for j in range(len(boundary_left)):
                binx[j] = (boundary_left[j] + boundary_right[j]) / 2.
                biny[j] = np.mean(dumy[(dumx >= boundary_left[j]) & (dumx <= boundary_right[j])])
                binyerr[j] = np.std(dumy[(dumx >= boundary_left[j]) & (dumx <= boundary_right[j])])
                
            axes[1].scatter(binx, biny, s = 30, marker = markerarr[l], facecolor = 'None', edgecolor = colorarr[l], zorder = 0)
            
                   
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
                    if(dumit == 0): dumxticklabel[it] = '{:02d}d/'.format(np.min(dayarr) + 1) + dumxticklabel[it]
                    dumit += 1
        
        axes[1].set_xticklabels(dumxticklabel)


        figure.savefig(filename+'.'+antname[m]+'.'+filetype, bbox_inches = 'tight')
        
        plt.close('all')


