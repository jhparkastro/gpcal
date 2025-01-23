#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 00:16:18 2021

@author: jpark
"""

import os

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


# Default matplotlib parameters
plt.rc('font', size=21)
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('font', serif='Helvetica Neue')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
plt.rc('axes', titlesize=28)
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.subplots_adjust(left = 0.15, bottom = 0.15)
       


direc = '/home/jpark/gpcal_mpifr_tutorial_check/gpcal/examples/timecal_vlba7mm/'

if not os.path.isdir(direc + 'epsfiles'):
    os.mkdir(direc + 'epsfiles')

session = 'bm462m.q.timecal'

sourcearr1 = ['3C273', '3C454.3', '3C345']
sourcearr2 = ['3C279', 'CTA102', '1633+382']

separr = [10.4, 6.8, 2.2]

selectstation = ['FD', 'MK', 'PT']



color1 = 'red'
color2 = 'royalblue'


recon = pd.read_csv(direc+'gpcal/{:}.timecal.final.dterm.csv'.format(session), delimiter = '\t', index_col = 0)

recon_ant = np.array(recon['antennas'])
recon_IF = np.array(recon['IF'])
recon_scan = np.array(recon['scan'])
recon_scantime = np.array(recon['scantime'])
recon_scansource = np.array(recon['scansource'])
recon_DRArr = np.array(recon['DRArr'])
recon_DLArr = np.array(recon['DLArr'])

recon_DRArr = np.array([complex(it) for it in recon_DRArr])
recon_DLArr = np.array([complex(it) for it in recon_DLArr])

select = (np.abs(recon_DRArr) != 0.) & (np.abs(recon_DLArr) != 0.)

recon_ant, recon_IF, recon_scan, recon_scantime, recon_scansource, recon_DRArr, recon_DLArr = \
    recon_ant[select], recon_IF[select], recon_scan[select], recon_scantime[select], recon_scansource[select], recon_DRArr[select], recon_DLArr[select]


uniqant = np.unique(recon_ant)
uniqtime = np.unique(recon_scantime)


tsep = 10/60.


ysize = 10

xsize = ysize * len(selectstation)

for k in range(len(sourcearr1)):
    
    sep = separr[k]
    
    source1 = sourcearr1[k]
    source2 = sourcearr2[k]
    

    fig, axs = plt.subplots(2, len(selectstation), sharex = False, gridspec_kw={'hspace': 0, 'wspace': 0.12}, figsize=(xsize, ysize))
    
    for ax in axs.flat:
        ax.tick_params(length=6, width=2,which = 'major')
        ax.tick_params(length=4, width=1.5,which = 'minor')
    
    fig.text(0.48, 0.03, 'UT (hour)', fontsize = 30)
    axs[0,0].set_ylabel(r'$Re(D_{\rm off\ axis})$ (\%)', fontsize = 28)
    axs[1,0].set_ylabel(r'$Im(D_{\rm off\ axis})$ (\%)', fontsize = 28)
    
        
    dum = 0
    
    for l in range(len(uniqant)):
        
        if(uniqant[l] not in selectstation):
            continue
        
        select1 = (recon_ant == uniqant[l]) & (recon_scansource == source1)
        
        time1 = recon_scantime[select1]
        DR1 = recon_DRArr[select1]
        DL1 = recon_DLArr[select1]
        IF1 = recon_IF[select1]
    
        
        select2 = (recon_ant == uniqant[l]) & (recon_scansource == source2)
        
        time2 = recon_scantime[select2]    
        IF2 = recon_IF[select2]
        DR2 = recon_DRArr[select2]
        DL2 = recon_DLArr[select2]
        
        
        drdev = []
        dldev = []
        for i in range(len(time1)):
            for j in range(len(time2)):
                if(np.abs(time1[i] - time2[j]) < tsep) & (np.abs(time1[i] - time2[j]) > 1./60.) & (IF1[i] == IF2[j]):
                    drdev.append(np.real(DR1[i] - DR2[j]))
                    drdev.append(np.imag(DR1[i] - DR2[j]))
                    dldev.append(np.real(DL1[i] - DL2[j]))
                    dldev.append(np.imag(DL1[i] - DL2[j]))
                    
        print("{:}, DR = {:4.2f}, DL = {:4.2f}".format(uniqant[l], np.mean(np.abs(drdev)) * 1e2, np.mean(np.abs(dldev)) * 1e2))
    
    
    
        axs[0,dum].plot(time1, np.real(DR1) * 1e2, markersize = 10, marker = 'o', markerfacecolor = color1, markeredgecolor = color1, color = color1, label = source1, linestyle = 'solid')
        axs[0,dum].plot(time2, np.real(DR2) * 1e2, markersize = 10, marker = 'o', markerfacecolor = color2, markeredgecolor = color2, color = color2, label = source2, linestyle = 'solid')
        
        axs[0,dum].plot(time1, np.real(DL1) * 1e2, markersize = 10, marker = 's', markerfacecolor = 'None', markeredgecolor = color1, color = color1, linestyle = 'dotted')
        axs[0,dum].plot(time2, np.real(DL2) * 1e2, markersize = 10, marker = 's', markerfacecolor = 'None', markeredgecolor = color2, color = color2, linestyle = 'dotted')
        
        
        axs[1,dum].plot(time1, np.imag(DR1) * 1e2, markersize = 10, marker = 'o', markerfacecolor = color1, markeredgecolor = color1, color = color1, linestyle = 'solid')
        axs[1,dum].plot(time2, np.imag(DR2) * 1e2, markersize = 10, marker = 'o', markerfacecolor = color2, markeredgecolor = color2, color = color2, linestyle = 'solid')
        
        axs[1,dum].plot(time1, np.imag(DL1) * 1e2, markersize = 10, marker = 's', markerfacecolor = 'None', markeredgecolor = color1, color = color1, linestyle = 'dotted')
        axs[1,dum].plot(time2, np.imag(DL2) * 1e2, markersize = 10, marker = 's', markerfacecolor = 'None', markeredgecolor = color2, color = color2, linestyle = 'dotted')
        
        
        drdevl1 = np.mean(np.abs(drdev)) * 1e2
        dldevl1 = np.mean(np.abs(dldev)) * 1e2
        if(len(drdev) != 0):
            axs[1,dum].annotate(r'$\langle L_{1} \rangle_{D_R}$' + ' = {:4.2f}'.format(drdevl1) + ' \%', xy=(0, 1), xycoords = 'axes fraction', xytext = (20, -20), size = 24, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
        if(len(dldev) != 0):
            axs[1,dum].annotate(r'$\langle L_{1} \rangle_{D_L}$' + ' = {:4.2f}'.format(dldevl1) + ' \%', xy=(0, 1), xycoords = 'axes fraction', xytext = (20, -50), size = 24, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
        
        leg1 = axs[0,dum].legend(loc='upper right', fontsize = 24, frameon=False, markerfirst=True, handlelength=0.05, labelspacing=0.05, markerscale = 1.0)    
        
        rcp = ax.scatter([], [], s = 30, marker = 'o', facecolor = 'black', edgecolor = 'black')
        lcp = ax.scatter([], [], s = 30, marker = 's', facecolor = 'None', edgecolor = 'black')
        
        leg2 = axs[0,dum].legend([rcp, lcp], ['Filled - RCP', 'Open - LCP'], loc='upper left', fontsize = 24, frameon = False, markerfirst=True, handlelength=0.5, labelspacing=0.20, markerscale = 2.0)    
        
        if(dum == 0):
            axs[0,dum].annotate(r'$\Delta d$' + ' = {:4.1f} degrees'.format(sep), xy=(1, 1), xycoords = 'axes fraction', xytext = (-20, -90), size = 24, textcoords='offset pixels', horizontalalignment='right', verticalalignment='top')
        
    
        axs[0,dum].add_artist(leg1)
    
        fig.align_ylabels(axs)
        
        xticks = axs[1,dum].get_xticks()
        dumxticks = np.copy(xticks)
        dumxticks[dumxticks > 24.] -= 24.
        
        xticklabel = [str(int(it)) for it in dumxticks]
        
        
        axs[0,dum].set_xticks(xticks)
        axs[1,dum].set_xticks(xticks)
        axs[1,dum].set_xticklabels(xticklabel)
        axs[0,dum].set_xticklabels([])
        
        axs[0,dum].set_title('{:s}'.format(uniqant[l]), fontsize = 30)
        
        
        sharetime = np.concatenate([time1, time2])
        
        axs[0,dum].set_xlim(np.min(sharetime) - (np.max(sharetime) - np.min(sharetime)) * 0.2, np.max(sharetime) + (np.max(sharetime) - np.min(sharetime)) * 0.2)
        axs[0,dum].set_ylim(axs[0,dum].get_ylim()[0] - (axs[0,dum].get_ylim()[1] - axs[0,dum].get_ylim()[0]) * 0.27, axs[0,dum].get_ylim()[1] + (axs[0,dum].get_ylim()[1] - axs[0,dum].get_ylim()[0]) * 0.8)
        axs[1,dum].set_xlim(np.min(sharetime) - (np.max(sharetime) - np.min(sharetime)) * 0.2, np.max(sharetime) + (np.max(sharetime) - np.min(sharetime)) * 0.2)
        axs[1,dum].set_ylim(axs[1,dum].get_ylim()[0] - (axs[1,dum].get_ylim()[1] - axs[1,dum].get_ylim()[0]) * 0.27, axs[1,dum].get_ylim()[1] + (axs[1,dum].get_ylim()[1] - axs[1,dum].get_ylim()[0]) * 0.8)
        
        
        if('bp249h' in session) & (uniqant[l] == "LA"):
            axs[0,dum].plot(np.repeat(24. + 3., 100), np.linspace(axs[0,dum].get_ylim()[0], axs[0,dum].get_ylim()[1], 100), color = 'black', linestyle = 'dashed')
            axs[1,dum].plot(np.repeat(24. + 3., 100), np.linspace(axs[1,dum].get_ylim()[0], axs[1,dum].get_ylim()[1], 100), color = 'black', linestyle = 'dashed')
            axs[0,dum].text(24. + 2.0, 0.75, 'Rain')
            axs[0,dum].text(24. + 1.97, 0.5, 'Start')
            
        
        dum += 1
        
        
    fig.savefig(direc+'epsfiles/{:}.{:}.{:}.dterm.comparison.selected.pdf'.format(session, source1, source2), bbox_inches = 'tight')
    fig.savefig(direc+'epsfiles/{:}.{:}.{:}.dterm.comparison.selected.png'.format(session, source1, source2), bbox_inches = 'tight')
    
    plt.close('all')



