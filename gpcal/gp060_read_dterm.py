#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:13:08 2024

@author: jpark
"""


import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


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



dtermread = pd.read_csv('/home/jpark/gp060.hsa.k.edt.pol.iter10.dterm.csv', header = 0, skiprows=0, delimiter = '\t', index_col = 0)
    
antname = np.array(dtermread['antennas'])
pol_IF = np.array(dtermread['IF'])
dum_DRArr = np.array(dtermread['DRArr'])
dum_DLArr = np.array(dtermread['DLArr'])

DRArr = np.array([complex(it) for it in dum_DRArr])
DLArr = np.array([complex(it) for it in dum_DLArr])


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf'] + \
               ['blue', 'red', 'orange', 'steelblue', 'green', 'slategrey', 'cyan', 'royalblue', 'purple', 'blueviolet', 'darkcyan', 'darkkhaki', 'magenta', 'navajowhite', 'darkred'] + \
                   ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf'] + \
                                  ['blue', 'red', 'orange', 'steelblue', 'green', 'slategrey', 'cyan', 'royalblue', 'purple', 'blueviolet', 'darkcyan', 'darkkhaki', 'magenta', 'navajowhite', 'darkred']
# Define a list of markers
markerarr = ['o', '^', 's', '<', 'p', '*', 'X', 'P', 'D', 'v', 'd', 'x'] * 5


select = (np.abs(DRArr) != 0.) & (np.abs(DLArr) != 0.)

antname = antname[select]
pol_IF = pol_IF[select]
DRArr = DRArr[select]
DLArr = DLArr[select]

uniqant = np.unique(antname)


fig, ax = plt.subplots(figsize=(8, 8))

ax.tick_params(length=6, width=2,which = 'major')
ax.tick_params(length=4, width=1.5,which = 'minor')

plt.grid()

dum = 0

for m in range(len(uniqant)):
    
    select = (antname == uniqant[m])
    
    drreal = np.real(DRArr[select]) * 1e2
    drimag = np.imag(DRArr[select]) * 1e2
    
    dlreal = np.real(DLArr[select]) * 1e2
    dlimag = np.imag(DLArr[select]) * 1e2
    
    ax.scatter(drreal, drimag, s = 180, facecolor = colors[dum], edgecolor = colors[dum], marker = markerarr[dum], label = antname[dum])
    ax.scatter(dlreal, dlimag, s = 180, facecolor = 'None', edgecolor = colors[dum], marker = markerarr[dum])
    
    dum += 1
    

dumbound = np.max([np.abs(DRArr), np.abs(DLArr)]) * 1e2 * 1.2
    

plt.xlim(-dumbound, dumbound)
plt.ylim(-dumbound, dumbound)
plt.xlabel('Real (\%)')
plt.ylabel('Imaginary (\%)')
# plt.title(dataname)


# ax.annotate('Calibrators:', xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
# for i in range(len(source)):
#     ax.annotate(source[i], xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25*(i+2)), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')


# if(pol >= 0):
#     ax.annotate('Pol-selfcal, Iteration = {:d}/{:d}'.format(pol, selfpoliter), xy=(1, 1), xycoords = 'axes fraction', xytext = (-25, -25), size = 22, textcoords='offset pixels', horizontalalignment='right', verticalalignment='top')


leg1 = ax.legend(loc='lower left', fontsize = 18, frameon=False, markerfirst=True, handlelength=0.3, labelspacing=0.3)

rcp = ax.scatter([], [], s = 120, facecolor = 'black', edgecolor = 'black', marker = 'o')
lcp = ax.scatter([], [], s = 120, facecolor = 'none', edgecolor = 'black', marker = 'o')

leg2 = ax.legend([rcp, lcp], ['Filled - RCP', 'Open - LCP'], loc = 'lower right', frameon=False, fontsize = 24, handlelength=0.3)

ax.add_artist(leg1)

plt.savefig('/home/jpark/gp060.hsa.DTerms.png', bbox_inches = 'tight')
plt.close('all')



# # If dplot_IFsep == True, then plot the D-terms for each IF separately.
# else:

#     for k in range(ifnum):
        
#         if(np.sum((IFarr == k+1)) == 0.):
#             continue
        
#         fig, ax = plt.subplots(figsize=(8, 8))
        
#         ax.tick_params(length=6, width=2,which = 'major')
#         ax.tick_params(length=4, width=1.5,which = 'minor')
        
#         plt.grid()

        
#         for m in range(len(antname)):
            
#             select = (antname == antname[m]) & (IFarr == k+1)
            
#             drreal = np.real(DRArr[select]) * 1e2
#             drimag = np.imag(DRArr[select]) * 1e2
            
#             dlreal = np.real(DLArr[select]) * 1e2
#             dlimag = np.imag(DLArr[select]) * 1e2
            
            
#             if(drreal == 0.) & (drimag == 0.) & (dlreal == 0.) & (dlimag == 0.): continue
            
#             ax.scatter(drreal[drreal != 0.], drimag[drimag != 0.], s = 180, facecolor = colors[m], edgecolor = colors[m], marker = markerarr[m], label = antname[m])
#             ax.scatter(dlreal[drreal != 0.], dlimag[drimag != 0.], s = 180, facecolor = 'None', edgecolor = colors[m], marker = markerarr[m])
            
#         if lpcal:
#             for l in range(len(source)):
#                 read = pd.read_csv(direc+'gpcal/'+dataname+source[l]+'.an', header = None, skiprows=1, delimiter = '\t')
#                 anname, lpcaldr, lpcaldl = read[1].to_numpy(), read[2].to_numpy(), read[3].to_numpy()
#                 dumant, dumdrreal, dumdrimag, dumdlreal, dumdlimag = anname[::2], lpcaldr[::2] * 1e2, lpcaldr[1::2] * 1e2, lpcaldl[::2] * 1e2, lpcaldl[1::2] * 1e2
#                 for m in range(len(antname)):
#                     dumx = dumdrreal[dumant == antname[m]]
#                     dumy = dumdrimag[dumant == antname[m]]
#                     if(dumx[k] == 0.) & (dumy[k] == 0.): continue
#                     ax.scatter(dumx[k], dumy[k], s = 20, facecolor = colors[m], edgecolor = colors[m], marker = markerarr[m], \
#                                alpha = 0.2)
#                     dumx = dumdlreal[dumant == antname[m]]
#                     dumy = dumdlimag[dumant == antname[m]]
#                     ax.scatter(dumx[k], dumy[k], s = 20, facecolor = 'None', edgecolor = colors[m], marker = markerarr[m], \
#                                alpha = 0.2)
                    

#         if(drange == None):
#             dumbound = np.max([np.abs(DRArr), np.abs(DLArr)]) * 1e2 * 1.2
#         else:
#             dumbound = drange
        
#         plt.xlim(-dumbound, dumbound)
#         plt.ylim(-dumbound, dumbound)
#         plt.xlabel('Real (\%)')
#         plt.ylabel('Imaginary (\%)')
#         plt.title(dataname + ', IF' + str(k+1))
        
        
#         sourcename = ''
#         for i in range(len(source)):
#             sourcename = sourcename + source[i]+'-'
        
#         ax.annotate('Calibrators:', xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
#         for i in range(len(source)):
#             ax.annotate(source[i], xy=(0, 1), xycoords = 'axes fraction', xytext = (25, -25*(i+2)), size = 18, textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')
        
#         if(pol >= 0):
#             ax.annotate('Pol-selfcal, Iteration = {:d}/{:d}'.format(pol, selfpoliter), xy=(1, 1), xycoords = 'axes fraction', xytext = (-25, -25), size = 22, textcoords='offset pixels', horizontalalignment='right', verticalalignment='top')

        
#         leg1 = ax.legend(loc='lower left', fontsize = 18, frameon=False, markerfirst=True, handlelength=0.3, labelspacing=0.3)
        
#         rcp = ax.scatter([], [], s = 120, facecolor = 'black', edgecolor = 'black', marker = 'o')
#         lcp = ax.scatter([], [], s = 120, facecolor = 'none', edgecolor = 'black', marker = 'o')
        
#         leg2 = ax.legend([rcp, lcp], ['Filled - RCP', 'Open - LCP'], loc = 'lower right', frameon=False, fontsize = 24, handlelength=0.3)
        
#         if lpcal:
#             gpcal = ax.scatter([], [], s = 180, facecolor = 'black', edgecolor = 'black', marker = 'o')
#             lpcal = ax.scatter([], [], s = 20, facecolor = 'black', edgecolor = 'black', marker = 'o')
#             leg3 = ax.legend([gpcal, lpcal], ['GPCAL', 'LPCAL'], loc = 'upper right', frameon=False, fontsize = 24, handlelength=0.3)
        
#         ax.add_artist(leg1)
        
#         if lpcal:
#             ax.add_artist(leg2)
        
        
#         plt.savefig(direc + 'gpcal/' + filename + '.IF'+str(k+1) + '.'+filetype, bbox_inches = 'tight')
#         plt.close('all')
                        

