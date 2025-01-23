
import numpy as np
import os

from Wizardry.AIPSData import AIPSUVData as WAIPSUVData

import matplotlib
import matplotlib.pyplot as plt


# Default matplotlib parameters
plt.rc('font', size=21)
matplotlib.rc('font', family='Times New Roman')
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



direc = '/home/jpark/gpcal_mpifr_tutorial_check/gpcal/examples/channelcal_vlba2cm/'


if not os.path.isdir(direc + 'epsfiles'):
    os.mkdir(direc + 'epsfiles')


inname = 'bl273ak.u'
sessionname = 'BL273AK'

refant = 'BR'
source = '1253-055'
sourcename = '3C 279'

# The selected time range for plotting: 1/00:54:14 - 1/00:55:44
time1 = 24. + 54. / 60. + 15. / 3600. 
time2 = 24. + 55. / 60. + 40. / 3600.

# The AIPS catalog class before the correction.
inclass1 = 'SPLAT'

# The AIPS catalog class after the correction.
inclass2 = 'DTCOR'

inseq = 1

indisk = 1


colorarr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf'] + \
               ['blue', 'red', 'orange', 'steelblue', 'green', 'slategrey', 'cyan', 'royalblue', 'purple', 'blueviolet', 'darkcyan', 'darkkhaki', 'magenta', 'navajowhite', 'darkred']
               

data = WAIPSUVData(inname, inclass1, inseq, indisk)

obsdate = data.header.date_obs
year = int(obsdate[0:4])
month = int(obsdate[5:7])
day = int(obsdate[8:10])
        
sutable = data.table('SU', 1)

wholesource = []
sourceid = []

for row in sutable:
    wholesource.append(row.source.replace(' ', ''))
    sourceid.append(row.id__no)

wholesource = np.array(wholesource)
sourceid = np.array(sourceid)

antname = []

antable = data.table('AN', 1)
for row in antable:
    antname.append(row.anname.replace(' ', ''))

nant = len(antname)
    
ifnum = data.header['naxis'][3]
nchan = data.header['naxis'][2]


data = WAIPSUVData(inname, inclass1, inseq, indisk)

time, ant1, ant2 = np.zeros(len(data)), np.zeros(len(data), dtype = int), np.zeros(len(data), dtype = int)
sourcearr = np.chararray(len(data), itemsize = 16)
dumvis = np.zeros((len(data), ifnum, nchan, 4, 3))

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



data2 = WAIPSUVData(inname, inclass2, inseq, indisk)

sutable = data2.table('SU', 1)

wholesource = []
sourceid = []

for row in sutable:
    wholesource.append(row.source.replace(' ', ''))
    sourceid.append(row.id__no)

wholesource = np.array(wholesource)
sourceid = np.array(sourceid)


cor_time, cor_ant1, cor_ant2 = np.zeros(len(data2)), np.zeros(len(data2), dtype = int), np.zeros(len(data2), dtype = int)
cor_sourcearr = np.chararray(len(data2), itemsize = 16)
cor_dumvis = np.zeros((len(data2), ifnum, nchan, 4, 3))

dum = 0
for row in data2:
    cor_time[dum] = row.time
    dumsel = (sourceid == row.source)
    if(np.sum(dumsel) == 0):
        cor_sourcearr[dum] = 'None'
    else:
        cor_sourcearr[dum] = wholesource[dumsel][0]
    cor_ant1[dum] = row.baseline[0]
    cor_ant2[dum] = row.baseline[1]
    cor_dumvis[dum,:,:,:,:] = row.visibility
    
    dum += 1
            
cor_time = cor_time * 24.
cor_ant1 = cor_ant1 - 1
cor_ant2 = cor_ant2 - 1



ysize = 8
xsize = 15


width_ratios = []
height_ratios = [0.4, 1, 0.2, 0.4, 1]

for l in range(nant-1):
    width_ratios = width_ratios + [1.] * ifnum
    if(l != (nant-2)):
        width_ratios.append(0.4)
    
gridspec = dict(hspace=0.02, wspace=0.0, width_ratios = width_ratios, height_ratios = height_ratios)


ncol = (len(antname) - 1) * (ifnum + 1) - 1

fig, axs = plt.subplots(5, ncol, sharex=True, sharey='row', figsize=(xsize,ysize), gridspec_kw = gridspec)

antname = np.array(antname)

dumantname = np.delete(antname, np.where(antname == refant)[0][0])

for l in range(nant-1):
    fig.text(axs[0, l * (ifnum+1) + int(ifnum/4)].get_position().x0 - (axs[0, l * (ifnum+1) + int(ifnum/4)].get_position().x1 - axs[0, l * (ifnum+1) + int(ifnum/4)].get_position().x0) * 0.5, \
             axs[0, l * (ifnum+1) + int(ifnum/4)].get_position().y1 + 0.02, '{:}-{:}'.format(refant, dumantname[l]))
    
    
for l in range(nant-2):
    for i in range(axs.shape[0]):
        axs[i, (l + 1) * (ifnum + 1) - 1].axis('off')
    
for j in range(axs.shape[1]):
    axs[2, j].axis('off')

plt.xticks([], [])

for ax in axs.flat:
    ax.tick_params(length=0, width=0.5,which = 'major')
    

axs[0,0].set_ylabel(r'$\phi$ (deg)')
axs[3,0].set_ylabel(r'$\phi$ (deg)')

axs[1,0].set_ylabel(r'$|V|$ (Jy)')
axs[4,0].set_ylabel(r'$|V|$ (Jy)')


fig.align_ylabels()
fig.text(0.5, 0.07, 'Frequency', ha='center')
fig.text(0.1, 0.95, '{:}, {:}, {:}'.format(sessionname, sourcename, 'RL'))
fig.text(0.5, 0.95, 'Before Correction', ha='center')
fig.text(0.5, 0.48, 'After Correction', ha='center')


pminarr = []
pmaxarr = []

dumselect = (sourcearr == source)
dumselect = dumselect & ((antname[ant1] == refant) | (antname[ant2] == refant))

selected_vis = dumvis[dumselect]
selected_ant1 = ant1[dumselect]
selected_ant2 = ant2[dumselect]
selected_time = time[dumselect]




dumselect = (cor_sourcearr == source)
dumselect = dumselect & ((antname[cor_ant1] == refant) | (antname[cor_ant2] == refant))

cor_selected_vis = cor_dumvis[dumselect]
cor_selected_ant1 = cor_ant1[dumselect]
cor_selected_ant2 = cor_ant2[dumselect]
cor_selected_time = cor_time[dumselect]


dum = 0

for l in range(nant):
    if(antname[l] == refant):
        continue    
    
    for ifn in range(ifnum):
        dumselect1 = ((antname[selected_ant1] == antname[l]) | (antname[selected_ant2] == antname[l]))
        dumselect2 = ((selected_time > time1) & (selected_time < time2))
        dumselect = (dumselect1 & dumselect2)
        
        rrvis = np.zeros(nchan, dtype = complex)
        llvis = np.zeros(nchan, dtype = complex)
        rlvis = np.zeros(nchan, dtype = complex)
        lrvis = np.zeros(nchan, dtype = complex)
        
        for chan in range(nchan):
            
            dumreal = selected_vis[dumselect, ifn, chan, 0, 0]
            dumimag = selected_vis[dumselect, ifn, chan, 0, 1]
            dumweight = selected_vis[dumselect, ifn, chan, 0, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
                
            rrvis[chan] = dumavg
            
            dumreal = selected_vis[dumselect, ifn, chan, 1, 0]
            dumimag = selected_vis[dumselect, ifn, chan, 1, 1]
            dumweight = selected_vis[dumselect, ifn, chan, 1, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
                
            llvis[chan] = dumavg
            
            dumreal = selected_vis[dumselect, ifn, chan, 2, 0]
            dumimag = selected_vis[dumselect, ifn, chan, 2, 1]
            dumweight = selected_vis[dumselect, ifn, chan, 2, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
            
            rlvis[chan] = dumavg
            
            dumreal = selected_vis[dumselect, ifn, chan, 3, 0]
            dumimag = selected_vis[dumselect, ifn, chan, 3, 1]
            dumweight = selected_vis[dumselect, ifn, chan, 3, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
            
            lrvis[chan] = dumavg
            
            
        if(np.abs(np.sum(rlvis)) < 1e-8):
            dum += 1
            continue
        
        dumx = np.arange(nchan) + 1
        dumdumselect = (np.abs(rlvis) != 0.)
        axs[0, dum].scatter(dumx[dumdumselect], np.degrees(np.angle(rlvis[dumdumselect])), s = 5, facecolor = colorarr[l], edgecolor = colorarr[l], alpha = 0.6)
        axs[1, dum].scatter(dumx[dumdumselect], np.abs(rlvis[dumdumselect]), s = 5, facecolor = colorarr[l], edgecolor = colorarr[l], alpha = 0.6)


        dumselect1 = ((antname[cor_selected_ant1] == antname[l]) | (antname[cor_selected_ant2] == antname[l]))
        dumselect2 = ((cor_selected_time > time1) & (cor_selected_time < time2))
        dumselect = (dumselect1 & dumselect2)
        
        cor_rrvis = np.zeros(nchan, dtype = complex)
        cor_llvis = np.zeros(nchan, dtype = complex)
        cor_rlvis = np.zeros(nchan, dtype = complex)
        cor_lrvis = np.zeros(nchan, dtype = complex)
        
        for chan in range(nchan):
            
            dumreal = cor_selected_vis[dumselect, ifn, chan, 0, 0]
            dumimag = cor_selected_vis[dumselect, ifn, chan, 0, 1]
            dumweight = cor_selected_vis[dumselect, ifn, chan, 0, 2]
            dum2select = (dumweight > 0.)
            
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
                
            cor_rrvis[chan] = dumavg
            
            dumreal = cor_selected_vis[dumselect, ifn, chan, 1, 0]
            dumimag = cor_selected_vis[dumselect, ifn, chan, 1, 1]
            dumweight = cor_selected_vis[dumselect, ifn, chan, 1, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
                
            cor_llvis[chan] = dumavg
            
            dumreal = cor_selected_vis[dumselect, ifn, chan, 2, 0]
            dumimag = cor_selected_vis[dumselect, ifn, chan, 2, 1]
            dumweight = cor_selected_vis[dumselect, ifn, chan, 2, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
            
            cor_rlvis[chan] = dumavg
            
            dumreal = cor_selected_vis[dumselect, ifn, chan, 3, 0]
            dumimag = cor_selected_vis[dumselect, ifn, chan, 3, 1]
            dumweight = cor_selected_vis[dumselect, ifn, chan, 3, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
            
            cor_lrvis[chan] = dumavg
            
        dumx = np.arange(nchan) + 1
        dumdumselect = (np.abs(rlvis) != 0.)
        axs[3, dum].scatter(dumx[dumdumselect], np.degrees(np.angle(cor_rlvis[dumdumselect])), s = 5, facecolor = colorarr[l], edgecolor = colorarr[l], alpha = 0.6)
        axs[4, dum].scatter(dumx[dumdumselect], np.abs(cor_rlvis[dumdumselect]), s = 5, facecolor = colorarr[l], edgecolor = colorarr[l], alpha = 0.6)
        
        
        dum += 1
        
    dum += 1
    

for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        yticks = axs[i,j].get_yticks()
        axs[i,j].set_xlim(axs[i,j].get_xlim())
        axs[i,j].set_ylim(axs[i,j].get_ylim())
        for k in range(len(yticks)-1):
            axs[i,j].plot(np.linspace(axs[i,j].get_xlim()[0], axs[i,j].get_xlim()[1], 100), np.repeat(yticks[k], 100), color = 'black', linewidth = 0.5)


        
plt.savefig(direc + '/epsfiles/{:}.possm.RL.pdf'.format(inname), bbox_inches='tight')  
plt.savefig(direc + '/epsfiles/{:}.possm.RL.png'.format(inname), bbox_inches='tight')  



ysize = 8
xsize = 15


width_ratios = []
height_ratios = [0.4, 1, 0.2, 0.4, 1]

for l in range(nant-1):
    width_ratios = width_ratios + [1.] * ifnum
    if(l != (nant-2)):
        width_ratios.append(0.4)
    
gridspec = dict(hspace=0.02, wspace=0.0, width_ratios = width_ratios, height_ratios = height_ratios)


ncol = (len(antname) - 1) * (ifnum + 1) - 1

fig, axs = plt.subplots(5, ncol, sharex=True, sharey='row', figsize=(xsize,ysize), gridspec_kw = gridspec)

antname = np.array(antname)

dumantname = np.delete(antname, np.where(antname == refant)[0][0])

for l in range(nant-1):
    fig.text(axs[0, l * (ifnum+1) + int(ifnum/4)].get_position().x0 - (axs[0, l * (ifnum+1) + int(ifnum/4)].get_position().x1 - axs[0, l * (ifnum+1) + int(ifnum/4)].get_position().x0) * 0.5, \
             axs[0, l * (ifnum+1) + int(ifnum/4)].get_position().y1 + 0.02, '{:}-{:}'.format(refant, dumantname[l]))
    
    
for l in range(nant-2):
    for i in range(axs.shape[0]):
        axs[i, (l + 1) * (ifnum + 1) - 1].axis('off')
    
for j in range(axs.shape[1]):
    axs[2, j].axis('off')

plt.xticks([], [])

for ax in axs.flat:
    ax.tick_params(length=0, width=0.5,which = 'major')
    

axs[0,0].set_ylabel(r'$\phi$ (deg)')
axs[3,0].set_ylabel(r'$\phi$ (deg)')

axs[1,0].set_ylabel(r'$|V|$ (Jy)')
axs[4,0].set_ylabel(r'$|V|$ (Jy)')


fig.align_ylabels()
fig.text(0.5, 0.07, 'Frequency', ha='center')
fig.text(0.1, 0.95, '{:}, {:}, {:}'.format(sessionname, sourcename, 'LR'))
fig.text(0.5, 0.95, 'Before Correction', ha='center')
fig.text(0.5, 0.48, 'After Correction', ha='center')


pminarr = []
pmaxarr = []

dumselect = (sourcearr == source)
dumselect = dumselect & ((antname[ant1] == refant) | (antname[ant2] == refant))

selected_vis = dumvis[dumselect]
selected_ant1 = ant1[dumselect]
selected_ant2 = ant2[dumselect]




dumselect = (cor_sourcearr == source)
dumselect = dumselect & ((antname[cor_ant1] == refant) | (antname[cor_ant2] == refant))

cor_selected_vis = cor_dumvis[dumselect]
cor_selected_ant1 = cor_ant1[dumselect]
cor_selected_ant2 = cor_ant2[dumselect]


dum = 0

for l in range(nant):
    if(antname[l] == refant):
        continue    
    
    for ifn in range(ifnum):
        dumselect1 = ((antname[selected_ant1] == antname[l]) | (antname[selected_ant2] == antname[l])) 
        dumselect2 = (selected_time > time1) & (selected_time < time2)

        dumselect = (dumselect1 & dumselect2)

        rrvis = np.zeros(nchan, dtype = complex)
        llvis = np.zeros(nchan, dtype = complex)
        rlvis = np.zeros(nchan, dtype = complex)
        lrvis = np.zeros(nchan, dtype = complex)
        
        for chan in range(nchan):
            
            dumreal = selected_vis[dumselect, ifn, chan, 0, 0]
            dumimag = selected_vis[dumselect, ifn, chan, 0, 1]
            dumweight = selected_vis[dumselect, ifn, chan, 0, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
                
            rrvis[chan] = dumavg
            
            dumreal = selected_vis[dumselect, ifn, chan, 1, 0]
            dumimag = selected_vis[dumselect, ifn, chan, 1, 1]
            dumweight = selected_vis[dumselect, ifn, chan, 1, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
                
            llvis[chan] = dumavg
            
            dumreal = selected_vis[dumselect, ifn, chan, 2, 0]
            dumimag = selected_vis[dumselect, ifn, chan, 2, 1]
            dumweight = selected_vis[dumselect, ifn, chan, 2, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
            
            rlvis[chan] = dumavg
            
            dumreal = selected_vis[dumselect, ifn, chan, 3, 0]
            dumimag = selected_vis[dumselect, ifn, chan, 3, 1]
            dumweight = selected_vis[dumselect, ifn, chan, 3, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
            
            lrvis[chan] = dumavg
        
        if(np.abs(np.sum(rlvis)) < 1e-8):
            dum += 1
            continue


        dumx = np.arange(nchan) + 1
        dumdumselect = (np.abs(rlvis) != 0.)
        axs[0, dum].scatter(dumx[dumdumselect], np.degrees(np.angle(lrvis[dumdumselect])), s = 5, facecolor = colorarr[l], edgecolor = colorarr[l], alpha = 0.6)
        axs[1, dum].scatter(dumx[dumdumselect], np.abs(lrvis[dumdumselect]), s = 5, facecolor = colorarr[l], edgecolor = colorarr[l], alpha = 0.6)


        dumselect = ((antname[cor_selected_ant1] == antname[l]) | (antname[cor_selected_ant2] == antname[l]))
        
        cor_rrvis = np.zeros(nchan, dtype = complex)
        cor_llvis = np.zeros(nchan, dtype = complex)
        cor_rlvis = np.zeros(nchan, dtype = complex)
        cor_lrvis = np.zeros(nchan, dtype = complex)
        
        for chan in range(nchan):
            
            dumreal = cor_selected_vis[dumselect, ifn, chan, 0, 0]
            dumimag = cor_selected_vis[dumselect, ifn, chan, 0, 1]
            dumweight = cor_selected_vis[dumselect, ifn, chan, 0, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
                
            cor_rrvis[chan] = dumavg
            
            dumreal = cor_selected_vis[dumselect, ifn, chan, 1, 0]
            dumimag = cor_selected_vis[dumselect, ifn, chan, 1, 1]
            dumweight = cor_selected_vis[dumselect, ifn, chan, 1, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
                
            cor_llvis[chan] = dumavg
            
            dumreal = cor_selected_vis[dumselect, ifn, chan, 2, 0]
            dumimag = cor_selected_vis[dumselect, ifn, chan, 2, 1]
            dumweight = cor_selected_vis[dumselect, ifn, chan, 2, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
            
            cor_rlvis[chan] = dumavg
            
            dumreal = cor_selected_vis[dumselect, ifn, chan, 3, 0]
            dumimag = cor_selected_vis[dumselect, ifn, chan, 3, 1]
            dumweight = cor_selected_vis[dumselect, ifn, chan, 3, 2]
            dum2select = (dumweight > 0.)
            
            if(np.sum(dum2select) > 0):
                dumavg = np.average(dumreal[dum2select] + 1j * dumimag[dum2select], weights = dumweight[dum2select])
            else:
                dumavg = 0j
            
            cor_lrvis[chan] = dumavg
 
        dumx = np.arange(nchan) + 1
        dumdumselect = (np.abs(rlvis) != 0.)
        axs[3, dum].scatter(dumx[dumdumselect], np.degrees(np.angle(cor_lrvis[dumdumselect])), s = 5, facecolor = colorarr[l], edgecolor = colorarr[l], alpha = 0.6)
        axs[4, dum].scatter(dumx[dumdumselect], np.abs(cor_lrvis[dumdumselect]), s = 5, facecolor = colorarr[l], edgecolor = colorarr[l], alpha = 0.6)
        
        
        dum += 1
        
    dum += 1
    

for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        yticks = axs[i,j].get_yticks()
        axs[i,j].set_xlim(axs[i,j].get_xlim())
        axs[i,j].set_ylim(axs[i,j].get_ylim())
        for k in range(len(yticks)-1):
            axs[i,j].plot(np.linspace(axs[i,j].get_xlim()[0], axs[i,j].get_xlim()[1], 100), np.repeat(yticks[k], 100), color = 'black', linewidth = 0.5)


plt.savefig(direc + '/epsfiles/{:}.possm.LR.pdf'.format(inname), bbox_inches='tight')  
plt.savefig(direc + '/epsfiles/{:}.possm.LR.png'.format(inname), bbox_inches='tight')  




