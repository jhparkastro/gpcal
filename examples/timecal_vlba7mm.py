
import gpcal as gp

import os

from multiprocessing import cpu_count


# AIPS user ID number for ParselTongue.
aips_userno = 99

# The working directory where the input UVFITS and image fits files are located.
direc = '/home/jpark/gpcal_mpifr_tutorial_check/gpcal/examples/timecal_vlba7mm/'

# The data name. The input files should have the names like dataname.sourcename.uvf and dataname.sourcename.fits (e.g., 18ja.d.edt.3C84.uvf).
timecalname = 'bm462m.q.timecal.'

# The list of calibrators which will be used for an initial D-term estimation using the similarity assumption.
calsour = ['3C273', '3C279', '3C345', '1633+382', '3C454.3', 'CTA102']

# The list of sources to which the best-fit D-terms will be applied.
source = ['3C273', '3C279', '3C345', '1633+382', '3C454.3', 'CTA102']


# Mapsize for CLEAN in Difmap.
ms = 2048

# Pixelsize for CLEAN in Difmap.
ps = 0.02

# Uvbin for CLEAN in Difmap.
uvbin = 0

# Uv power-law index for CLEAN in Difmap.
uvpower = -1

# Perform CLEAN until the peak intensity within the CLEAN windows reach the map rms-noise.
dynam = 1.5

# Use multiple cores to speed up the code.
multiproc = True

# Use 40% of the cores for multiprocessing
nproc = int(cpu_count() * 0.4)

# The number of antennas for which time-dependent leakages are corrected in each iteration.
timecal_freepar = 2

# We will average the data in frequency to increase the SNR.
freqavg = True

# Iterate 10 times of time-dependent leakage correction.
timecal_iter = 10

# Output the figures in the format of png.
filetype = 'png'

# What we have the data set after correcting for the stable (on-axis) leakages with GPCAL. Their names are like "bm462m.q.edt.pol.iter10.3C273.pcal.uvf". 
# Let's copy those files into new files with their names following the convention of the timecal class (e.g., "bm462m.q.timecal.3C273.uvf" in this tutorial). We also need to prepare CLEAN window files.
for l in range(len(calsour)):
    print('cp {:} {:}'.format(direc + 'bm462m.q.edt.pol.iter10.{:}.pcal.uvf'.format(calsour[l]), direc + '{:}{:}.uvf'.format(timecalname, calsour[l])))
    os.system('cp {:} {:}'.format(direc + 'bm462m.q.edt.pol.iter10.{:}.pcal.uvf'.format(calsour[l]), direc + '{:}{:}.uvf'.format(timecalname, calsour[l])))
    print('cp {:} {:}'.format(direc + 'bm462m.q.edt.{:}.win'.format(calsour[l]), direc + '{:}{:}.win'.format(timecalname, calsour[l])))
    os.system('cp {:} {:}'.format(direc + 'bm462m.q.edt.{:}.win'.format(calsour[l]), direc + '{:}{:}.win'.format(timecalname, calsour[l])))
    
    
for l in range(len(source)):
    print('cp {:} {:}'.format(direc + 'bm462m.q.edt.pol.iter10.{:}.pcal.uvf'.format(source[l]), direc + '{:}{:}.uvf'.format(timecalname, source[l])))
    os.system('cp {:} {:}'.format(direc + 'bm462m.q.edt.pol.iter10.{:}.pcal.uvf'.format(source[l]), direc + '{:}{:}.uvf'.format(timecalname, source[l])))
    print('cp {:} {:}'.format(direc + 'bm462m.q.edt.{:}.win'.format(source[l]), direc + '{:}{:}.win'.format(timecalname, source[l])))
    os.system('cp {:} {:}'.format(direc + 'bm462m.q.edt.{:}.win'.format(source[l]), direc + '{:}{:}.win'.format(timecalname, source[l])))
    

# Load the GPCAL class timecal using the above parameters.
obs = gp.timecal.timecal(aips_userno, direc, timecalname, calsour, source, timecal_freepar = timecal_freepar, multiproc = multiproc, nproc = nproc, freqavg = freqavg, \
                        timecal_iter = timecal_iter, ms = ms, ps = ps, uvbin = uvbin, uvpower = uvpower, dynam = dynam, filetype = filetype)

# Run GPCAL.
obs.dtermsolve()


