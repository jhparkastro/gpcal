
import gpcal as gp

import timeit

from multiprocessing import cpu_count

multiproc = True
nproc = cpu_count() - 1


# AIPS user ID number for ParselTongue.
aips_userno = 99


# The working directory where the input UVFITS and image fits files are located.
direc = '/home/jpark/gpcaltest/examples/vlba_2cm/'


# The data name. The input files should have the names like dataname.sourcename.uvf and dataname.sourcename.fits (e.g., bl229ae.u.edt.OJ287.uvf).
dataname = 'bl229ae.u.edt.'

# The list of calibrators which will be used for an initial D-term estimation using the similarity assumption.
calsour = ['0851+202', '2258-022']

# The list of the number of CLEAN sub-models for calsour.
cnum = [3, 3]

# The list of booleans specifying whether the sub-model division will be done automatically or manually.
autoccedt = [False, False]


# Perform instrumental polarization self-calibraiton.
selfpol = True

# Iterate 10 times of instrumental polarization self-calibraiton.
selfpoliter = 10

# Mapsize for CLEAN in Difmap.
ms = 2048

# Pixelsize for CLEAN in Difmap.
ps = 0.06

# Uvbin for CLEAN in Difmap.
uvbin = 2

# Uv power-law index for CLEAN in Difmap.
uvpower = -1

# Perform CLEAN until the peak intensity within the CLEAN windows reach the map rms-noise.
dynam = 1

# The list of calibrators which will be used for additional D-term estimation using instrumental polarization self-calibration. This list does not have to be the same as calsour.
polcalsour = ['0415+379', '0430+052', '0502+049', '0851+202', '1151+408', '1637+574', '2200+420', '2201+315', '2258-022']
# polcalsour = ['0430+052']

# The list of sources to which the best-fit D-terms will be applied.
source = ['0256+075', '0300+470', '0415+379', '0430+052', '0502+049', '0851+202', '1151+408', '1226+023', '1637+574', '1928+738', '2200+420', '2201+315', '2258-022']

# Perform additional self-calibration with CALIB in Difmap.
selfcal = True

# CALIB parameters.
soltype = 'L1R'
solmode = 'A&P'
solint = 10./60.
weightit = 1


# Draw vplots, fitting residual plots, and field-rotation angle plots.
vplot = True
resplot = True
parplot = True

# Draw D-term plots for each IF separately.
dplot_IFsep = True

# Output the figures in the format of png.
filetype = 'png'

# The D-term plots will be shown for ranges of (-15%, 15%) for both the real and imaginary parts.
drange = 15.


time1 = timeit.default_timer()

obs = gp.gpcal.polcal(aips_userno, direc, dataname, calsour, source, cnum, autoccedt, polcalsour = polcalsour, ms = ms, ps = ps, uvbin = uvbin, uvpower = uvpower, dynam = dynam, selfpoliter = selfpoliter, \
                      dplot_IFsep = dplot_IFsep, selfcal=selfcal, vplot=vplot, resplot=resplot, parplot = parplot, selfpol=selfpol, filetype = filetype, multiproc = multiproc, nproc = nproc)
    
# Run GPCAL.
obs.dtermsolve()


time2 = timeit.default_timer()


# Print time elapsed for different processes.
print('Time elapsed for AIPS-related processes = {:d} seconds.\nTime elapsed for Difmap-related processes = {:d} seconds.\nTime elapsed for GPCAL-related processes = {:d} seconds.\nTotal time elapsed = {:d}'\
      .format(int(round(obs.aipstime)), int(round(obs.difmaptime)), int(round((time2 - time1) - obs.aipstime - obs.difmaptime - obs.gpcaltime)), int(round(time2 - time1))))

