

import gpcal as gp

from multiprocessing import cpu_count
import timeit


# AIPS user ID number for ParselTongue.
aips_userno = 99


# The working directory where the input UVFITS and image fits files are located.
direc = '/home/jpark/gpcal_mpifr_tutorial_check/gpcal/examples/vlba_7mm/'


# The data name. The input files should have the names like dataname.sourcename.uvf and dataname.sourcename.fits (e.g., bm413i.q.edt.OJ287.uvf).
dataname = 'bm413i.q.edt.'

# The list of calibrators which will be used for an initial D-term estimation using the similarity assumption.  We will assume that 3C 84 is unpolarized.
calsour = ['OJ287', '3C84']

# The list of the number of CLEAN sub-models for calsour.
cnum = [3, 0]

# The list of booleans specifying whether the sub-model division will be done automatically or manually.
autoccedt = [False, True]


# Perform instrumental polarization self-calibraiton.
selfpol = True

# Iterate 10 times of instrumental polarization self-calibraiton.
selfpoliter = 10

# Mapsize for CLEAN in Difmap.
ms = 2048

# Pixelsize for CLEAN in Difmap.
ps = 0.04

# Uvbin for CLEAN in Difmap.
uvbin = 0

# Uv power-law index for CLEAN in Difmap.
uvpower = -1

# Perform CLEAN until the peak intensity within the CLEAN windows reach the map rms-noise.
dynam = 1

# The list of calibrators which will be used for additional D-term estimation using instrumental polarization self-calibration. This list does not have to be the same as calsour.
polcalsour = ['OJ287', '3C273', '3C279', '3C345', '3C454.3', '3C84']

# We will assume that 3C 84 is unpolarized.
polcal_unpol = [False, False, False, False, False, True]

# The list of sources to which the best-fit D-terms will be applied.
source = ['0235+164', '3C84', '3C111', '0420-014', '3C120', '0716+714', 'OJ287', '1156+295', '3C273', '1510-089', '3C345', 'MKN501', '1749+096', 'BLLAC', '3C279', '3C454.3']

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

# Use multiple cores to speed up the code.
multiproc = True

# Use 40% of the cores for multiprocessing
nproc = int(cpu_count() * 0.4)

time1 = timeit.default_timer()

# Load the GPCAL class POLCAL using the above parameters.
obs = gp.gpcal.polcal(aips_userno, direc, dataname, calsour, source, cnum, autoccedt, polcalsour = polcalsour, polcal_unpol = polcal_unpol, ms = ms, ps = ps, uvbin = uvbin, uvpower = uvpower, dynam = dynam, selfpoliter = selfpoliter, \
                      dplot_IFsep = dplot_IFsep, selfcal=selfcal, soltype = soltype, solmode = solmode, solint = solint, weightit = weightit, \
                      vplot=vplot, resplot=resplot, parplot = parplot, selfpol=selfpol, filetype = filetype, multiproc = multiproc, nproc = nproc)

# Run GPCAL.
obs.dtermsolve()


time2 = timeit.default_timer()


# Print time elapsed for different processes.
print('Time elapsed for AIPS-related processes = {:d} seconds.\nTime elapsed for Difmap-related processes = {:d} seconds.\nTime elapsed for GPCAL-related processes = {:d} seconds.\nTotal time elapsed = {:d}'\
      .format(int(round(obs.aipstime)), int(round(obs.difmaptime)), int(round((time2 - time1) - obs.aipstime - obs.difmaptime - obs.gpcaltime)), int(round(time2 - time1))))

