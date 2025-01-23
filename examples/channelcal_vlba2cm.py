
import os
import subprocess

import gpcal as gp
import gpcal.aipsutil as au

from multiprocessing import cpu_count

from AIPSData import AIPSUVData


# AIPS user ID number for ParselTongue.
aips_userno = 99

# Working directory
direc = '/home/jpark/gpcal_mpifr_tutorial_check/gpcal/examples/channelcal_vlba2cm/'

# We will download the uvfits file from the shared Dropbox folder. The file size is large and it could not be uploaded to the Github repository.
uvfits_url = 'https://www.dropbox.com/scl/fi/hsqrm7vzr4vnhzf25wnzr/bl273ak.u.avspc.uvfits?rlkey=drfc5cm8zfu51ylkx57df3ivj&dl=0'
uvfits_file = direc + 'bl273ak.u.avspc.uvfits'

if not os.path.exists(uvfits_file):
    
    curdirec = os.getcwd()
    
    os.chdir(direc)
    
    p = subprocess.Popen('wget -O {:} {:}'.format(uvfits_file, uvfits_url), stdout=subprocess.PIPE, shell=True)
    
    (output, err) = p.communicate()  
    
    p_status = p.wait()
    
    print("Command output: " + output)
    
    os.chdir(curdirec)


# AIPS catalog Mapname
inname = 'bl273ak.u'

# AIPS catalog Class
inclass = 'AVSPC'

# AIPS catalog Seq number
inseq = 1

# AIPS catalog Disk number
indisk = 1

# We will use CL 1
clver = 1

# If the data we want to load onto AIPS already exists, let us delete it.
data = AIPSUVData(inname, inclass, indisk, inseq)
if(data.exists() == True):
    data.clrstat()
    data.zap()

# Loading the data into AIPS.
au.runfitld(inname, inclass, uvfits_file)

# The data is not self-calibrated, so there are residual gains in the data. Let us conduct self-calibration to correct for the gains.
doselfcal = True

# We will do EVPA calibration to correct for the RCP - LCP phase difference of the reference antenna existing in the data.
doevpacal = True

# Data name
dataname = 'bl273ak.u.channel'

# The list of calibrators that will be used for frequency-dependent leakage estimation.
polcalsour = ['0851+202', '1226+023', '1253-055']

# If True, GPCAL will assume that the source is unpolarized. Each element of the list corresponds to the one in the "polcalsour" list.
polcal_unpol = [False, False, False]

# The list of sources that the derived frequency-dependent leakage will be applied.
source = ['0415+379', '0430+052', '0851+202', '1222+216', '1226+023', '1253-055', '1641+399', '1803+784', '1807+698', '2200+420']

# The list of UVFITS files for "polcalsour", which will be used for deriving Stokes Q and U models with CLEAN. These models will be used for deriving the frequency-dependent polarimetric leakages.
channel_calsour_uvf = ["{:}.edt.pol.iter10.{:}.pcal.uvf".format(inname, it) for it in polcalsour]

# The list of CLEAN window files for "polcalsour". These files will be used for deriving Stokes Q and U models with CLEAN.
channel_calsour_win = ["{:}.edt.{:}.win".format(inname, it) for it in polcalsour]

# The list of Stokes I CLEAN models for "polcalsour". These models will be used for Stokes I self-calibration of the sources in "polcalsour"
channel_calsour_models = ["{:}.edt.{:}.mod".format(inname, it) for it in polcalsour]

# channel_source_uvf = ["{:}.edt.pol.iter10.{:}.pcal.uvf".format(inname, it) for it in source]
# channel_source_win = ["{:}.edt.{:}.win".format(inname, it) for it in source]

# The list of Stokes I CLEAN models for "source". These models will be used for Stokes I self-calibration of the sources in "source"
channel_source_models = ["{:}.edt.{:}.mod".format(inname, it) for it in source]

# The EVPA calibration values derived by referring to the MOJAVE calibrated data.
clcorprm = [-91.824, -104.386, -106.949, -99.412] # BL273AK, U, from 0851+202 and 1253-055

# We will make CLEAN models for each IF separately.
clean_IF_combine = False


# Mapsize for CLEAN in Difmap.
ms = 2048

# Pixelsize for CLEAN in Difmap.
ps = [0.04, 0.08, 0.04]

# Uvbin for CLEAN in Difmap.
uvbin = 0

# Uv power-law index for CLEAN in Difmap.
uvpower = -1

# Perform CLEAN until the peak intensity within the CLEAN windows reach 1.5 times the map rms-noise.
dynam = 1.5

# Solution interval for self-calibration.
solint = 0.2 / 60.

# Use multiple cores to speed up the code.
multiproc = True

# Use 40% of the cores for multiprocessing
nproc = int(cpu_count() * 0.4)


# Load the GPCAL class channelcal using the above parameters.             
obs = gp.channelcal.channelcal(aips_userno, direc, dataname, polcalsour, source, inname = inname, inclass = inclass, inseq = inseq, indisk = indisk, clver = clver, clcorprm = clcorprm, \
                ms = ms, ps = ps, uvbin = uvbin, uvpower = uvpower, dynam = dynam, \
                channel_calsour_models = channel_calsour_models, \
                channel_calsour_uvf = channel_calsour_uvf, channel_calsour_win = channel_calsour_win, clean_IF_combine = clean_IF_combine, \
                channel_source_models = channel_source_models, solint = solint, doselfcal = doselfcal, doevpacal = doevpacal, \
                multiproc = multiproc, nproc = nproc)

# Run GPCAL.
obs.dtermsolve()
    

