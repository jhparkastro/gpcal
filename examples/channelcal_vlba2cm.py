
import os
import subprocess

import gpcal as gp
import gpcal.aipsutil as au

from multiprocessing import cpu_count

from AIPSData import AIPSUVData


aips_userno = 711



direc = '/home/jpark/gpcalclone/examples/channelcal_vlba2cm/'


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

    


inname = 'bl273ak.u'
inclass = 'AVSPC'
inseq = 1
indisk = 1

clver = 1


data = AIPSUVData(inname, inclass, indisk, inseq)
if(data.exists() == True):
    data.clrstat()
    data.zap()

au.runfitld(inname, inclass, uvfits_file)


doselfcal = True

selfcal_doclean = True
selfcal_doclean = False

calsour_doclean = True

doevpacal = True

dataname = 'bl273ak.u.channel'
outputname = dataname


polcalsour = ['0851+202', '1226+023', '1253-055']
polcal_unpol = [False, False, False]
source = ['0415+379', '0430+052', '0851+202', '1222+216', '1226+023', '1253-055', '1641+399', '1803+784', '1807+698', '2200+420']

channel_calsour_uvf = ["{:}.edt.pol.iter10.{:}.pcal.uvf".format(inname, it) for it in polcalsour]
channel_calsour_win = ["{:}.edt.{:}.win".format(inname, it) for it in polcalsour]
channel_calsour_models = ["{:}.edt.{:}.mod".format(inname, it) for it in polcalsour]

channel_source_uvf = ["{:}.edt.pol.iter10.{:}.pcal.uvf".format(inname, it) for it in source]
channel_source_win = ["{:}.edt.{:}.win".format(inname, it) for it in source]
channel_source_models = ["{:}.edt.{:}.mod".format(inname, it) for it in source]

clcorprm = [-91.824, -104.386, -106.949, -99.412] # BL273AK, U, from 0851+202 and 1253-055


clean_IF_combine = True
clean_IF_combine = False

ms = 2048
ps = [0.04, 0.08, 0.04]
uvbin = 0
uvpower = -1
dynam = 1.5
shift_x = 0
shift_y = 0

solint = 0.2 / 60.


max_count = cpu_count()
nproc = max_count - 1

             
obs = gp.channelcal.channelcal(aips_userno, direc, dataname, polcalsour, source, inname = inname, inclass = inclass, inseq = inseq, indisk = indisk, clver = clver, clcorprm = clcorprm, outputname = outputname, \
                ms = ms, ps = ps, uvbin = uvbin, uvpower = uvpower, dynam = dynam, shift_x = shift_x, shift_y = shift_y, \
                selfcal_doclean = selfcal_doclean, calsour_doclean = calsour_doclean, channel_calsour_models = channel_calsour_models, \
                channel_calsour_uvf = channel_calsour_uvf, channel_calsour_win = channel_calsour_win, clean_IF_combine = clean_IF_combine, \
                channel_source_uvf = channel_source_uvf, channel_source_models = channel_source_models, channel_source_win = channel_source_win, solint = solint, doselfcal = doselfcal, doevpacal = doevpacal, \
                multiproc = True, nproc = nproc, snver = 0)

obs.dtermsolve()
    

