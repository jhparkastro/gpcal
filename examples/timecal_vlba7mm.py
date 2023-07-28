

import gpcal as gp

from multiprocessing import cpu_count



aips_userno = 711


direc = '/home/jpark/gpcalclone/examples/timecal_vlba7mm/'


tdcalname = 'bm462m.q.tdcal.'

outputname = tdcalname

calsour = ['3C273', '3C279', '3C345', '1633+382', '3C454.3', 'CTA102']
source = ['3C273', '3C279', '3C345', '1633+382', '3C454.3', 'CTA102']



ms = 2048
ps = 0.02
uvbin = 0
uvpower = -1
dynam = 1.5
shift_x = 0
shift_y = 0


timecal_freepar = 2
freqavg = True
clean_IF_combine = True
timecal_iter = 10

filetype = 'png'


nproc = cpu_count() - 1

obs = gp.timecal.timecal(aips_userno, direc, tdcalname, calsour, source, timecal_freepar = timecal_freepar, multiproc = True, nproc = nproc, freqavg = freqavg, clean_IF_combine = clean_IF_combine, outputname = outputname, \
                        timecal_iter = timecal_iter, ms = ms, ps = ps, uvbin = uvbin, uvpower = uvpower, shift_x = shift_x, shift_y = shift_y, dynam = dynam, filetype = filetype, init_change_source = False)

obs.dtermsolve()


