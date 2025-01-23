
# After running GPCAL on the BL229AE data set, you may run this script to get the correction factors for the EVPA calibration.

import os
import numpy as np
import pandas as pd


script_direc = '/home/jpark/gpcal_mpifr_tutorial_check/gpcal/examples/'

direc = '/home/jpark/gpcal_mpifr_tutorial_check/gpcal/examples/vlba_2cm/'

os.system('cp {:}GPCAL_Difmap_pinal_v1 {:}'.format(script_direc, direc))


# Computed by performing CLEAN on the BL229AE uvfits files downloaded from the MOJAVE website.
evpa_0851 = -40.01 / 2.
evpa_1226 = -100.418 / 2.
evpa_2200 = 41.74 / 2.


Sourcen = ['0851+202', '1226+023', '2200+420']
Source = [it + '.' for it in Sourcen]

Session = 'bl229ae.u.edt.'

Freq = 'u'

IFNum = 8


curdirec = os.getcwd()

ms = 2048
ps = 0.08
uvbin = 0
uvpower = -1
dynam = 1.5

selfpoliter = 10

for l in range(len(Source)):
    for i in range(IFNum):
        data = Session+'pol.iter{:d}.{:s}.dtcal.uvf'.format(selfpoliter,Sourcen[l])
        mask = Session+'{:s}.win'.format(Sourcen[l])
        save = Session+'pol.iter{:d}.{:s}.dtcal'.format(selfpoliter,Sourcen[l])
        
        os.chdir(direc)
        command = "echo @GPCAL_Difmap_pinal_v1 %s,%s,%s,%s,%s,%s,%s,%s,%s | difmap " %(data,i+1,ms,ps,uvbin,uvpower,mask,dynam,save)
        os.system(command)

os.chdir(curdirec)


Tot_evpacor = np.zeros((IFNum, len(Source)))

for l in range(len(Source)):
    if(Source[l] == '0851+202.'):realevpa = np.repeat(evpa_0851, IFNum)
    if(Source[l] == '1226+023.'):realevpa = np.repeat(evpa_1226, IFNum)
    if(Source[l] == '2200+420.'):realevpa = np.repeat(evpa_2200, IFNum)
    
    
    evpaarr = []
    for i in range(IFNum):
        qread = pd.read_csv(direc+Session+'pol.iter{:d}.{:s}.dtcal'.format(selfpoliter,Sourcen[l])+'.IF'+str(i+1)+'.qflux', header = None, skiprows=0)
        qflux = qread[0][0]
        uread = pd.read_csv(direc+Session+'pol.iter{:d}.{:s}.dtcal'.format(selfpoliter,Sourcen[l])+'.IF'+str(i+1)+'.uflux', header = None, skiprows=0)
        uflux = uread[0][0]
        evpa = np.degrees(np.angle(qflux + 1j*uflux))
        evpaarr.append(realevpa[i] * 2. - evpa)
    
    evpaarr = np.array(evpaarr)
    
    print(Source[l].replace('.', ': ') + ', '.join('{:5.3f}'.format(it) for it in evpaarr))
    
    Tot_evpacor[:, l] = evpaarr


for ifn in range(IFNum):
    unwrap_evpacor = np.degrees(np.unwrap(np.radians(Tot_evpacor[ifn,:])))
    
    Tot_evpacor[ifn,:] = unwrap_evpacor


mean_evpacor = np.mean(Tot_evpacor, axis = 1)
std_evpacor = np.std(Tot_evpacor, axis = 1)

print('clcorprm = ' + ', '.join('{:5.3f}'.format(it) for it in mean_evpacor))
print('clcorprmerr = ' + ', '.join('{:5.3f}'.format(it) for it in std_evpacor))

