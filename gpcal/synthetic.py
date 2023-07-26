
import numpy as np
import pandas as pd

from AIPSData import AIPSUVData, AIPSImage
from Wizardry.AIPSData import AIPSUVData as WAIPSUVData

import aipsutil as au
import obshelpers as oh

import os

from IPython import embed


def synthetic_deq(nant, pang1, pang2, ant1, ant2, llamp, llphas, rramp, rrphas, model_ireal, model_iimag, model_rlreal, model_rlimag, model_lrreal, model_lrimag, stokes, *p):
    
    RiRj_Real = np.zeros(len(pang1))
    RiRj_Imag = np.zeros(len(pang1))
    LiLj_Real = np.zeros(len(pang1))
    LiLj_Imag = np.zeros(len(pang1))
    
    RiLj_Real = np.zeros(len(pang1))
    RiLj_Imag = np.zeros(len(pang1))
    LiRj_Real = np.zeros(len(pang1))
    LiRj_Imag = np.zeros(len(pang1))
    
    model_iamp = np.abs(model_ireal + 1j * model_iimag)
    model_iphas = np.angle(model_ireal + 1j * model_iimag)
    model_rlamp = np.abs(model_rlreal + 1j * model_rlimag)
    model_rlphas = np.angle(model_rlreal + 1j * model_rlimag)
    model_lramp = np.abs(model_lrreal + 1j * model_lrimag)
    model_lrphas = np.angle(model_lrreal + 1j * model_lrimag)
    
    dump = np.array(p)
    
    dumreal = dump[2*ant1]
    dumimag = dump[2*ant1 + 1]
    Tot_D_iR_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_iR_phas = np.angle(dumreal + 1j*dumimag)

    dumreal = dump[2*nant + 2*ant2]
    dumimag = dump[2*nant + 2*ant2 + 1]
    Tot_D_jL_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_jL_phas = np.angle(dumreal + 1j*dumimag)
    
    dumreal = dump[2*nant + 2*ant1]
    dumimag = dump[2*nant + 2*ant1 + 1]
    Tot_D_iL_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_iL_phas = np.angle(dumreal + 1j*dumimag)
    
    dumreal = dump[2*ant2]
    dumimag = dump[2*ant2 + 1]
    Tot_D_jR_amp = np.absolute(dumreal + 1j * dumimag)
    Tot_D_jR_phas = np.angle(dumreal + 1j*dumimag)
    
    
    RiRj_Real += model_ireal + Tot_D_iR_amp * Tot_D_jR_amp * model_iamp * np.cos(model_iphas + Tot_D_iR_phas - Tot_D_jR_phas + 2. * (pang1 - pang2))
    RiRj_Imag += model_iimag + Tot_D_iR_amp * Tot_D_jR_amp * model_iamp * np.sin(model_iphas + Tot_D_iR_phas - Tot_D_jR_phas + 2. * (pang1 - pang2))
    LiLj_Real += model_ireal + Tot_D_iL_amp * Tot_D_jL_amp * model_iamp * np.cos(model_iphas + Tot_D_iL_phas - Tot_D_jL_phas - 2. * (pang1 - pang2))
    LiLj_Imag += model_iimag + Tot_D_iL_amp * Tot_D_jL_amp * model_iamp * np.sin(model_iphas + Tot_D_iL_phas - Tot_D_jL_phas - 2. * (pang1 - pang2))
    
    RiRj_Real += \
        Tot_D_iR_amp * model_lramp * np.cos(Tot_D_iR_phas + model_lrphas + 2. * pang1) + \
        Tot_D_jR_amp * model_rlamp * np.cos(-Tot_D_jR_phas + model_rlphas - 2. * pang2)
    
    RiRj_Imag += \
        Tot_D_iR_amp * model_lramp * np.sin(Tot_D_iR_phas + model_lrphas + 2. * pang1) + \
        Tot_D_jR_amp * model_rlamp * np.sin(-Tot_D_jR_phas + model_rlphas - 2. * pang2)
    
    LiLj_Real += \
        Tot_D_iL_amp * model_rlamp * np.cos(Tot_D_iL_phas + model_rlphas - 2. * pang1) + \
        Tot_D_jL_amp * model_lramp * np.cos(-Tot_D_jL_phas + model_lrphas + 2. * pang2)
    
    LiLj_Imag += \
        Tot_D_iL_amp * model_rlamp * np.sin(Tot_D_iL_phas + model_rlphas - 2. * pang1) + \
        Tot_D_jL_amp * model_lramp * np.sin(-Tot_D_jL_phas + model_lrphas + 2. * pang2)
        
    
    RiLj_Real += model_rlreal + Tot_D_iR_amp * Tot_D_jL_amp * model_lramp * np.cos(model_lrphas + Tot_D_iR_phas - Tot_D_jL_phas + 2. * (pang1 + pang2))
    RiLj_Imag += model_rlimag + Tot_D_iR_amp * Tot_D_jL_amp * model_lramp * np.sin(model_lrphas + Tot_D_iR_phas - Tot_D_jL_phas + 2. * (pang1 + pang2))
    LiRj_Real += model_lrreal + Tot_D_iL_amp * Tot_D_jR_amp * model_rlamp * np.cos(model_rlphas + Tot_D_iL_phas - Tot_D_jR_phas - 2. * (pang1 + pang2))
    LiRj_Imag += model_lrimag + Tot_D_iL_amp * Tot_D_jR_amp * model_rlamp * np.sin(model_rlphas + Tot_D_iL_phas - Tot_D_jR_phas - 2. * (pang1 + pang2))
        
    RiLj_Real += \
      Tot_D_iR_amp * model_iamp * np.cos(Tot_D_iR_phas + model_iphas + 2. * pang1) + \
      Tot_D_jL_amp * model_iamp * np.cos(-Tot_D_jL_phas + model_iphas + 2. * pang2)
    
    RiLj_Imag += \
      Tot_D_iR_amp * model_iamp * np.sin(Tot_D_iR_phas + model_iphas + 2. * pang1) + \
      Tot_D_jL_amp * model_iamp * np.sin(-Tot_D_jL_phas + model_iphas + 2. * pang2)

    LiRj_Real += \
      Tot_D_iL_amp * model_iamp * np.cos(Tot_D_iL_phas + model_iphas - 2. * pang1) + \
      Tot_D_jR_amp * model_iamp * np.cos(-Tot_D_jR_phas + model_iphas - 2. * pang2)

    LiRj_Imag += \
      Tot_D_iL_amp * model_iamp * np.sin(Tot_D_iL_phas + model_iphas - 2. * pang1) + \
      Tot_D_jR_amp * model_iamp * np.sin(-Tot_D_jR_phas + model_iphas - 2. * pang2)  
   
    if(stokes == 'RR'): return RiRj_Real + 1j * RiRj_Imag
    if(stokes == 'LL'): return LiLj_Real + 1j * LiLj_Imag
    if(stokes == 'LR'): return LiRj_Real + 1j * LiRj_Imag
    if(stokes == 'RL'): return RiLj_Real + 1j * RiLj_Imag




def create_synthetic(direc, uvfname, imodelname, qmodelname, umodelname, outputname, dtermfile = None, thermal = True, leakage = True, dtermread = None, \
                     dtermamp = 5e-2, dtermstd = 2e-2, dr_manual = None, dl_manual = None, \
                     time_leakage = False, time_dtermread = None, time_dtermamp = 0.2e-2, time_dtermamp_manual = None, \
                     ms = 2048, ps = 0.1, noisefactor = 1., tsep = 2. / 60):
    
    print("Creating synthetic data...\n\nuvfits file = {:}\n\nStokes I model = {:}\nStokes Q model = {:}\nStokes U model = {:}\nOutput uvfits file = {:}"\
          .format(uvfname, imodelname, qmodelname, umodelname, outputname))
        
    print("\nAdd thermal noise = {:}\nAdd leakage = {:}\nAdd time-dependent leakage = {:}\n".format(thermal, leakage, time_leakage))
        
    
    inname = 'SYN'
    
    au.runfitld(inname, 'EDIT', direc + uvfname)
    
    data = AIPSUVData('SYN', 'EDIT', 1, 1)
    
    antname, antx, anty, antz, antmount, f_par, f_el, phi_off = oh.get_antcoord(data)
    
    nant = len(np.unique(antname))
    
    antname = np.array(antname)
    
    ifnum = data.header['naxis'][3]
    
    obsra = data.header.crval[4]
    obsdec = data.header.crval[5]
    
    lonarr, latarr, heightarr = oh.coord(antname, antx, anty, antz)
    
    year, month, day = oh.get_obsdate(data)
    
    source = data.header['object']
    
    if (dtermread != None):
        read = pd.read_csv(direc + dtermread, delimiter = '\t', index_col = 0)
        
        read_DRArr = np.array(read['DRArr'])
        read_DLArr = np.array(read['DLArr'])
        
        read_DRArr = np.array([complex(it) for it in read_DRArr])
        read_DLArr = np.array([complex(it) for it in read_DLArr])
        
        rand_DR_amp = np.mean(np.abs(np.array(read_DRArr).astype('complex128')))
        rand_DR_std = np.std(np.abs(np.array(read_DRArr).astype('complex128')))
        
        rand_DL_amp = np.mean(np.abs(np.array(read_DLArr).astype('complex128')))
        rand_DL_std = np.std(np.abs(np.array(read_DLArr).astype('complex128')))
    
    else:
        rand_DR_amp = dtermamp
        rand_DR_std = dtermstd
        
        rand_DL_amp = dtermamp
        rand_DL_std = dtermstd
        
    
    total_DRArr, total_DLArr, total_IF, total_ant = [], [], [], []
    
    for ifn in range(ifnum):

        dum_DRamp = np.random.normal(loc = rand_DR_amp, scale = rand_DR_std, size = nant)
        dum_DRArr = dum_DRamp * np.exp(1j * np.radians(np.random.uniform(low = 0, high = 360, size = len(dum_DRamp))))

        dum_DLamp = np.random.normal(loc = rand_DL_amp, scale = rand_DL_std, size = nant)
        dum_DLArr = dum_DLamp * np.exp(1j * np.radians(np.random.uniform(low = 0, high = 360, size = len(dum_DLamp))))
        
        total_DRArr = total_DRArr + dum_DRArr.tolist()
        total_DLArr = total_DLArr + dum_DLArr.tolist()
        total_IF = total_IF + [ifn+1] * len(dum_DRArr)
        total_ant = total_ant + range(0, nant)
    
    
    total_DRArr = np.array(total_DRArr)
    total_DLArr = np.array(total_DLArr)
    total_IF = np.array(total_IF)
    total_ant = np.array(total_ant)
    
    
    df = pd.DataFrame(total_DRArr.transpose())
    df['IF'] = total_IF
    df['antennas'] = total_ant
    df['DRArr'] = total_DRArr
    df['DLArr'] = total_DLArr
    del df[0]
    
    
    df.to_csv(direc + '{:s}.onaxis.dterm'.format(uvfname), sep = "\t")
    
    

    f = open(direc + 'GPCAL_Difmap_uvsub','w')
    f.write('observe %1\nselect i\nmapsize %2, %3\nrmodel %4\nsave %5\nrmodel %6\nsave %7\nexit')
    f.close()

    curdirec = os.getcwd()
    
    os.chdir(direc)
    command = "echo @GPCAL_Difmap_uvsub %s,%s,%s,%s,%s,%s,%s | difmap" %(uvfname, ms, ps, qmodelname, qmodelname.replace('.mod', '.uvsub'), umodelname, umodelname.replace('.mod', '.uvsub'))
    os.system(command)
    
    os.chdir(curdirec)
        
    
    
    total_ant, total_IF, total_scantime, total_scansource, total_DRArr, total_DLArr = [], [], [], [], [], []
    
    
    if (dtermread == None):
        onaxis_read = pd.read_csv(direc + '{:s}.onaxis.dterm'.format(uvfname), delimiter = '\t')
    else:
        onaxis_read = pd.read_csv(direc + dtermread, delimiter = '\t')
        
    onaxis_DRArr = np.array([complex(it) for it in onaxis_read['DRArr']])
    onaxis_DLArr = np.array([complex(it) for it in onaxis_read['DLArr']])
    onaxis_IF = np.array(onaxis_read['IF'])
    onaxis_ant = np.array(onaxis_read['antennas'])
    
    
    
    if(dr_manual != None):
        drkey = list(dr_manual.keys())
        for j in range(len(drkey)):
            onaxis_DRArr[antname[onaxis_ant] == drkey[j]] = dr_manual[drkey[j]]
    
    if(dl_manual != None):
        dlkey = list(dl_manual.keys())
        for j in range(len(dlkey)):
            onaxis_DLArr[antname[onaxis_ant] == dlkey[j]] = dl_manual[dlkey[j]]
    
    
    au.runfitld(inname, 'EDIT', direc + uvfname)
    au.runfitld(inname, 'IMAP', direc + imodelname.replace('.mod', '.fits'))
    au.runfitld(inname, 'QMAP', direc + qmodelname.replace('.mod', '.uvsub.fits'))
    au.runfitld(inname, 'UMAP', direc + umodelname.replace('.mod', '.uvsub.fits'))
    
    uvsub_i = AIPSUVData(inname, 'UVSUB', 1, 1)
    if(uvsub_i.exists() == True):
        uvsub_i.clrstat()
        uvsub_i.zap()
        
    uvsub_q = AIPSUVData(inname, 'UVSUB', 1, 2)
    if(uvsub_q.exists() == True):
        uvsub_q.clrstat()
        uvsub_q.zap()
    
    uvsub_u = AIPSUVData(inname, 'UVSUB', 1, 3)
    if(uvsub_u.exists() == True):
        uvsub_u.clrstat()
        uvsub_u.zap()
    
    au.runuvsub(inname, 'EDIT', 'IMAP', 1, 1)
    au.runuvsub(inname, 'EDIT', 'QMAP', 1, 2)
    au.runuvsub(inname, 'EDIT', 'UMAP', 1, 3)
    
    data = AIPSUVData(inname, 'EDIT', 1, 1)
    uvsub_i = WAIPSUVData(inname, 'UVSUB', 1, 1)
    uvsub_q = WAIPSUVData(inname, 'UVSUB', 1, 2)
    uvsub_u = WAIPSUVData(inname, 'UVSUB', 1, 3)
    
    
    
    data = WAIPSUVData(inname, 'EDIT', 1, 1)
    
    for ifn in range(ifnum):
        
        if leakage:
            dum_DRArr = onaxis_DRArr[onaxis_IF == ifn + 1]
            dum_DLArr = onaxis_DLArr[onaxis_IF == ifn + 1]
            
        else:
            dum_DRArr = np.repeat(0., nant) + 1j * np.repeat(0., nant)
            dum_DLArr = np.repeat(0., nant) + 1j * np.repeat(0., nant)
            
        if thermal:
            noisefactor = noisefactor
        else:
            noisefactor = 1e-12
        

        dumu, dumv, ifarr, time, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        
        for vis in data:
            dumu.append(vis.uvw[0])
            dumv.append(vis.uvw[1])
            ifarr.append(ifn+1)
            time.append(vis.time)
            rrreal.append(vis.visibility[ifn,0,0,0])
            rrimag.append(vis.visibility[ifn,0,0,1])
            rrweight.append(vis.visibility[ifn,0,0,2])
            llreal.append(vis.visibility[ifn,0,1,0])
            llimag.append(vis.visibility[ifn,0,1,1])
            llweight.append(vis.visibility[ifn,0,1,2])
            rlreal.append(vis.visibility[ifn,0,2,0])
            rlimag.append(vis.visibility[ifn,0,2,1])
            rlweight.append(vis.visibility[ifn,0,2,2])
            lrreal.append(vis.visibility[ifn,0,3,0])
            lrimag.append(vis.visibility[ifn,0,3,1])
            lrweight.append(vis.visibility[ifn,0,3,2])
            ant1.append(vis.baseline[0])
            ant2.append(vis.baseline[1])
        
        
        ireal, iimag, qreal, qimag, ureal, uimag = [], [], [], [], [], []
        
        for vis in uvsub_i:
            ireal.append(vis.visibility[ifn,0,0,0])
            iimag.append(vis.visibility[ifn,0,0,1])
            
        for vis in uvsub_q:
            qreal.append(vis.visibility[ifn,0,0,0])
            qimag.append(vis.visibility[ifn,0,0,1])
        
        for vis in uvsub_u:
            ureal.append(vis.visibility[ifn,0,0,0])
            uimag.append(vis.visibility[ifn,0,0,1])
        

        time, dumu, dumv, ifarr, time, ant1, ant2, rrreal, rrimag, rrweight, llreal, llimag, llweight, rlreal, rlimag, rlweight, lrreal, lrimag, lrweight = \
            np.array(time), np.array(dumu), np.array(dumv), np.array(ifarr), np.array(time), np.array(ant1), np.array(ant2), np.array(rrreal), np.array(rrimag), np.array(rrweight), np.array(llreal), np.array(llimag), np.array(llweight), \
            np.array(rlreal), np.array(rlimag), np.array(rlweight), np.array(lrreal), np.array(lrimag), np.array(lrweight)
        
        
        ant1 = ant1.astype('int32')
        ant2 = ant2.astype('int32')
        
        ireal, iimag, qreal, qimag, ureal, uimag = np.array(ireal), np.array(iimag), np.array(qreal), np.array(qimag), np.array(ureal), np.array(uimag)
                        
        mod_q = qreal + 1j * qimag
        mod_u = ureal + 1j * uimag
        
        
        mod_ireal = ireal
        mod_iimag = iimag
        mod_rlreal, mod_rlimag, mod_lrreal, mod_lrimag = np.real(mod_q + 1j*mod_u), np.imag(mod_q + 1j*mod_u), np.real(mod_q - 1j*mod_u), np.imag(mod_q - 1j*mod_u)
        mod_rlamp, mod_rlphas, mod_lramp, mod_lrphas = np.absolute(mod_rlreal + 1j*mod_rlimag), np.angle(mod_rlreal + 1j*mod_rlimag), np.absolute(mod_lrreal + 1j*mod_lrimag), np.angle(mod_lrreal + 1j*mod_lrimag)
        
        
        time = time * 24.
        origtime = np.copy(time)
        ant1 = ant1 - 1
        ant2 = ant2 - 1
        
        
        
        longarr1, latarr1, f_el1, f_par1, phi_off1 = oh.coordarr(lonarr, latarr, f_el, f_par, phi_off, ant1)
        longarr2, latarr2, f_el2, f_par2, phi_off2 = oh.coordarr(lonarr, latarr, f_el, f_par, phi_off, ant2)
        
        sourcearr = np.repeat(source, len(time))
        
        
        boundary_left, boundary_right, boundary_source = oh.get_scans(time, sourcearr, tsep = tsep)
        boundary_time = [(it1 + it2) / 2. for it1, it2 in zip(boundary_left, boundary_right)]
        
        scannum = len(boundary_left)
        
        
        yeararr, montharr, dayarr, raarr, decarr = oh.calendar(sourcearr, [source], [year], [month], [day], [obsra], [obsdec])
        
        for i in range(10):
            dayarr[time>=24.] += 1 
            time[time>=24.] -= 24. 
        
        
        pang1 = oh.get_parang(yeararr, montharr, dayarr, time, raarr, decarr, longarr1, latarr1, f_el1, f_par1, phi_off1)
        pang2 = oh.get_parang(yeararr, montharr, dayarr, time, raarr, decarr, longarr2, latarr2, f_el2, f_par2, phi_off2)
        
        rramp = np.absolute(rrreal + 1j*rrimag)
        rrphas = np.angle(rrreal + 1j*rrimag)
        llamp = np.absolute(llreal + 1j*llimag)
        llphas = np.angle(llreal + 1j*llimag)       
        
        
        
        for s in range(scannum):
            select = (origtime >= boundary_left[s]) & (origtime <= boundary_right[s])
            
            scan_pang1, scan_pang2, scan_ant1, scan_ant2, scan_llamp, scan_llphas, scan_rramp, scan_rrphas, scan_llweight, scan_rrweight, scan_lrweight, scan_rlweight, \
            scan_model_ireal, scan_model_iimag, scan_model_rlreal, scan_model_rlimag, scan_model_lrreal, scan_model_lrimag = \
            pang1[select], pang2[select], ant1[select], ant2[select], llamp[select], llphas[select], rramp[select], rrphas[select], llweight[select], rrweight[select], lrweight[select], rlweight[select], \
            mod_ireal[select], mod_iimag[select], mod_rlreal[select], mod_rlimag[select], mod_lrreal[select], mod_lrimag[select]
            
            if time_leakage:
                
                if (time_dtermread != None):
                    read = pd.read_csv(time_dtermread, delimiter = '\t', index_col = 0)
                    
                    dum_read_IF = read['IF']
                    dum_read_ant = read['antennas']
                    dum_read_scantime = read['scantime']
                    
                    dum_read_DRArr = np.array([complex(it) for it in read['DRArr']])
                    dum_read_DLArr = np.array([complex(it) for it in read['DLArr']])
                    
                    td_DRArr = []
                    td_DLArr = []
                    
                    for i in range(nant):
                        dum_read_select = (dum_read_IF == ifn+1) & (dum_read_ant == antname[i]) & (np.abs(dum_read_scantime - boundary_time[i]) < 1e-4)
                        td_DRArr.append(dum_read_DRArr[dum_read_select][0])
                        td_DLArr.append(dum_read_DLArr[dum_read_select][0])
                        
                          
                    td_DRArr = np.array(td_DRArr)
                    td_DLArr = np.array(td_DLArr)
                
                else:
                    
                    td_DRArr = np.random.normal(loc = 0., scale = time_dtermamp, size = nant) + 1j * np.random.normal(loc = 0., scale = time_dtermamp, size = nant)
                    td_DLArr = np.random.normal(loc = 0., scale = time_dtermamp, size = nant) + 1j * np.random.normal(loc = 0., scale = time_dtermamp, size = nant)
                    
                              
                    if(time_dtermamp_manual != None):
                        key = list(time_dtermamp_manual.keys())
                        for j in range(len(key)):
                            td_DRArr[antname == key[j]] = np.random.normal(loc = 0., scale = time_dtermamp_manual[key[j]]) + 1j * np.random.normal(loc = 0., scale = time_dtermamp_manual[key[j]])
                            td_DLArr[antname == key[j]] = np.random.normal(loc = 0., scale = time_dtermamp_manual[key[j]]) + 1j * np.random.normal(loc = 0., scale = time_dtermamp_manual[key[j]])
                    
            else:
                td_DRArr = np.repeat(0j, nant)
                td_DLArr = np.repeat(0j, nant)
            
        
            td_DRArr += dum_DRArr
            td_DLArr += dum_DLArr
            
            uniq1 = np.unique(scan_ant1)
            uniq2 = np.unique(scan_ant2)
            
            concat = np.concatenate([uniq1, uniq2])
            concatuniq = np.unique(concat)
            
            total_td_DRArr = td_DRArr[concatuniq]
            total_td_DLArr = td_DLArr[concatuniq]

            total_DRArr = total_DRArr + total_td_DRArr.tolist()
            total_DLArr = total_DLArr + total_td_DLArr.tolist()
            total_IF = total_IF + [ifn+1] * len(total_td_DRArr)
            total_ant = total_ant + antname[concatuniq].tolist()
            total_scantime = total_scantime + [boundary_time[s]] * len(total_td_DRArr)
            total_scansource = total_scansource + [source] * len(total_td_DRArr)
        
        
                                
            dumiter = []
            
            for m in range(nant):
                dumiter.append(np.real(td_DRArr[m]))
                dumiter.append(np.imag(td_DRArr[m]))
            
            for n in range(nant):
                dumiter.append(np.real(td_DLArr[n]))
                dumiter.append(np.imag(td_DLArr[n]))

            
            var_dterm_rr = synthetic_deq(nant, scan_pang1, scan_pang2, scan_ant1, scan_ant2, scan_llamp, scan_llphas, scan_rramp, scan_rrphas, scan_model_ireal, scan_model_iimag, scan_model_rlreal, scan_model_rlimag, \
                                   scan_model_lrreal, scan_model_lrimag, 'RR', *dumiter)
            var_dterm_ll = synthetic_deq(nant, scan_pang1, scan_pang2, scan_ant1, scan_ant2, scan_llamp, scan_llphas, scan_rramp, scan_rrphas, scan_model_ireal, scan_model_iimag, scan_model_rlreal, scan_model_rlimag, \
                                   scan_model_lrreal, scan_model_lrimag, 'LL', *dumiter)
            var_dterm_lr = synthetic_deq(nant, scan_pang1, scan_pang2, scan_ant1, scan_ant2, scan_llamp, scan_llphas, scan_rramp, scan_rrphas, scan_model_ireal, scan_model_iimag, scan_model_rlreal, scan_model_rlimag, \
                                   scan_model_lrreal, scan_model_lrimag, 'LR', *dumiter)
            var_dterm_rl = synthetic_deq(nant, scan_pang1, scan_pang2, scan_ant1, scan_ant2, scan_llamp, scan_llphas, scan_rramp, scan_rrphas, scan_model_ireal, scan_model_iimag, scan_model_rlreal, scan_model_rlimag, \
                                   scan_model_lrreal, scan_model_lrimag, 'RL', *dumiter)
            
            
            var_dterm_rr_real = np.real(var_dterm_rr)
            var_dterm_rr_imag = np.imag(var_dterm_rr)
            var_dterm_ll_real = np.real(var_dterm_ll)
            var_dterm_ll_imag = np.imag(var_dterm_ll)
            
            var_dterm_lr_real = np.real(var_dterm_lr)
            var_dterm_lr_imag = np.imag(var_dterm_lr)
            var_dterm_rl_real = np.real(var_dterm_rl)
            var_dterm_rl_imag = np.imag(var_dterm_rl)
            
            scan_rrweight[scan_rrweight <= 0.] = np.median(scan_rrweight)
            scan_llweight[scan_llweight <= 0.] = np.median(scan_llweight)
            scan_lrweight[scan_lrweight <= 0.] = np.median(scan_lrweight)
            scan_rlweight[scan_rlweight <= 0.] = np.median(scan_rlweight)
            
                
            if(np.sum(scan_lrweight > 0.) == 0.) & (np.sum(scan_rlweight > 0.) == 0.):
                continue
            
            var_dterm_rr_real = np.random.normal(loc = var_dterm_rr_real, scale = 1. / np.sqrt(scan_rrweight) * noisefactor)
            var_dterm_rr_imag = np.random.normal(loc = var_dterm_rr_imag, scale = 1. / np.sqrt(scan_rrweight) * noisefactor)
            var_dterm_ll_real = np.random.normal(loc = var_dterm_ll_real, scale = 1. / np.sqrt(scan_llweight) * noisefactor)
            var_dterm_ll_imag = np.random.normal(loc = var_dterm_ll_imag, scale = 1. / np.sqrt(scan_llweight) * noisefactor)
            
            var_dterm_lr_real = np.random.normal(loc = var_dterm_lr_real, scale = 1. / np.sqrt(scan_lrweight) * noisefactor)
            var_dterm_lr_imag = np.random.normal(loc = var_dterm_lr_imag, scale = 1. / np.sqrt(scan_lrweight) * noisefactor)
            var_dterm_rl_real = np.random.normal(loc = var_dterm_rl_real, scale = 1. / np.sqrt(scan_rlweight) * noisefactor)
            var_dterm_rl_imag = np.random.normal(loc = var_dterm_rl_imag, scale = 1. / np.sqrt(scan_rlweight) * noisefactor)
            
            
            dum = 0   
            for vis in data:
                if(vis.time*24. >= boundary_left[s] - 1e-5) & (vis.time*24. <= boundary_right[s] + 1e-5):
                    vis.visibility[ifn,0,0,0] = var_dterm_rr_real[dum]
                    vis.visibility[ifn,0,0,1] = var_dterm_rr_imag[dum]
                    vis.visibility[ifn,0,1,0] = var_dterm_ll_real[dum]
                    vis.visibility[ifn,0,1,1] = var_dterm_ll_imag[dum]
                    vis.visibility[ifn,0,2,0] = var_dterm_rl_real[dum]
                    vis.visibility[ifn,0,2,1] = var_dterm_rl_imag[dum]
                    vis.visibility[ifn,0,3,0] = var_dterm_lr_real[dum]
                    vis.visibility[ifn,0,3,1] = var_dterm_lr_imag[dum]
                    vis.update()
                    vis.update()
                    dum += 1
                    
    
    for vis in data:
        vis.update()
        
    
    data = AIPSUVData(inname, 'EDIT', 1, 1)


    filename = direc + outputname
    au.runfittp(inname, 'EDIT', filename)
    
    
    imap = AIPSImage(inname, 'IMAP', 1, 1)
    qmap = AIPSImage(inname, 'QMAP', 1, 1)
    umap = AIPSImage(inname, 'UMAP', 1, 1)
    
    data.zap()
    imap.zap()
    qmap.zap()
    umap.zap()
    uvsub_i.zap()
    uvsub_q.zap()
    uvsub_u.zap()


    total_ant, total_IF, total_scantime, total_scansource, total_DRArr, total_DLArr = \
        np.array(total_ant), np.array(total_IF), np.array(total_scantime), np.array(total_scansource), np.array(total_DRArr), np.array(total_DLArr)

        
    df = pd.DataFrame(total_DRArr.transpose())
    df['IF'] = total_IF
    df['antennas'] = total_ant
    df['scantime'] = total_scantime
    df['scansource'] = total_scansource
    df['DRArr'] = total_DRArr
    df['DLArr'] = total_DLArr
    del df[0]
    
    if (dtermfile == None):
        df.to_csv(direc + '{:s}.td.dterm.csv'.format(uvfname), sep = "\t")
    else:
        df.to_csv(direc + '{:s}'.format(dtermfile), sep = "\t")
    
    os.system('rm {:}'.format(direc + '*uvsub*'))

