# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:32:52 2024

@author: ymu204
"""
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import pandas as pd
import copy

def loop_two_identical_phages(trange=np.arange(0,20,0.01),rbacs=[1.0],kbinds=[1E-8],kbursts=[0.5],nbursts=[100.0],rs=[1E-8],b0_inits=[1E5],b1_inits=[np.nan],b2_inits=[np.nan],b12_inits=[np.nan],I1_inits=[0],I2_inits=[0],p1_inits=[1E4],p2_inits=[1E4]):
    output=pd.DataFrame()
    full_output=pd.DataFrame()
    for b0_init in b0_inits:#There should be a smarter way of looping through all parameter combinations
        for b1_init in b1_inits:
            for b2_init in b2_inits:
                for b12_init in b12_inits:
                    for I1_init in I1_inits:
                        for I2_init in I2_inits:
                            for p1_init in p1_inits:
                                for p2_init in p2_inits:
                                    for rbac in rbacs:
                                        for kbind in kbinds:
                                            for kburst in kbursts:
                                                for nburst in nbursts:
                                                    for r in rs:
                                                        kbind1=copy.copy(kbind)
                                                        kburst1=copy.copy(kburst)
                                                        nburst1=copy.copy(nburst)
                                                        kbind2=copy.copy(kbind)
                                                        kburst2=copy.copy(kburst)
                                                        nburst2=copy.copy(nburst)
                                                        r1=copy.copy(r)
                                                        r2=copy.copy(r)
                                                        if np.isnan(b1_init):
                                                            b1_init=b0_init*r1
                                                        if np.isnan(b2_init):
                                                            b2_init=b0_init*r2
                                                        if np.isnan(b12_init):
                                                            b12_init=b0_init*r1*r2
                                                        t_col1,bac_col1=get_peak_one_phage(trange=trange,b_init=b0_init+b2_init,I_init=I1_init,p_init=p1_init,rbac=rbac,kbind=kbind1,kburst=kburst1,nburst=nburst1)[:2]
                                                        t_col2,bac_col2=get_peak_one_phage(trange=trange,b_init=b0_init+b1_init,I_init=I2_init,p_init=p2_init,rbac=rbac,kbind=kbind2,kburst=kburst2,nburst=nburst2)[:2]
                                                        if ~np.isnan(t_col1) and ~np.isnan(t_col2):
                                                            two_phage_output=solve_dalldt_two_phage(trange=np.arange(0,np.max([t_col1,t_col2])*10.0,np.min([t_col1,t_col2])/100.0),rbac=rbac,kbind1=kbind1,kbind2=kbind2,kburst1=kburst1,kburst2=kburst2,nburst1=nburst1,nburst2=nburst2,r1=r1,r2=r2,b0_init=b0_init,b1_init=b1_init,b2_init=b2_init,b12_init=b12_init,I1_init=I1_init,I2_init=I2_init,p1_init=p1_init,p2_init=p2_init)
                                                            ##something about multipeak height filtering
                                                            all_bacs_t_peaks,all_bacs_peaks=get_peaks(two_phage_output['all_bac'].values[0],two_phage_output['t'].values,two_phage_output['all_bac'].values)
                                                            b1_t_peaks,b1_peaks=get_peaks(two_phage_output['b1'].values[0],two_phage_output['t'].values,two_phage_output['b1'].values)
                                                            b2_t_peaks,b2_peaks=get_peaks(two_phage_output['b2'].values[0],two_phage_output['t'].values,two_phage_output['b2'].values)
                                                            #if len(b1_t_peaks)==1 and len(b2_t_peaks)!=1:
                                                            num_peaks=len(all_bacs_t_peaks)
                                                            local_output=pd.DataFrame({'num_peaks':num_peaks,'t_col1':t_col1,'t_col2':t_col2,'bac_col1':bac_col1,'bac_col2':bac_col2,'all_baccol':np.max(two_phage_output['all_bac']),'b12':np.max(two_phage_output['b12']),'rbac':rbac,'kbind1':kbind1,'kbind2':kbind2,'kburst1':kburst1,'kburst2':kburst2,'nburst1':nburst1,'nburst2':nburst2,'r1':r1,'r2':r2,'b0_init':b0_init,'b1_init':b1_init,'b2_init':b2_init,'b12_init':b12_init,'I1_init':I1_init,'I2_init':I2_init,'p1_init':p1_init,'p2_init':p2_init},index=[0])
                                                            output=pd.concat([output,local_output])
                                                            full_output=pd.concat([full_output,two_phage_output])
                                                            
                                                            
                                                            #else:
                                                            #    local_output=pd.DataFrame({'num_peaks':np.nan,'t_col1':t_col1,'t_col2':t_col2,'bac_col1':bac_col1,'bac_col2':bac_col2,'all_baccol':np.nan,'b12':np.nan,'rbac':rbac,'kbind1':kbind1,'kbind2':kbind2,'kburst1':kburst1,'kburst2':kburst2,'nburst1':nburst1,'nburst2':nburst2,'r1':r1,'r2':r2,'b0_init':b0_init,'b1_init':b1_init,'b2_init':b2_init,'b12_init':b12_init,'I1_init':I1_init,'I2_init':I2_init,'p1_init':p1_init,'p2_init':p2_init},index=[0])
                                                            #    output=pd.concat([output,local_output])
                                                        else:
                                                            local_output=pd.DataFrame({'num_peaks':np.nan,'t_col1':t_col1,'t_col2':t_col2,'bac_col1':bac_col1,'bac_col2':bac_col2,'all_baccol':np.nan,'b12':np.nan,'rbac':rbac,'kbind1':kbind1,'kbind2':kbind2,'kburst1':kburst1,'kburst2':kburst2,'nburst1':nburst1,'nburst2':nburst2,'r1':r1,'r2':r2,'b0_init':b0_init,'b1_init':b1_init,'b2_init':b2_init,'b12_init':b12_init,'I1_init':I1_init,'I2_init':I2_init,'p1_init':p1_init,'p2_init':p2_init},index=[0])
                                                            output=pd.concat([output,local_output])
    output['delta_t']=output['t_col1']-output['t_col2']
    return output,full_output

def get_peaks(min_value,x_array,y_array):
    peaks, _ = find_peaks(y_array)
    true_peaks = peaks[y_array[peaks] > min_value]
    y_array[true_peaks]
    return x_array[true_peaks],y_array[true_peaks]

def loop_two_different_phages(trange=np.arange(0,20,0.01),rbacs=[1.0],kbind1s=[1E-8],kbind2s=[1E-8],kburst1s=[0.5],kburst2s=[0.5],nburst1s=[100.0],nburst2s=[100.0],r1s=[1E-8],r2s=[1E-8],b0_inits=[1E5],b1_inits=[np.nan],b2_inits=[np.nan],b12_inits=[np.nan],I1_inits=[0],I2_inits=[0],p1_inits=[1E4],p2_inits=[1E4]):
    output=pd.DataFrame()
    full_output=pd.DataFrame()
    for b0_init in b0_inits:
        for b1_init in b1_inits:
            for b2_init in b2_inits:
                for b12_init in b12_inits:
                    for I1_init in I1_inits:
                        for I2_init in I2_inits:
                            for p1_init in p1_inits:
                                for p2_init in p2_inits:
                                    for rbac in rbacs:
                                        for kbind1 in kbind1s:
                                            for kbind2 in kbind2s:
                                                for kburst1 in kburst1s:
                                                    for kburst2 in kburst2s:
                                                        for nburst1 in nburst1s:
                                                            for nburst2 in nburst2s:
                                                                for r1 in r1s:
                                                                    for r2 in r2s:
                                                                        if np.isnan(b1_init):
                                                                            b1_init=b0_init*r1
                                                                        if np.isnan(b2_init):
                                                                            b2_init=b0_init*r2
                                                                        if np.isnan(b12_init):
                                                                            b12_init=b0_init*r1*r2
                                                                        t_col1,bac_col1=get_peak_one_phage(trange=trange,b_init=b0_init+b2_init,I_init=I1_init,p_init=p1_init,rbac=rbac,kbind=kbind1,kburst=kburst1,nburst=nburst1)[:2]
                                                                        t_col2,bac_col2=get_peak_one_phage(trange=trange,b_init=b0_init+b1_init,I_init=I2_init,p_init=p2_init,rbac=rbac,kbind=kbind2,kburst=kburst2,nburst=nburst2)[:2]
                                                                        if not np.isnan(t_col1) or np.isnan(t_col2):
                                                                            two_phage_output=solve_dalldt_two_phage(trange=np.arange(0,(t_col1+t_col2)*10.0,np.min([t_col1,t_col2])/100.0),rbac=rbac,kbind1=kbind1,kbind2=kbind2,kburst1=kburst1,kburst2=kburst2,nburst1=nburst1,nburst2=nburst2,r1=r1,r2=r2,b0_init=b0_init,b1_init=b1_init,b2_init=b2_init,b12_init=b12_init,I1_init=I1_init,I2_init=I2_init,p1_init=p1_init,p2_init=p2_init)
                                                                            local_output=pd.DataFrame({'t_col1':t_col1,'t_col2':t_col2,'bac_col1':bac_col1,'bac_col2':bac_col2,'all_baccol':np.max(two_phage_output['all_bac']),'b12':np.max(two_phage_output['b12']),'rbac':rbac,'kbind1':kbind1,'kbind2':kbind2,'kburst1':kburst1,'kburst2':kburst2,'nburst1':nburst1,'nburst2':nburst2,'r1':r1,'r2':r2,'b0_init':b0_init,'b1_init':b1_init,'b2_init':b2_init,'b12_init':b12_init,'I1_init':I1_init,'I2_init':I2_init,'p1_init':p1_init,'p2_init':p2_init},index=[0])
                                                                            output=pd.concat([output,local_output])
                                                                            
                                                                            full_output=pd.concat([full_output,two_phage_output])
                                                                        else:
                                                                            local_output=pd.DataFrame({'t_col1':t_col1,'t_col2':t_col2,'bac_col1':bac_col1,'bac_col2':bac_col2,'all_baccol':np.nan,'b12':np.nan,'rbac':rbac,'kbind1':kbind1,'kbind2':kbind2,'kburst1':kburst1,'kburst2':kburst2,'nburst1':nburst1,'nburst2':nburst2,'r1':r1,'r2':r2,'b0_init':b0_init,'b1_init':b1_init,'b2_init':b2_init,'b12_init':b12_init,'I1_init':I1_init,'I2_init':I2_init,'p1_init':p1_init,'p2_init':p2_init},index=[0])
                                                                            output=pd.concat([output,local_output])
    output['delta_t']=output['t_col1']-output['t_col2']
    return output,full_output
                                                                        

def solve_dalldt_two_phage(trange=np.arange(0,20,0.01),rbac=1.0,kbind1=1E-8,kbind2=1E-8,kburst1=0.5,kburst2=0.5,nburst1=100.0,nburst2=100.0,r1=1E-8,r2=1E-8,b0_init=1E5,b1_init=np.nan,b2_init=np.nan,b12_init=np.nan,I1_init=0,I2_init=0,p1_init=1E4,p2_init=1E4):
    assert r1<1 and r1>0
    assert r2<2 and r2>0
    if np.isnan(b1_init):
        b1_init=b0_init*r1
    if np.isnan(b2_init):
        b2_init=b0_init*r2
    if np.isnan(b12_init):
        b12_init=b0_init*r1*r2
    
    v=[b0_init,b1_init,b2_init,b12_init,I1_init,I2_init,p1_init,p2_init,rbac,kbind1,kbind2,kburst1,kburst2,nburst1,nburst2,r1,r2]
    res=solve_ivp(dalldt_two_phage,[np.min(trange),np.max(trange)],v,t_eval=trange)#,atol=atol_array,rtol=rtol)
    output=pd.DataFrame()
    output['t']=res['t']
    output['b0']=res['y'][0,:]
    output['b1']=res['y'][1,:]
    output['b2']=res['y'][2,:]
    output['b12']=res['y'][3,:]
    output['I1']=res['y'][4,:]
    output['I2']=res['y'][5,:]
    output['p1']=res['y'][6,:]
    output['p2']=res['y'][7,:]
    output['rbac']=res['y'][8,:]
    output['kbind1']=res['y'][9,:]
    output['kbind2']=res['y'][10,:]
    output['kburst1']=res['y'][11,:]
    output['kburst2']=res['y'][12,:]
    output['nburst1']=res['y'][13,:]
    output['nburst2']=res['y'][14,:]
    output['r1']=res['y'][15,:]
    output['r2']=res['y'][16,:]
    
    output['b0_init']=b0_init
    output['b1_init']=b1_init
    output['b2_init']=b2_init
    output['b12_init']=b12_init
    output['I1_init']=I1_init
    output['I2_init']=I2_init
    output['p1_init']=p1_init
    output['p2_init']=p2_init
    
    output['all_bac']=output['b0']+output['b1']+output['b2']+output['I1']+output['I2']
    output['all_infected']=output['I1']+output['I2']
    output['all_non-infected']=output['b0']+output['b1']+output['b2']
    return output

def dalldt_two_phage(t,v):
    [b0,b1,b2,b12,I1,I2,p1,p2,rbac,kbind1,kbind2,kburst1,kburst2,nburst1,nburst2,r1,r2]=v
    db0dt=b0*(rbac-r1-r2)-p1*b0*kbind1-p2*b0*kbind2
    db1dt=b0*r1+b1*rbac-p2*b1*kbind2
    db2dt=b0*r2+b2*rbac-p1*b2*kbind1
    
    db12dt=b0*r1*r2+b2*r1+b1*r2
    
    
    dp1dt=nburst1*I1*kburst1-p1*(b0+b2)*kbind1
    dI1dt=p1*(b0+b2)*kbind1-I1*kburst1
    
    dp2dt=nburst2*I2*kburst2-p2*(b0+b1)*kbind2
    dI2dt=p2*(b0+b1)*kbind2-I2*kburst2
    return [db0dt,db1dt,db2dt,db12dt,dI1dt,dI2dt,dp1dt,dp2dt,0,0,0,0,0,0,0,0,0]

def dpdt_one_phage(t,v):
    [b,I,p,bres,rbac,kbind,kburst,nburst,r]=v
    dbdt=(rbac-r)*b-p*kbind*b
    dIdt=p*b*kbind-I*kburst
    dpdt=nburst*I*kburst-p*b*kbind
    dbresdt=b*r+bres*rbac
    return [dbdt,dIdt,dpdt,dbresdt,0,0,0,0,0]

def get_peak_one_phage(trange=np.arange(0,20,0.01),b_init=1E5,I_init=0,p_init=1E4,rbac=1.0,kbind=1E-8,kburst=0.5,nburst=100.0,bres_init=0,r=1E-8):
    v0=[b_init,I_init,p_init,bres_init,rbac,kbind,kburst,nburst,r]
    res=solve_ivp(dpdt_one_phage,[np.min(trange),np.max(trange)],v0,t_eval=trange)
    curve=pd.DataFrame({'t':trange,'b':res['y'][0],'I':res['y'][1],'p':res['y'][2],'bres':res['y'][3],'all_bac':res['y'][3]+res['y'][0]})
    t_col=res.t[np.argmax(res.y[0,:]+res.y[1,:])]
    bac_col=np.max(res.y[0,:]+res.y[1,:])
    if t_col==np.max(res.t):
        return np.nan,np.nan
        #return get_peak_one_phage(trange*100.0,v0)
    elif np.sum(trange<t_col)<10:
        return np.nan,np.nan
        #return get_peak_one_phage(trange/100.0,v0)
    else:
        return t_col,bac_col,curve
    
def loop_one_phage(trange=np.arange(0,20,0.01),p_inits=[1E4],b_inits=[1E5],I_init=0,rbac=1.0,kbinds=[1E-8],kbursts=[0.5],nbursts=[100]):
    peaks=pd.DataFrame()
    curves=pd.DataFrame()
    for p_init in p_inits:
        for b_init in b_inits:
            for kbind in kbinds:
                for kburst in kbursts:
                    for nburst in nbursts:
                        t_col,bac_col,curve=get_peak_one_phage(trange=trange,b_init=b_init,I_init=I_init,p_init=p_init,rbac=rbac,kbind=kbind,kburst=kburst,nburst=nburst)
                        local_output=pd.DataFrame({'t_col':t_col,'bac_col':bac_col,'p_init':p_init,'b_init':b_init,'kbind':kbind,'kburst':kburst,'nburst':nburst},index=[0])
                        peaks=pd.concat([peaks,local_output])
                        curve['p_init']=p_init
                        curve['b_init']=b_init
                        curve['kbind']=kbind
                        curve['kburst']=kburst
                        curve['nburst']=nburst
                        curves=pd.concat([curves,curve])
    return peaks,curves



def plot_colors(x,y,c,cmap=plt.get_cmap('seismic')):#sns.diverging_palette(250, 15, center="dark", as_cmap=True)
    ncol=len(np.unique(c))
    norm = plt.Normalize(min(c),max(c))
    for i in range(ncol):
        col=np.unique(c)[i]
        loc_x=x[c==col]
        loc_y=y[c==col]
        plt.plot(loc_x,loc_y, color=cmap(norm(col)))    
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #colorbar = plt.colorbar(sm)
    #colorbar.set_label('c')
    
generate_data=False
load_data=False
plot_data=False

if generate_data:
    single_peaks,single_curves=loop_one_phage(trange=np.arange(0,50,0.01),p_inits=[1E4],b_inits=[1E5],I_init=0,rbac=1.0,kbinds=[1E-8],kbursts=[0.5],nbursts=np.logspace(1,3,20))
    dual_peaks,dual_curves=loop_two_different_phages(nburst1s=np.logspace(1,3,20))
    dual_curves['s1']=dual_curves['p1']*dual_curves['kbind2']/dual_curves['rbac']
    dual_curves['s2']=dual_curves['p2']*dual_curves['kbind2']/dual_curves['rbac']
    dual_peaks['p_resistance']=1-np.exp(-dual_peaks['b12'])
    
    single_curves.to_csv('data/single_curves.csv',index=False)
    dual_curves.to_csv('data/dual_curves.csv',index=False)
    dual_peaks.to_csv('data/dual_peaks.csv',index=False)
    
    ###Basel_phages
    all_data=pd.read_csv('Basel_collapse_times.csv')
    times_array=np.zeros([len(np.unique(all_data['col_num']))*len(np.unique(all_data['row_num'])),len(np.unique(all_data['plate']))])
    counter_plate=-1
    for plate in np.unique(all_data['plate']):
        counter_plate+=1
        counter_well=-1
        for col in np.unique(all_data['col_num']):
            for row in np.unique(all_data['row_num']):
                counter_well+=1
                t_col_a=all_data[np.logical_and(all_data['col_num']==col,np.logical_and(all_data['row_num']==row,all_data['plate']==plate))]['t_col'].values
                assert len(t_col_a)==1
                times_array[counter_well,counter_plate]=t_col_a[0]
    
    times_array=times_array[:69,:]
    times_array=times_array[np.sum(~np.isnan(times_array),axis=1)>0,:]
    
    delta_times_array=np.zeros([np.shape(times_array)[0],np.shape(times_array)[0],np.shape(times_array)[1]])
    for p in range(np.shape(times_array)[1]):
        for i in range(np.shape(times_array)[0]):
            for j in range(np.shape(times_array)[0]):
                delta_times_array[i,j,p]=times_array[i,p]-times_array[j,p]
                
    thr=0.5
    no_match=np.sum(np.abs(delta_times_array)>thr,axis=2)
    yes_match=np.sum(np.abs(delta_times_array)<thr,axis=2)
    dunno=np.sum(np.isnan(delta_times_array),axis=2)
    
    all_match=np.logical_and(yes_match>0,no_match==0)
    np.savetxt('data/all_match.csv',all_match,delimiter=',')
    N=np.shape(all_match)[0]
    z=(np.sum(all_match)-N)/2
    n=np.arange(1,N)
    p=1-(1-2*z/(N*(N-1)))**(n*(n-1)/2)
    cocktail_size=pd.DataFrame({'p':p,'n':n})
    cocktail_size.to_csv('data/cocktail_size.csv',index=False)
    
    kbind,_=loop_two_identical_phages(p1_inits=np.logspace(1,7,20),kbind=np.logspace(-7,-9,20))
    kburst,_=loop_two_identical_phages(p1_inits=np.logspace(1,7,20),kbursts=np.logspace(np.log10(0.5),np.log10(50),20))
    nburst,_=loop_two_identical_phages(p1_inits=np.logspace(1,7,20),nbursts=np.logspace(np.log10(50),np.log10(5000),20))
    rbac,_=loop_two_identical_phages(p1_inits=np.logspace(1,7,20),rbacs=np.logspace(np.log10(0.2),np.log10(2),20))
    p2,_=loop_two_identical_phages(p1_inits=np.logspace(1,7,20),p2_inits=np.logspace(3,5,20))
    b0,_=loop_two_identical_phages(p1_inits=np.logspace(1,7,20),b0_inits=np.logspace(4,6,20))
    
    kbind.to_csv('data/vary_kbind.csv',index=False)
    kburst.to_csv('data/vary_kburst.csv',index=False)
    nburst.to_csv('data/vary_nburst.csv',index=False)
    rbac.to_csv('data/vary_rbac.csv',index=False)
    p2.to_csv('data/vary_p2.csv',index=False)
    b0.to_csv('data/vary_b0.csv',index=False)
    

if load_data:
    single_curves=pd.read_csv('data/single_curves.csv')
    dual_curves=pd.read_csv('data/dual_curves.csv')
    dual_peaks=pd.read_csv('data/dual_peaks.csv')
    cocktail_size=pd.read_csv('data/cocktail_size.csv')

    kbind=pd.read_csv('data/vary_kbind.csv')
    kburst=pd.read_csv('data/vary_kburst.csv')
    nburst=pd.read_csv('data/vary_nburst.csv')
    rbac=pd.read_csv('data/vary_rbac.csv')
    p2=pd.read_csv('data/vary_p2.csv')
    b0=pd.read_csv('data/vary_b0.csv')

if plot_data:
    w=15.93/2.54
    h=15.93/2.54/3
    fig=plt.figure(figsize=[w,h*2])
    plt.subplot(2,3,1)
    plot_colors(single_curves['t'],single_curves['all_bac'],np.log10(single_curves['nburst']),cmap=sns.diverging_palette(250, 15, center="dark", as_cmap=True))
    plt.yscale('log')
    # plt.xlabel('Time (h)')
    plt.ylabel('#Bacteria')
    plt.xlim(0,50)
    plt.ylim(1,3E9)
    plt.text(12,1E1,'Single phage')
    plt.yticks([1E0,1E3,1E6,1E9])
    plt.xticks([0,20,50],['','',''])
    
    plt.subplot(2,3,2)
    plot_colors(dual_curves['t'],dual_curves['all_bac'],np.log10(dual_curves['nburst1']/dual_curves['nburst2']),cmap=sns.diverging_palette(250, 15, center="dark", as_cmap=True))
    plt.yscale('log')
    #plt.xlabel('Time (h)')
    #plt.ylabel('#Bacteria')
    plt.xlim(0,50)
    plt.ylim(1,3E9)
    plt.xticks([0,25,50],['','',''])
    plt.yticks([1E0,1E3,1E6,1E9],['','','',''])
    plt.text(2,1E1,'Dual phage')
     
    plt.subplot(2,3,4)
    plot_colors(dual_curves['t'],dual_curves['s1'],np.log10(dual_curves['nburst1']/dual_curves['nburst2']),cmap=sns.diverging_palette(250, 15, center="dark", as_cmap=True))
    plt.yscale('log')
    plt.xlabel('Time (h)')
    plt.ylabel('Phage selection pressure')
    plt.xlim(0,50)
    plt.axhline(1,c='k',linestyle='--')
    plt.yticks([1E-4,1E-2,1E0,1E2])
    plt.text(10,1E-4,'Phage 1')
     
    plt.subplot(2,3,5)
    plot_colors(dual_curves['t'],dual_curves['s2'],np.log10(dual_curves['nburst1']/dual_curves['nburst2']),cmap=sns.diverging_palette(250, 15, center="dark", as_cmap=True))
    plt.yscale('log')
    plt.xlabel('Time (h)')
    #plt.ylabel('#Selecation phage 2')
    plt.xlim(0,50)
    plt.axhline(1,c='k',linestyle='--')
    plt.yticks([1E-4,1E-2,1E0,1E2],['','','',''])
    plt.text(10,1E-4,'Phage 2')
     
    plt.subplot(2,3,3)
    plt.scatter(dual_peaks['nburst1']/100,(1-np.exp(-dual_peaks['b12']))*100,c=np.log10(dual_peaks['nburst1']/dual_peaks['nburst2']),cmap=sns.diverging_palette(250, 15, center="dark", as_cmap=True))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Burst size ratio')
    plt.ylabel('Dual-resistance %')
    plt.axvspan(np.min(dual_peaks['nburst1'])/100,0.5,color='b',alpha=0.1)
    plt.axvspan(2,np.max(dual_peaks['nburst1'])/100,color='r',alpha=0.1)
    plt.xlim(np.min(dual_peaks['nburst1'])/100,np.max(dual_peaks['nburst1'])/100)
    plt.xticks([0.1,1,10],['0.1','1','10'])
    plt.text(np.sqrt(np.min(dual_peaks['nburst1'])/100*0.5),1E-2,'Phage 2 first',rotation=90,ha='center', va='center',c='b')
    plt.text(np.sqrt(2*np.max(dual_peaks['nburst1']/100)),1E-2,'Phage 1 first',rotation=90,ha='center', va='center',c='r')
    plt.text(np.sqrt(0.5*2),1E-1, 'Simultaneous \n pressure',ha='center', va='center',rotation=90)
    
    plt.subplot(2,3,6)
    plt.scatter(cocktail_size['n'],cocktail_size['p']*100,color='k')
    plt.xlabel('#Phages in cocktail')
    plt.ylabel('Simultaneous selection %')
    plt.ylim(0,100)
    plt.xlim(0,14)
    plt.xticks([0,5,10])
    plt.yticks([0,50,100])
    
    plt.tight_layout()
    
    plt.gcf().text(0.01, 0.95, 'a')
    plt.gcf().text(1/3, 0.95, 'b')
    plt.gcf().text(0.01, 0.5, 'c')
    plt.gcf().text(1/3, 0.5, 'd')
    plt.gcf().text(2/3, 0.95, 'e')
    plt.gcf().text(2/3, 0.5, 'f')
    
    plt.savefig('results_fig.pdf',transparent=True)
    
    
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    # Create a colormap
    cmap=sns.diverging_palette(250, 15, center="dark", as_cmap=True)
    norm = plt.Normalize(vmin=1, vmax=3)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb.set_label('Burst size')
    plt.xticks([1,2,3],['10','100','1000'])
    plt.show()
    plt.savefig('Colorbar.pdf',transparent=True)
    
    w=15.93/2.54
    h=15.93/2.54/3
    fig=plt.figure(figsize=[w,h*2])
    plt.subplot(2,3,1)
    plot_colors(kbind['delta_t'],(1-np.exp(-kbind['b12']))*100,np.log10(kbind['kbind1']),cmap=sns.color_palette("rocket", as_cmap=True))
    plt.yscale('log')
    plt.xlim(-2.5,2.5)
    plt.ylim(1E-7,1E2)
    #plt.ylabel('Simultaneous selection %')
    #plt.xlabel('$\Delta$t$_\mathrm{collapse}$ (h)')
    plt.yticks([1E-7,1E-4,1E-1,1E2])
    plt.xticks([-2,0,2],['','',''])
    plt.text(-2,1E-6,'k$_\mathrm{bind}$')
    
    plt.subplot(2,3,2)
    plot_colors(kburst['delta_t'],(1-np.exp(-kburst['b12']))*100,np.log10(kburst['kburst1']),cmap=sns.color_palette("rocket", as_cmap=True))
    plt.yscale('log')
    plt.xlim(-2.5,2.5)
    plt.ylim(1E-7,1E2)
    #plt.ylabel('Simultaneous selection %')
    #plt.xlabel('$\Delta$t$_\mathrm{collapse}$ (h)')
    plt.yticks([1E-7,1E-4,1E-1,1E2],['','','',''])
    plt.xticks([-2,0,2],['','',''])
    plt.text(-2,1E-6,'k$_\mathrm{burst}$')
    
    plt.subplot(2,3,3)
    plot_colors(nburst['delta_t'],(1-np.exp(-nburst['b12']))*100,np.log10(nburst['nburst1']),cmap=sns.color_palette("rocket", as_cmap=True))
    plt.yscale('log')
    plt.xlim(-2.5,2.5)
    plt.ylim(1E-7,1E2)
    #plt.ylabel('Simultaneous selection %')
    #plt.xlabel('$\Delta$t$_\mathrm{collapse}$ (h)')
    plt.yticks([1E-7,1E-4,1E-1,1E2],['','','',''])
    plt.xticks([-2,0,2],['','',''])
    plt.text(-2,1E-6,'n$_\mathrm{burst}$')
    
    plt.subplot(2,3,4)
    plot_colors(b0['delta_t'],(1-np.exp(-b0['b12']))*100,np.log10(b0['b0_init']),cmap=sns.color_palette("rocket", as_cmap=True))
    plt.yscale('log')
    plt.xlim(-2.5,2.5)
    plt.ylim(1E-7,1E2)
    #plt.ylabel('Simultaneous selection %')
    plt.xlabel('$\Delta$t$_\mathrm{collapse}$ (h)')
    plt.yticks([1E-7,1E-4,1E-1,1E2])
    plt.xticks([-2,0,2])
    plt.text(-2,1E-6,'b$_\mathrm{init}$')
    
    plt.subplot(2,3,5)
    plot_colors(p2['delta_t'],(1-np.exp(-p2['b12']))*100,np.log10(p2['p1_init']),cmap=sns.color_palette("rocket", as_cmap=True))
    plt.yscale('log')
    plt.xlim(-2.5,2.5)
    plt.ylim(1E-7,1E2)
    #plt.ylabel('Simultaneous selection %')
    plt.xlabel('$\Delta$t$_\mathrm{collapse}$ (h)')
    plt.yticks([1E-7,1E-4,1E-1,1E2],['','','',''])
    plt.xticks([-2,0,2])
    plt.text(-2,1E-6,'p$_\mathrm{init}$')
    
    plt.subplot(2,3,6)
    plot_colors(rbac['delta_t'],(1-np.exp(-rbac['b12']))*100,np.log10(rbac['rbac']),cmap=sns.color_palette("rocket", as_cmap=True))
    plt.yscale('log')
    plt.xlim(-2.5,2.5)
    plt.ylim(1E-7,1E2)
    #plt.ylabel('Simultaneous selection %')
    plt.xlabel('$\Delta$t$_\mathrm{collapse}$ (h)')
    plt.yticks([1E-7,1E-4,1E-1,1E2],['','','',''])
    plt.xticks([-2,0,2])
    plt.text(-2,1E-6,'$\mu$')
    
    plt.tight_layout()
    plt.gcf().text(0.015,0.55,'Simultaneous selection %',rotation=90,ha='center', va='center')
    
    plt.savefig('suppl_results_fig.pdf',transparent=True)
    plt.savefig('suppl_results_fig.png',transparent=True,dpi=3000)