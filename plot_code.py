#plot ERPs based on eventprobs/resampling 
#     - New VERSION (only on P, with BRPs) What about 8th panel? Can always use this for legends)
############################################

#resample
import scipy
import seaborn as sns
import matplotlib.font_manager as font_manager
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from scipy.stats import sem
import numpy as np

epoch_data = []
estConds0 = []
hmp_model = []

#resp-locked
data_resp = epoch_data.to_array().squeeze()[:,:,:,:500] #max 1 sec
data_resp.values = np.zeros(data_resp.shape) * np.nan

#set every trial
for pp in data_resp.participant:
    for tr in data_resp.epochs:
       tmp = epoch_data.sel(participant=pp, epochs=tr).data.values
       tmp = tmp[:,~np.isnan(tmp[0,:])]
       if tmp.shape[1] < 500: # add nans at the start
           tmp = np.concatenate((np.tile(np.nan,(tmp.shape[0],500-tmp.shape[1])),tmp),axis=1)
       
       #find nan idx, go there back 500
       data_resp[np.where(data_resp.participant==pp)[0][0],tr, :, :] = tmp[:,-500:]


#resampled
data_resampled = epoch_data.to_array().squeeze()
data_resampled.values = np.zeros(data_resampled.shape) * np.nan

#for each trial and each stage, resample data to average duration
model = estConds0
times = hmp_model.compute_times(hmp_model, model, add_rt=True, fill_value=0, center_measure='median',estimate_method='max').unstack()
times_mean = hmp_model.compute_times(hmp_model, model, duration=False, mean=True, add_rt=True, fill_value=0, extra_dim='condition', center_measure='median',estimate_method='max').values
times_mean = np.round(times_mean)

#go to middle of event
shift = hmp_model.event_width_samples//2
times[1:5,:,:] = times[1:5,:,:] + shift #do not shift onset and RT
times_mean[:,1:5] = times_mean[:,1:5] + shift

for pp in times.participant:
    data_pp = epoch_data.sel(participant=pp).data.values
    times_pp = np.round(times.sel(participant=pp).values)
    for tr in times.trials.values:
        for st in np.arange(times_mean.shape[1]-1)+1:
            if not np.isnan(data_pp[tr,:,:]).all():

                dat = data_pp[tr, :, int(times_pp[st-1,tr]):int(times_pp[st,tr])]
                datt = dat.shape[1]

                #if stage > 0 we resample, in the few other cases it will have nans
                if datt > 0:                

                    #pad extra
                    dat = np.concatenate([np.tile(dat[:,0],(datt,1)).T, dat, np.tile(dat[:,-1],(datt,1)).T],axis=1)

                    c = epoch_data.sel(participant=pp).data[tr,0,0].conditionNew.values
                    cidx = np.where(model.conds_dict[0]['conditionNew'] == c)[0][0]
                    newdur = int(times_mean[cidx,st] - times_mean[cidx,st-1])
                    
                    #resample
                    datnew = scipy.signal.resample_poly(dat, newdur,datt, axis=1, padtype='median')

                    #unpad
                    datnew = datnew[:,(newdur+1):(2*newdur)+1] 

                    data_resampled[np.where(data_resampled.participant==pp)[0][0],tr,:,int(times_mean[cidx,st-1]):int(times_mean[cidx,st])] = datnew

#HMPRPs, not resampled
time_hmprps = 300 #on either side

hmprps = []
for ev in range(4):
    hmprps.append(epoch_data.to_array().squeeze()[:,:,:,:int(time_hmprps)]) #600 ms (-300-300)
    hmprps[-1].values = np.zeros(hmprps[-1].shape) * np.nan

for pp in times.participant:
    data_pp = epoch_data.sel(participant=pp).data.values
    times_pp = np.round(times.sel(participant=pp).values)
    for tr in times.trials.values:
        for ev in range(4):
            if not np.isnan(data_pp[tr,:,:]).all(): #trial x channel x samples

                time_ev = int(times_pp[ev+1,tr])
                dat = data_pp[tr, :, np.max([0,int(time_ev-time_hmprps/2)]):int(time_ev+time_hmprps/2)]
                
                if dat.shape[1] < time_hmprps: #add nans where necessary
                    if time_ev < time_hmprps/2: #add at start
                        dat = np.concatenate((np.tile(np.nan,(tmp.shape[0],int(time_hmprps/2-time_ev))),dat),axis=1)
                    if time_ev > data_pp.shape[2] - time_hmprps/2:
                        dat = np.concatenate((dat,np.tile(np.nan,(tmp.shape[0],int(time_hmprps/2) - (data_pp.shape[2] - time_ev))),),axis=1)

                hmprps[ev][np.where(hmprps[ev].participant==pp)[0][0],tr, :, :] = dat


#general settings

xlims = (0,1000)
ylimsERP = [(-5e-6,5e-6), (-5e-6,5e-6), (-14e-6,14e-6)]

names = ['Stimulus-locked ERP Pz', 'Respond-locked ERP Pz', 'Discovered events', 'Event distributions', 'ERP - Event 1','ERP - Event 2','ERP - Event 3', 'ERP - Event 4', 'ERP - Trial Resampled']
erp_names = names[0:2] + [names[-1]]
hmprp_names = names[4:8]
chans = ['Pz']
time_step = 1000/hmp_model.sfreq

condition_names_idx = ['Aloud', 'SilentCorrect', 'SilentIncorrect', 'New']
condition_names = ['Aloud', 'Silent Correct', 'Silent Incorrect', 'New']
condition_names_rev = condition_names.copy()
condition_names_rev.reverse()

source_font = {'fontname':'Source Sans Pro'}

channel_data = epoch_data.to_array().squeeze() #standard data

plot_data = [channel_data, data_resp, data_resampled]


#create fig and axes
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(19/2.54,30/2.54)) 
grid = plt.GridSpec(5, 4, wspace=0.7, hspace=.45, figure=fig)

ax_ERPs = [] #stim_locked, resp_locked, resampled
ax_ERPs.append(plt.subplot(grid[0,:2]))
ax_ERPs.append(plt.subplot(grid[0,2:]))
ax_ERPs.append(plt.subplot(grid[4,1:3]))

ax_discovered = plt.subplot(grid[1,:2])
ax_distris = plt.subplot(grid[1,2:])

gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec = grid[2,:], wspace = .05)
gs2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec = grid[3,:], wspace = .05)

ax_HMPRPs1 = [plt.subplot(gs1[0,0])]
for i in range(3):
    ax_HMPRPs1.append(plt.subplot(gs1[0,1+i], sharey=ax_HMPRPs1[0], ylim=ylimsERP[0]))
    plt.setp(ax_HMPRPs1[-1].get_yticklabels(), visible=False)
ax_HMPRPs2 = [plt.subplot(gs2[0,0])]
for i in range(3):
    ax_HMPRPs2.append(plt.subplot(gs2[0,1+i], sharey=ax_HMPRPs2[0], ylim=ylimsERP[2]))
    plt.setp(ax_HMPRPs2[-1].get_yticklabels(), visible=False)

    




#plot ERPs: stim locked, resp locked, resampled

for erp in range(3):

    ax_cur =  ax_ERPs[erp]

    # get channels and average across channels
    chans_sel = plot_data[erp][:,:,np.isin(channel_data.channels, chans),:]
    chans_sel = chans_sel.mean('channels')

    #and across condition
    means = []
    ses = []
    for cond in condition_names_idx:
        chans_cond = chans_sel.where(chans_sel.conditionNew==cond, drop=True)
        means.append(chans_cond.groupby('participant').mean('epochs').mean('participant'))
        ses.append(chans_cond.groupby('participant').mean('epochs').reduce(sem, dim='participant',nan_policy='omit'))
    means = np.array(means)
    ses = np.array(ses)

    #0 line
    ax_cur.axhline(y = 0, linestyle = '--',color='lightgrey',linewidth=.5)

    #if resampled plot lines of each event and resp
    if erp == 2:
        for c in range(4):
            ax_cur.vlines(times_mean[c,1:5]*time_step-time_step/2,ylimsERP[erp][0], ylimsERP[erp][1],linestyle = '--',color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][c],alpha=.5,linewidth=.5)
            ax_cur.vlines(times_mean[c,5]*time_step-time_step/2,ylimsERP[erp][0], ylimsERP[erp][1],linestyle = '--',color='darkgrey',alpha=.5,linewidth=.5)
    
    #errors
    for c in range(4):
        ax_cur.fill_between(np.arange(time_step/2, xlims[1], time_step), means[c,:int(np.round(xlims[1]/time_step))] - ses[c,:int(np.round(xlims[1]/time_step))], means[c,:int(np.round(xlims[1]/time_step))] + ses[c,:int(np.round(xlims[1]/time_step))],
                color=['C0','C1', 'C2','C3'][c], alpha=0.2, linewidth=.1)
    
    #means
    ax_cur.plot(np.arange(time_step/2, xlims[1], time_step), means[:,:int(np.round(xlims[1]/time_step))].T,label=condition_names,linewidth=1)
    ax_cur.set_title(erp_names[erp],fontsize=10, fontweight='semibold', **source_font)
    ax_cur.set_ylabel('Voltage',fontsize=9, **source_font, fontweight='light')
    ax_cur.set_xlabel('Time (ms)',fontsize=9, **source_font, fontweight='light')
    ax_cur.tick_params(axis='x', labelsize=8)
    ax_cur.tick_params(axis='y', labelsize=8)
    #ax[fig_idx].set_xticklabels(ax[fig_idx].get_xticklabels(),**source_font, fontweight='light')
    for label in ax_cur.get_xticklabels():
        label.set_fontname('Source Sans Pro') 
        label.set_fontweight('light')
    for label in ax_cur.get_yticklabels():
        label.set_fontname('Source Sans Pro') 
        label.set_fontweight('light')
    ax_cur.yaxis.offsetText.set_fontsize(8)
    ax_cur.yaxis.offsetText.set_fontname('Source Sans Pro') 
    ax_cur.yaxis.offsetText.set_fontweight('light')

    ax_cur.set_ylim(ylimsERP[erp])
    ax_cur.set_xlim(xlims)
    if erp > 0:
        leg = ax_cur.legend(prop=font_manager.FontProperties(family='Source Sans Pro',weight='light', size=8),framealpha=.6, borderpad=.2, labelspacing = .3, handlelength=.6, handletextpad=.4,borderaxespad=0.2)
        leg.get_frame().set_linewidth(0.0)
 
#plot HMPRPs
for ev in range(8):

    if ev < 4:
        ax_cur =  ax_HMPRPs1[ev]
    else:
        ev = ev - 4
        ax_cur =  ax_HMPRPs2[ev]

    # get channels and average across channels
    chans_sel = hmprps[ev][:,:,np.isin(channel_data.channels, chans),:]
    chans_sel = chans_sel.mean('channels')

    #and across condition
    means = []
    ses = []
    for cond in condition_names_idx:
        chans_cond = chans_sel.where(chans_sel.conditionNew==cond, drop=True)
        means.append(chans_cond.groupby('participant').mean('epochs').mean('participant'))
        ses.append(chans_cond.groupby('participant').mean('epochs').reduce(sem, dim='participant',nan_policy='omit'))
    means = np.array(means)
    ses = np.array(ses)

    #0 line
    ax_cur.axhline(y = 0, linestyle = '--',color='lightgrey',linewidth=.5)
    if ev == 0:
        ax_cur.axvline(x = 0, linestyle = '--',color='lightgrey',linewidth=.5)

    #plot lines of each event and resp
    for c in range(4):
        ax_cur.vlines(times_mean[c,ev+1]*time_step-time_step/2,ylimsERP[2][0], ylimsERP[2][1],linestyle = '--',color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][c],alpha=.5,linewidth=.5)
        if ev == 3: #add RT
            ax_cur.vlines(times_mean[c,5]*time_step-time_step/2,ylimsERP[erp][0], ylimsERP[erp][1],linestyle = '--',color='darkgrey',alpha=.5,linewidth=.5)
    
    #errors
    for c in range(4):
        ax_cur.fill_between(np.arange(time_step/2, time_hmprps*2, time_step)-time_hmprps+times_mean[c,ev+1]*time_step-time_step/2, means[c,:] - ses[c,:], means[c,:] + ses[c,:], color=['C0','C1', 'C2','C3'][c], alpha=0.2, linewidth=.1)
    
    #means
    for c in range(4):
        ax_cur.plot(np.arange(time_step/2, time_hmprps*2, time_step)-time_hmprps+times_mean[c,ev+1]*time_step-time_step/2, means[c,:],label=condition_names[c],linewidth=1)
    ax_cur.set_title(hmprp_names[ev],fontsize=10, fontweight='semibold', **source_font)
    if ev == 0:
        ax_cur.set_ylabel('Voltage',fontsize=9, **source_font, fontweight='light')
    ax_cur.set_xlabel('Time (ms)',fontsize=9, **source_font, fontweight='light')
    ax_cur.tick_params(axis='x', labelsize=8)
    ax_cur.tick_params(axis='y', labelsize=8)
    #ax[fig_idx].set_xticklabels(ax[fig_idx].get_xticklabels(),**source_font, fontweight='light')
    for label in ax_cur.get_xticklabels():
        label.set_fontname('Source Sans Pro') 
        label.set_fontweight('light')
    for label in ax_cur.get_yticklabels():
        label.set_fontname('Source Sans Pro') 
        label.set_fontweight('light')
    ax_cur.yaxis.offsetText.set_fontsize(8)
    ax_cur.yaxis.offsetText.set_fontname('Source Sans Pro') 
    ax_cur.yaxis.offsetText.set_fontweight('light')
    ax_cur.yaxis.offsetText.set_text('')

    #ax_HMPRPs1[-1].get_yaxis().get_major_formatter().set_useOffset(False)


    #ax_cur.set_ylim(ylimsERP[erp])
    xmid = np.median(times_mean[:,ev+1])*time_step
    ax_cur.set_xlim((xmid-time_hmprps-50,xmid+time_hmprps+50))
    

#plot discovered events

ax_cur = ax_discovered
hmp.visu.plot_topo_timecourse(epoch_data, estConds0, info, hmp_model, magnify=1.2, sensors=False, as_time=True,contours=0, title="Neutral condition model",center_measure='median',estimate_method='max',ax=ax_cur, vmin=ylimsERP[-1][0],vmax=ylimsERP[-1][1])

ax_cur.set_title(names[2],fontsize=10, fontweight='semibold', **source_font)
ax_cur.set_ylabel('',fontsize=18)
ax_cur.set_xlabel('Time (ms)',fontsize=9, **source_font, fontweight='light')
ax_cur.tick_params(axis='x', labelsize=8)
ax_cur.tick_params(axis='y', labelsize=9)
ax_cur.set_yticklabels(labels=condition_names_rev,**source_font, fontweight='light')
#ax[fig_idx].set_ylim((0,5))
ax_cur.set_xlim(xlims)
for label in ax_cur.get_xticklabels():
    label.set_fontname('Source Sans Pro') 
    label.set_fontweight('light')

#adjust
for child in ax_cur.get_children():
    if type(child) == matplotlib.collections.LineCollection:
        child.set(linewidth=.5)
        
        #change color response line
        if np.allclose(child.get_ec()[0], np.array([0.12156863, 0.46666667, 0.70588235, 1.])):
            child.set_ec('darkgrey')

    #topo (or colorbar?)
    if type(child) == matplotlib.axes._axes.Axes:
        
        if child.get_ylabel() == 'Voltage': #colorbar

            child.tick_params(axis='y', labelsize=8)
            for label in child.get_yticklabels():
                label.set_fontname('Source Sans Pro') 
                label.set_fontweight('light')
            child.yaxis.offsetText.set_fontsize(8)
            child.yaxis.offsetText.set_fontname('Source Sans Pro') 
            child.yaxis.offsetText.set_fontweight('light')
            child.set_ylabel('',fontsize=9, **source_font, fontweight='light')
            ip = InsetPosition(ax_cur,[.86,.6,.02,.3])
            child.set_axes_locator(ip)
        else: #topo
            for child2 in child.get_children():
                if type(child2) == matplotlib.lines.Line2D:
                    child2.set(linewidth=.5) 

#plot distributions of events
                    
ax_cur = ax_distris

densities = []

spaces = np.array([1, .66, .39, .18, 0])
heights = np.diff(spaces) * -1

for ev in range(4):
    
    #events
    #subax_evs = ax[fig_idx].inset_axes([0,.75-.25*ev,1,.25],sharex=ax[fig_idx])
    subax_evs = ax_cur.inset_axes([0,spaces[ev+1],1,heights[ev]],sharex=ax_cur)

    for c in range(4):
        times_ev4 = times.sel(event=ev+1)
        times_ev4_c = times_ev4.stack(trial_x_participant=('participant','trials')).dropna('trial_x_participant')
        times_ev4_c = times_ev4_c[times_ev4_c.cond==c]

        #plot events
        subax_evs.vlines(times_ev4_c.values*time_step-time_step/2,.725-.15*c, .625-.15*c,linestyle = '-',color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][c],alpha=.08, linewidth=.5)

        #add median
        subax_evs.vlines(times_mean[c,ev+1]*time_step-time_step/2,.745-.15*c, .615-.15*c,linestyle = '-',color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][c],alpha=1,linewidth=1.0)

    subax_evs.set_ylim((0,1))
    subax_evs.axis('off')

    #densities

    #subax_dens = ax[fig_idx].inset_axes([0,.75-.25*ev,1,.25],sharex=ax[fig_idx])
    subax_dens = ax_cur.inset_axes([0,spaces[ev+1],1,heights[ev]],sharex=ax_cur)

    for c in range(4):
        times_ev4 = times.sel(event=ev+1)
        times_ev4_c = times_ev4.stack(trial_x_participant=('participant','trials')).dropna('trial_x_participant')
        times_ev4_c = times_ev4_c[times_ev4_c.cond==c]
        #plot densities
        #calc density
        densities.append(scipy.stats.gaussian_kde(times_ev4_c.values*time_step-time_step/2).evaluate(np.arange(0,xlims[1])))
        subax_dens.plot(densities[-1],color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][c],linewidth=1.0,label=condition_names[c])

    #subax.set_xlim(xlims)
    subax_dens.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        left=False,
        right=False,
        labelleft=False) # labels along the bottom edge are off
    subax_dens.set_ylabel('')
    #subax_dens.set_ylim((0, np.max(densities)*.8))
    subax_dens.set_ylim((0, np.array(np.max(densities)*1.05*heights/heights[0])[ev])) #heights/heights[0]
    subax_dens.axis('off')

for yval in spaces[1:4]: #[.25,.5,.75]:
    ax_cur.axhline(y = yval, color='grey',linewidth=.5)

ax_cur.set_title(names[3],fontsize=10, fontweight='semibold', **source_font)
ax_cur.set_xlabel('Time (ms)',fontsize=9, **source_font, fontweight='light')
ax_cur.tick_params(axis='x', labelsize=8)
ax_cur.tick_params(axis='y', labelsize=9)
#ax[fig_idx].set_yticks(np.array([0, .25, .5, .75])+.125, labels=['Event 4','Event 3','Event 2','Event 1'])
ax_cur.set_yticks(spaces[0:4]-heights/2, labels=['Event 1','Event 2','Event 3','Event 4'],**source_font, fontweight='light')
for label in ax_cur.get_xticklabels():
    label.set_fontname('Source Sans Pro') 
    label.set_fontweight('light')
ax_cur.set_xlim(xlims)

#switch off 1e-5 on ev 2-4, move it on ev1
#make x-axes all 500 wide or so
#remove space in between


#fig.tight_layout()
plt.savefig('HMP-ERPs.pdf',dpi=300,transparent=True,bbox_inches='tight',backend='cairo')