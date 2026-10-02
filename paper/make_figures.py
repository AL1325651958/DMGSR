"""Eight evidence-backed figures. Python; 183-mm width; editable vector text.
Figure contracts: protocol, architecture, skill, reliability, calibration,
heterogeneity, fixed-case fidelity, and actual denoising evolution.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'paper';OUT=P/'figures';OUT.mkdir(exist_ok=True)
sys.path.insert(0,'C:/Users/13256/.agents/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':7,
 'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
 'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
 'figure.facecolor':'white','savefig.facecolor':'white','axes.linewidth':.6})
A=np.load(P/'evidence/aligned_ensembles.npz');B=np.load(P/'evidence/final_ensembles.npz')
R=json.loads((P/'evidence/results.json').read_text());M=pd.read_csv(P/'evidence/monthly_metrics.csv')
N=['Neighbor','Laplacian','Diffusion','UNet','UNet_cal','Fusion']
LABEL={'Neighbor':'Neighbour propagation','Laplacian':'Laplacian smoothing','Diffusion':'Diffusion','UNet':'U-Net', 'UNet_cal':'Calibrated U-Net','Fusion':'Fusion'}
C={'Neighbor':'#aeb7c1','Laplacian':'#788897','Diffusion':'#9282AA','UNet':'#C29B69','UNet_cal':'#A06A35','Fusion':'#18858A'}

def label(ax,s):ax.annotate(s,xy=(0,1),xytext=(-16,8),textcoords='offset points',xycoords='axes fraction',weight='bold',fontsize=9)
def finish(fig,num,slug):
    stem=f'Fig{num:02d}_{slug}'
    require_matplotlib_panel_alignment(fig,json_out=OUT/f'{stem}.alignment.json',strict=True)
    fig.savefig(OUT/f'{stem}.pdf');fig.savefig(OUT/f'{stem}.svg');fig.savefig(OUT/f'{stem}.png',dpi=300)
    plt.close(fig)
def base(n=2,height=3):
    f,ax=plt.subplots(1,n,figsize=(7.2,height),gridspec_kw={'left':.10,'right':.95,'bottom':.23,'top':.84,'wspace':.50})
    return f,np.atleast_1d(ax)

def fig1():
    f,axes=base(2,3.2);ax=axes[0]
    axes[0].get_gridspec().update(bottom=.35)
    im=ax.imshow(A['train_mean'],origin='lower',extent=[100,110,20,30],aspect='auto',cmap='cividis')
    ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)',title='Training-period mean radiation')
    c=f.add_axes([.10,.11,.34,.025]);c.set_label('<colorbar>');f.colorbar(im,cax=c,orientation='horizontal').set_label('Daily mean irradiance (W/m²)',fontsize=6)
    ax=axes[1];ax.broken_barh([(2000,10)],(2.5,.55),facecolors='#637B8A');ax.broken_barh([(2010,1)],(1.5,.55),facecolors='#C29B69');ax.broken_barh([(2011,1)],(.5,.55),facecolors='#18858A')
    ax.set(xlim=(1999.5,2015),ylim=(0,3.5),yticks=[2.775,1.775,.775],yticklabels=['Fit weights','Calibrate','Report'],xticks=[2000,2005,2010,2014],xlabel='Calendar year',title='Chronological development protocol')
    ax.text(2012.1,2.15,'2012–2014:\nnot used here',fontsize=6,color='#536173')
    for ax,s in zip(axes,'ab'):label(ax,s)
    finish(f,1,'data_protocol')

def fig2():
    f,axes=base(1,3.5);ax=axes[0];ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor=color,edgecolor='#8E9AA4',lw=.7))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':.8,'color':'#657582'})
    box(.05,2.15,2.1,1.6,'Observed radiation + mask\nDaily weather\nCoarse monthly reference', '#E7EDF1')
    box(2.85,3.6,2.5,1.4,'Spatial U-Net ensemble\n8 independently fitted members\nConvolutional encoder–decoder','#F0E5D7')
    box(2.85,.75,2.5,1.8,'Masked temporal diffusion\n12 samples; 64 reverse steps\nSolar residual + graph penalties\nConstraint feedback every 8 steps','#EAE4F0')
    box(6.05,2.0,1.65,1.9,'Mean fusion\n+ residual pooling\n+ spread scaling\n2010 calibration','#DDEDEC')
    box(8.35,2.0,1.6,1.9,'2011 report\nPoint errors\nCRPS + coverage\nInterval width','#DDEDEC')
    arrow((2.2,3.4),(2.8,4.3));arrow((2.2,2.5),(2.8,1.65));arrow((5.4,4.3),(6.0,3.45));arrow((5.4,1.65),(6.0,2.5));arrow((7.75,2.95),(8.3,2.95))
    ax.text(5,.05,'Graph structure enters through four-neighbour losses and boundary correction, not learned attention.',ha='center',fontsize=6)
    finish(f,2,'architecture')

def fig3():
    f,axes=base(2,3.6)
    for ax,key,title in zip(axes,['rmse','crps'],['Point reconstruction','Probabilistic score']):
        val=[R['summary'][n][key] for n in N];ax.barh(np.arange(6),val,color=[C[n] for n in N],height=.62)
        ax.set(yticks=range(6),yticklabels=[LABEL[n] for n in N],xlabel=f'{key.upper()} (W/m²)',title=title,xlim=(0,max(val)*1.22));ax.invert_yaxis()
        for j,v in enumerate(val):ax.text(v+.4,j,f'{v:.2f}',va='center',fontsize=6)
    axes[0].get_gridspec().update(left=.20,wspace=.95)
    for ax,s in zip(axes,'ab'):label(ax,s)
    finish(f,3,'matched_skill')

def fig4():
    f,axes=base(2,3.1);nom=np.arange(.1,1,.1);mask=B['missing'].copy();mask[:12]=False;y=B['truth'][mask]
    for n in ['Diffusion','UNet','UNet_cal','Fusion']:
        ss=B[n].transpose(1,0,2,3,4)[:,mask];cov=[]
        for q in nom:
            lo,hi=np.quantile(ss,[(1-q)/2,(1+q)/2],axis=0);cov.append(np.mean((y>=lo)&(y<=hi)))
        axes[0].plot(nom,cov,'-o',ms=3,color=C[n],label=LABEL[n])
        axes[1].scatter(R['summary'][n]['width90'],R['summary'][n]['coverage90'],s=45,c=C[n],label=LABEL[n])
    axes[0].plot([0,1],[0,1],'--',color='#657582',lw=.8)
    axes[0].set(xlabel='Nominal central interval probability',ylabel='Empirical coverage',xlim=(0,1),ylim=(0,1),title='Reliability across interval levels')
    axes[1].axhline(.9,color='#657582',ls='--',lw=.8)
    axes[1].set(xlabel='Mean 90% interval width (W/m²)',ylabel='Empirical 90% coverage',ylim=(.45,1.0),title='Coverage–width trade-off')
    handles,labels=axes[0].get_legend_handles_labels();f.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.005),ncol=4,frameon=False)
    for ax,s in zip(axes,'ab'):label(ax,s)
    finish(f,4,'reliability')

def fig5():
    f,axes=base(2,3.1);grid=pd.read_csv(P/'evidence/fusion_search.csv');tab=grid.pivot(index='g',columns='w',values='objective')
    axes[0].get_gridspec().update(bottom=.35)
    im=axes[0].imshow(tab.values,origin='lower',extent=[-.025,1.025,.375,8.125],aspect='auto',cmap='cividis_r')
    axes[0].scatter(R['fusion_selection']['w'],R['fusion_selection']['g'],marker='x',s=50,c='#D55E00',lw=1.3)
    axes[0].set(xlabel='U-Net mean weight w',ylabel='Spread multiplier γ',title='Fusion calibration objective')
    cb=f.add_axes([.10,.11,.34,.022]);cb.set_label('<colorbar>');f.colorbar(im,cax=cb,orientation='horizontal').set_label('CRPS + 8 × |coverage − 0.9|',fontsize=6)
    ug=pd.read_csv(P/'evidence/unet_search.csv');axes[1].plot(ug.g,ug.objective,color=C['UNet_cal'])
    axes[1].axvline(R['unet_calibration']['g'],ls='--',color='#657582',lw=.8)
    axes[1].set(xlabel='U-Net spread multiplier',ylabel='Calibration objective (W/m²)',title='Matched U-Net spread calibration')
    for ax,s in zip(axes,'ab'):label(ax,s)
    finish(f,5,'calibration_search')

def fig6():
    f,axes=base(2,3.1)
    for n in ['Diffusion','UNet_cal','Fusion']:
        rows=M[(M.method==n)&M.month.str.startswith('2011')]
        axes[0].plot(range(1,13),rows.rmse,'-o',ms=3,label=LABEL[n],color=C[n])
    rows=M[(M.method=='Fusion')&M.month.str.startswith('2011')]
    axes[1].bar(range(1,13),rows.n,color='#AFC5CC')
    axes[0].set(xlabel='Reporting month (2011)',ylabel='Missing-cell RMSE (W/m²)',title='Month-level heterogeneity',xticks=[1,3,5,7,9,12])
    axes[1].set(xlabel='Reporting month (2011)',ylabel='Missing day–grid values',title='Unequal evaluation support',xticks=[1,3,5,7,9,12])
    handles,labels=axes[0].get_legend_handles_labels();f.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.015),ncol=3,frameon=False)
    for ax,s in zip(axes,'ab'):label(ax,s)
    finish(f,6,'monthly_heterogeneity')

def fig7():
    data=np.load(P/'evidence/fixed_case_members.npz');u=data['unet'];d=data['diffusion'];y=data['truth'];w=R['fusion_selection']['w'];g=R['fusion_selection']['g']
    um=u.mean(0,keepdims=True);dm=d.mean(0,keepdims=True);fused=w*um+(1-w)*dm+g*np.concatenate((w*(u-um),(1-w)*(d-dm)))
    np.savez_compressed(P/'evidence/fixed_case_recalibrated.npz',truth=y,unet=u,diffusion=d,fusion=fused,missing=data['missing'])
    fig=plt.figure(figsize=(7.2,4.9));gs=fig.add_gridspec(2,3,left=.10,right=.90,bottom=.18,top=.87,hspace=.65,wspace=.28,height_ratios=[1,1.2])
    fields=[y[11],d.mean(0)[11],fused.mean(0)[11]]
    vmax=np.ceil(max(q.max() for q in fields)/50)*50
    for j,(field,title) in enumerate(zip(fields,['POWER reference','Diffusion mean','Recalibrated fusion mean'])):
        ax=fig.add_subplot(gs[0,j]);im=ax.imshow(field,origin='lower',vmin=0,vmax=vmax,cmap='cividis',aspect='auto',interpolation='nearest')
        ax.add_patch(Rectangle((2.5,2.5),4,4,fill=False,ec='#D55E00',lw=.8,ls='--'));ax.set(xticks=[],yticks=[],title=title);label(ax,'abc'[j])
    cb=fig.add_axes([.925,.608,.013,.255]);cb.set_label('<colorbar>');fig.colorbar(im,cax=cb).set_label('Radiation (W/m²)',fontsize=6)
    ax=fig.add_subplot(gs[1,:]);days=np.arange(9,16);ax.plot(days,y[8:15,3:7,3:7].mean((1,2)),'-o',color='#202D42',ms=3,label='POWER reference')
    for s,n in [(d,'Diffusion'),(u,'UNet'),(fused,'Fusion')]:
        ts=s[:,8:15,3:7,3:7].mean((2,3));ax.plot(days,ts.mean(0),'-o',ms=3,color=C[n],label=LABEL[n])
        if n=='Fusion':
            lo,hi=np.quantile(ts,[.05,.95],axis=0);ax.fill_between(days,lo,hi,color=C[n],alpha=.16,label='Fusion 90% interval')
    ax.set(xlabel='Day of July 2011',ylabel='Gap-block mean (W/m²)',title='Fixed seven-day illustration, separate from the reporting masks',xticks=days);label(ax,'d')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.30),ncol=3,frameon=False)
    finish(fig,7,'spatiotemporal_case')

def fig8():
    a=np.load(P/'evidence/diffusion_trace.npz');s=a['stages'];y=a['truth'];miss=a['missing'];means=s.mean(1)
    rmse=np.sqrt(((means-y)[:,miss]**2).mean(1));cov=[]
    for ss in s:
        lo,hi=np.quantile(ss[:,miss],[.05,.95],axis=0);cov.append(np.mean((y[miss]>=lo)&(y[miss]<=hi)))
    f,axes=base(2,3.2)
    axes[0].plot(range(8),rmse[:8],'-o',color=C['Diffusion'],ms=3);axes[0].plot([7,8],rmse[7:],'--o',color='#D55E00',ms=3)
    axes[1].plot(range(8),cov[:8],'-o',color=C['Diffusion'],ms=3);axes[1].plot([7,8],cov[7:],'--o',color='#D55E00',ms=3);axes[1].axhline(.9,ls=':',color='#657582')
    for ax in axes:ax.set(xticks=range(9),xticklabels=['8','16','24','32','40','48','56','64','Post'],xlabel='Completed reverse steps / postprocessing')
    axes[0].set(ylabel='Missing-cell RMSE (W/m²)',title='Reconstruction does not improve monotonically')
    axes[1].set(ylabel='Empirical 90% coverage',ylim=(0,1),title='Final constraints can reduce ensemble coverage')
    for ax,s in zip(axes,'ab'):label(ax,s)
    finish(f,8,'diffusion_trajectory')

if __name__=='__main__':
    for fn in [fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8]:fn();print(fn.__name__,'done',flush=True)
    # Python/Pillow contact sheet for whole-bundle visual QA, not a scientific figure.
    files=sorted(OUT.glob('Fig*.png'));thumbs=[]
    for p in files:
        im=Image.open(p).convert('RGB');im.thumbnail((900,630));tile=Image.new('RGB',(920,660),'white');tile.paste(im,((920-im.width)//2,(660-im.height)//2));thumbs.append(tile)
    sheet=Image.new('RGB',(1840,2640),'#e7ebef')
    for j,im in enumerate(thumbs):sheet.paste(im,((j%2)*920,(j//2)*660))
    sheet.save(OUT/'all_figures_preview.png')
