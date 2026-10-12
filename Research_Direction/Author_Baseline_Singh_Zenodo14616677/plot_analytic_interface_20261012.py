from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

root=Path(__file__).resolve().parents[2]
out=root/'.build/analytic-interface-20261012'
z=np.load(out/'traces.npz');t=z['times']*1e6
names=['Binary','Arithmetic area','Reciprocal mix'];colors=['#777777','#d16b26','#2474ad']
fig,axs=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
ax=axs[0,0];ax.plot(t,z['exact'][1],color='black',lw=2,label='Analytic')
for j in range(3):ax.plot(t,z['traces'][2,j,1],color=colors[j],ls=[':', '--','-.'][j],label=names[j])
ax.set(xlim=(9,11),ylim=(-.02,.37),xlabel='Time (microseconds)',ylabel='Displacement',title='Reflected wave: 4x grid, shifted interface');ax.legend(fontsize=8)
ax=axs[0,1];ax.plot(t,z['exact_shift_difference'],color='black',lw=2,label='Analytic difference')
for j in range(3):ax.plot(t,z['traces'][0,j,1]-z['traces'][0,j,0],color=colors[j],ls=[':', '--','-.'][j],label=names[j])
ax.set(xlim=(9,11),xlabel='Time (microseconds)',ylabel='Shifted minus base',title='Small-shift response: original grid');ax.legend(fontsize=8)
for j in range(3):
 axs[1,0].loglog([1,2,4],100*z['reflection_errors'][:,j].max(axis=-1),'o-',color=colors[j],label=names[j])
 axs[1,1].loglog([1,2,4],100*z['difference_errors'][:,j],'o-',color=colors[j],label=names[j])
for ax in axs[1]:
 ax.set_xticks([1,2,4],labels=['1','2','4']);ax.xaxis.set_minor_locator(NullLocator());ax.set_xlabel('Spatial and temporal refinement factor');ax.grid(alpha=.25,which='both');ax.legend(fontsize=8)
axs[1,0].set(ylabel='Worst-phase reflected-wave error (%)',title='Error against independent analytic solution')
axs[1,1].set(ylabel='Small-shift response error (%)',title='Base-wave agreement does not imply derivative accuracy')
axs[1,1].axhline(10,color='black',ls=':',alpha=.5);axs[1,1].text(1.03,11,'Finest-grid screen: 10%',fontsize=8)
fig.suptitle('1D sharp-interface benchmark | fixed two phases | no inverse fitting',fontsize=13)
fig.savefig(out/'comparison.png',dpi=170)
plt.close(fig)
print(out/'comparison.png')
