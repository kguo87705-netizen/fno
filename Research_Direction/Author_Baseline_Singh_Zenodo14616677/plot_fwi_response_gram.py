"""Observation mismatch and six-direction response diagnostic figure."""
from pathlib import Path
import json,torch,numpy as np,matplotlib.pyplot as plt
root=Path('.build/author-fwi-response-gram-v2-20261008');r=json.loads((root/'report.json').read_text());s=torch.load(root/'responses.pt',weights_only=True,map_location='cpu');names=r['direction_names']
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
t=np.arange(1,2001)*.03;axes[0,0].plot(t,s['observation'][0,12],label='Released observation');axes[0,0].plot(t,s['truth_prediction'][0,12],label='Truth forward, same stencil',alpha=.8);axes[0,0].set(title='Case 1, shot 0, receiver 12',xlabel='Time (us)',ylabel='Signed wave amplitude');axes[0,0].legend(fontsize=8)
vals=[r['directions'][n]['derivative_checks'][0]['response_relative_to_observation'] for n in names];axes[0,1].bar(names,vals);axes[0,1].set(yscale='log',title='Local response per unit relative material parameter',ylabel='||Jv|| / ||released observation||');axes[0,1].tick_params(axis='x',rotation=25)
im=axes[1,0].imshow(s['correlation'],vmin=-1,vmax=1,cmap='coolwarm');axes[1,0].set(title='Normalized response correlation',xticks=range(6),yticks=range(6),xticklabels=names,yticklabels=names);axes[1,0].tick_params(axis='x',rotation=25);fig.colorbar(im,ax=axes[1,0],shrink=.8)
axes[1,1].plot(range(1,7),r['unit_response_singular_values'],'o-');axes[1,1].set(title='Unit-response singular values (six directions only)',xlabel='Index',ylabel='Singular value',ylim=(0,1.4));axes[1,1].text(.02,.12,'Truth forward vs released observation: 51.55% L2\nAuthor/reference truth forward parity: exact\nSelf-consistent truth gradient: zero\nProjected positive steps mostly clipped at gamma=1',transform=axes[1,1].transAxes,fontsize=9)
fig.savefig(root/'observation_and_response_gram.png',dpi=180);plt.close(fig)
