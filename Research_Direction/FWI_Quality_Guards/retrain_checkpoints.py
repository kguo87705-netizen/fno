"""Replay the original EXP-RT100 training exactly; save extra checkpoints only."""
import random
from common import *

def run(seed):
    random.seed(2);np.random.seed(2);torch.manual_seed(2)
    destination=BUILD/'checkpoints'/f'seed{seed}';destination.mkdir(parents=True,exist_ok=True)
    def save_initial(ns):
        torch.save(ns['model'].state_dict(),destination/'epoch0.pt')
    def checkpoint(ns):
        e=int(ns['epoch'])+1
        if e in [10,30,100]:torch.save(ns['model'].state_dict(),destination/f'epoch{e}.pt')
    code=(SUP/'Pretraining.py').read_text(encoding='utf-8-sig')
    code=code.replace('epochs = 1','epochs = 100').replace('torch.manual_seed(30)',f'torch.manual_seed({seed})')
    data=ROOT/'.build/singh-coverage-paired-20261012/experiment/data'
    code=code.replace('destinationFolder = "data"','destinationFolder = '+repr(str(data)))
    code=code.replace('numberOfSamples = 800','numberOfSamples = 200').replace('random.sample(range(800), numberOfSamples)','random.sample(range(numberOfSamples), numberOfSamples)')
    code=code[:code.index('# Save the pretrained model')]
    code=code.replace('# Training parameters','save_initial(globals())\n\n# Training parameters')
    code=code.replace('    if epoch % 10 == 0:','    checkpoint(globals())\n    if epoch % 10 == 0:')
    ns=dict(__file__=str(AUTHOR/'Pretraining.py'),__name__='__main__',save_initial=save_initial,checkpoint=checkpoint)
    t=time.perf_counter();exec(compile(code,'unchanged_exp_rt100_replay','exec'),ns)
    torch.cuda.synchronize();elapsed=time.perf_counter()-t
    old=ROOT/('.build/singh-coverage-100epochs-20261013/experiment/model_Unet_200' if seed==30 else f'.build/singh-training-seeds-20261013/seed{seed-30}/experiment/model_Unet_200')
    reference=torch.load(old,map_location='cpu',weights_only=True)
    state={k:v.cpu() for k,v in ns['model'].state_dict().items()}
    equal=state.keys()==reference.keys() and all(torch.equal(state[k],reference[k]) for k in state)
    maxdiff=max(float((state[k].double()-reference[k].double()).abs().max()) for k in state)
    row=dict(seed=seed,seconds=elapsed,exact_match=equal,max_abs_difference=maxdiff,reference=str(old.relative_to(ROOT)))
    dump(destination/'reproduction.json',row)
    np.savez(destination/'loss_history.npz',training=ns['trainingCostHistory'],validation=ns['validationCostHistory'])
    print('RETRAIN_RESULT',json.dumps(row),flush=True)
    if not equal:raise RuntimeError('STOP: epoch100 differs from EXP-RT100')
    return row

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
