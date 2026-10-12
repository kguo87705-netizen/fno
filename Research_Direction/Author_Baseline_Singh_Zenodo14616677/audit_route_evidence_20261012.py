"""Exploratory raw-data audit; no new physical solves and no altered original claims."""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/".build/tx-design-value-20261012"
OUT=ROOT/".build/route-evidence-audit-20261012"
PAIRS=list(combinations(range(4),2))
KNOTS=np.array([.44,.48,.52,.56])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def fit(observation,bank):
    # Independent implementation returning both the parameter and waveform residual.
    values=[];loss=[]
    for i in range(3):
        delta=bank[i+1]-bank[i]
        distance=observation-bank[i]
        coefficient=np.clip((distance@delta)/(delta@delta),0,1)
        residual=distance-coefficient*delta
        values.append(KNOTS[i]+coefficient*(KNOTS[i+1]-KNOTS[i]))
        loss.append(float(residual@residual))
    j=int(np.argmin(loss))
    return float(values[j]),j,loss[j]


if __name__=="__main__":
    start=time.perf_counter();torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False)
    protocol=json.loads((OLD/"protocol.json").read_text(encoding="utf-8"))
    manifest=json.loads((OLD/"data_manifest.json").read_text(encoding="utf-8"))
    summary=json.loads((OLD/"summary.json").read_text(encoding="utf-8"))
    paths=[OLD/"protocol.json",OLD/"data_manifest.json",OLD/"parameter_estimates.npz",OLD/"summary.json",Path(__file__)]
    dump("audit_plan.json",dict(status="exploratory_reanalysis_before_calculation",input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
         comparisons="all six fixed two-source pairs; original sensitivity policy; noiseless interpolation projection; bias/variance; truth-informed oracle ceiling",
         primary_reanalysis_noise_fraction=.01,physical_calls=0,network_forwards=0,training_updates=0,
         disclosure="Retrospective development-set audit. Best pair and oracle selected using evaluation labels are not deployable or independent test results."))
    for name,h in protocol["input_sha256"].items():assert sha(ROOT/name)==h,name
    for row in manifest["records"]:assert sha(OLD/row["file"])==row["sha256"],row["file"]
    assert sha(OLD/"background.pt")==manifest["background_sha256"]
    bg=torch.load(OLD/"background.pt",weights_only=True).numpy().astype(np.float64)
    scale=np.sqrt(np.mean(bg**2));archive=np.load(OLD/"parameter_estimates.npz",allow_pickle=False)
    rows=[];errors=[];estimates={};projection=[];bias_variance=[];reproduction=[]
    selections={x["geometry"]:x["selected"] for x in summary["action_selections"]}
    for gi,g in enumerate(protocol["geometries"]):
        def load(v):
            return torch.load(OLD/f"{g['name']}__gamma_{v:.2f}.pt",weights_only=True)["traces"].numpy().astype(np.float64)-bg
        bank4=np.stack([load(v) for v in KNOTS])
        banks=[bank4[:,p,:,:].reshape(4,-1) for p in PAIRS]
        for ti,value in enumerate(protocol["evaluation_materials"]):
            truth=load(value);case=np.empty((100,6))
            original=archive[f"{g['name']}_gamma_{value:.2f}_noise_0.01"]
            for seed in range(100):
                rng=np.random.default_rng(np.random.SeedSequence([20261012,gi,ti,1,seed]))
                observed=truth+rng.normal(0,.01*scale,truth.shape)
                for pi,p in enumerate(PAIRS):
                    case[seed,pi]=fit(observed[list(p)].reshape(-1),banks[pi])[0]
            assert np.allclose(case[:,0],original[:,0],rtol=0,atol=2e-12)
            assert np.allclose(case[:,-1],original[:,1],rtol=0,atol=2e-12)
            errors.append(np.abs(case-value));estimates[f"{g['name']}_gamma_{value:.2f}"]=case
            for pi,p in enumerate(PAIRS):
                observed=truth[list(p)].reshape(-1);b=banks[pi]
                estimated,segment,loss=fit(observed,b)
                segment_true=int(np.searchsorted(KNOTS,value)-1)
                d=b[segment_true+1]-b[segment_true]
                weight=(value-KNOTS[segment_true])/(KNOTS[segment_true+1]-KNOTS[segment_true])
                at_truth=b[segment_true]+weight*d
                remainder=observed-at_truth
                predicted_bias=(remainder@d)/(d@d)*(KNOTS[segment_true+1]-KNOTS[segment_true])
                projection.append(dict(geometry=g["name"],gamma=value,pair=list(p),noiseless_signed_error=estimated-value,
                    interpolation_projection_bias=float(predicted_bias),same_segment=segment==segment_true,
                    projection_identity_error=float(abs(estimated-value-predicted_bias)),
                    linearized_noise_std=float(.01*scale/np.sqrt(d@d)*(KNOTS[segment_true+1]-KNOTS[segment_true]))))
                e=case[:,pi]-value
                rows.append(dict(geometry=g["name"],gamma=value,pair=list(p),mae=float(np.abs(e).mean()),
                     signed_bias=float(e.mean()),variance=float(e.var()),mse=float(np.square(e).mean())))
            for ci,policy in enumerate(["fixed_left","fixed_right","physics_selected","full_four_source"]):
                e=original[:,ci]-value
                bias_variance.append(dict(geometry=g["name"],gamma=value,policy=policy,
                     bias_squared=float(e.mean()**2),variance=float(e.var()),mse=float(np.square(e).mean())))
        print("audited "+g["name"],flush=True)
    error=np.stack(errors)  # [18 geometry-material cases,100 seeds,6 pairs]
    fixed=error.mean((0,1));best=int(np.argmin(fixed))
    # Oracle uses labels only to quantify an unattainable optimistic upper bound.
    oracle_per_observation=float(error.min(2).mean())
    oracle_per_case=float(error.mean(1).min(1).mean())
    oracle_geometry=float(error.reshape(6,3,100,6).mean((1,2)).min(1).mean())
    pooled=[]
    for policy in ["fixed_left","fixed_right","physics_selected","full_four_source"]:
        rr=[x for x in bias_variance if x["policy"]==policy]
        bias=float(np.mean([x["bias_squared"] for x in rr]));variance=float(np.mean([x["variance"] for x in rr]))
        pooled.append(dict(policy=policy,mean_mse=bias+variance,bias_squared_fraction=bias/(bias+variance),mean_bias_squared=bias,mean_variance=variance))
    result=dict(status="completed",input_hashes_verified=True,physical_calls=0,network_forwards=0,training_updates=0,
          fixed_pair_metrics=[dict(pair=list(p),mae=float(fixed[i])) for i,p in enumerate(PAIRS)],
          best_fixed_pair_retrospective=list(PAIRS[best]),best_fixed_pair_mae=float(fixed[best]),
          truth_informed_oracle_mae=dict(per_observation=oracle_per_observation,per_geometry_material=oracle_per_case,per_geometry=oracle_geometry),
          oracle_per_geometry_relative_headroom=1-oracle_geometry/float(fixed[best]),
          original_bias_variance=pooled,
          max_noiseless_projection_identity_error=max(x["projection_identity_error"] for x in projection),
          all_estimates_in_true_segments=all(x["same_segment"] for x in projection),
          original_left_right_estimates_reproduced=True,
          provenance_limit="Older R2 cost-boundary folder has only protocol/summary/verification, no saved target/predictions/checkpoint hashes: historical recorded result, not independently waveform-reverified in this audit.",
          elapsed_seconds=time.perf_counter()-start)
    dump("summary.json",result);dump("case_metrics.json",rows);dump("projection_identity.json",projection);dump("bias_variance.json",bias_variance)
    np.savez_compressed(OUT/"six_pair_estimates.npz",**estimates)
    dump("verification.json",dict(status="passed",checks=["original source and all waveform hashes", "original 1800 left/right estimates reproduced within 2e-12", "MSE equals bias squared plus variance", "noiseless error explained by exact segment projection identity"],
        scientific_scope="secondary exploratory audit; no changed original gate; oracle not deployable"))
    assert result["max_noiseless_projection_identity_error"]<1e-10
    for row in rows:assert np.isclose(row["mse"],row["signed_bias"]**2+row["variance"],rtol=1e-12)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
