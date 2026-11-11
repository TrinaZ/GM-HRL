"""Does the Sub-Controller use the Meta-Controller's goal? Validation instances only.
Modes: normal | neutral (goal attribute fixed to 0.5) | frozen (first goal kept all episode) | random (goal ~ N(0,1) at each meta step).
Also: decision agreement between normal and neutral goal along the normal trajectory."""
import sys, json, numpy as np, torch
repo=sys.argv[1]; sys.path.insert(0,repo)
from gmhrl.agent import GMHRLAgent
from gmhrl.generate import write_split
from gmhrl.instance import load_instance
from gmhrl.sim import MD2POSSim, SimConfig
torch.set_num_threads(1)
dom=sys.argv[2]; N=200
paths=write_split(sys.argv[3],dom,N,range(50,55))
sims=[MD2POSSim(load_instance(p),SimConfig(),seed=0) for p in paths]
seeds=[2000,2001,2002,2003]
res={}
for s in [0,1,2]:
    ag=GMHRLAgent.load(f"{repo}/runs_v2/gmhrl_{dom}_N{N}_s{s}/best.pt")
    net=ag.net; om=net.meta_dist; oe=net.encode; osub=net.sub_logits
    for mode in ["normal","neutral","frozen","random"]:
        st={"agree":0,"n":0,"spread":[]}
        net.meta_dist=om; net.encode=oe; net.sub_logits=osub; ag.cfg.hierarchical=True
        if mode=="neutral": ag.cfg.hierarchical=False
        if mode=="frozen":
            mem={}
            def md(sx,z,mem=mem):
                d,v=om(sx,z)
                if "mu" not in mem: mem["mu"]=d.mean.detach()
                return torch.distributions.Normal(mem["mu"],d.stddev),v
            net.meta_dist=md
        if mode=="random":
            def md(sx,z):
                d,v=om(sx,z); return torch.distributions.Normal(torch.randn_like(d.mean),d.stddev),v
            net.meta_dist=md
        if mode=="normal":
            last={}
            def enc(x,ei,ea,g,last=last):
                last.update(x=x,ei=ei,ea=ea,g=g); return oe(x,ei,ea,g)
            def sub(h,z,s_,cand,cw,last=last,st=st):
                out=osub(h,z,s_,cand,cw)
                h2,z2=oe(last["x"],last["ei"],last["ea"],None)
                out2=osub(h2,z2,s_,cand,cw)
                st["agree"]+=int(torch.argmax(out[0])==torch.argmax(out2[0])); st["n"]+=1
                if last["g"] is not None and len(cand)>1:
                    p=net.goal_annotation(last["x"],last["g"])[cand]; st["spread"].append(float(p.max()-p.min()))
                return out
            net.encode=enc; net.sub_logits=sub
        rows=[]
        for sim in sims:
            for sd in seeds:
                if mode=="random": torch.manual_seed(sd)
                m=ag.rollout(sim,sd,train=False)["metrics"]; rows.append((m["makespan"],m["sla_violation_rate"]))
        res[(s,mode)]=(np.array(rows), st)
    net.meta_dist=om; net.encode=oe; net.sub_logits=osub
out={}
for mode in ["normal","neutral","frozen","random"]:
    mk=np.mean([res[(s,mode)][0][:,0] for s in [0,1,2]],axis=0); sl=np.mean([res[(s,mode)][0][:,1] for s in [0,1,2]],axis=0)
    out[mode]=(mk,sl)
base_mk,base_sl=out["normal"]
from scipy.stats import wilcoxon
print(f"== {dom} N=200, validation k=50..54 x 4 seeds, mean over 3 models")
for mode in ["normal","neutral","frozen","random"]:
    mk,sl=out[mode]; d=100*np.mean(mk/base_mk-1)
    p=wilcoxon(mk,base_mk).pvalue if mode!="normal" and not np.allclose(mk,base_mk) else 1.0
    print(f"{mode:8s} makespan {mk.mean():8.1f} ({d:+.2f}% vs normal, p={p:.2g})  SLA {sl.mean():.3f}")
for s in [0,1,2]:
    st=res[(s,"normal")][1]
    print(f"seed {s}: decision agreement normal vs neutral goal = {100*st['agree']/max(st['n'],1):.1f}% of {st['n']} decisions; goal-attribute spread among candidates mean {np.mean(st['spread']):.3f}")
