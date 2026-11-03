# Numbers for the text of Section 6

- main/genomic: GM-HRL vs HEFT by N: N=50 +0.02% (p=1), N=100 -0.77% (p=0.45), N=200 -3.17% (p=2.5e-10), N=500 -2.52% (p=1.3e-10), N=1000 -2.18% (p=1.1e-10)

- main/video: GM-HRL vs HEFT by N: N=50 +0.51% (p=0.14), N=100 +0.27% (p=1), N=200 -2.74% (p=1.4e-07), N=500 -2.27% (p=3.4e-10), N=1000 -1.82% (p=1.5e-10)

- main/ai_ft: GM-HRL vs HEFT by N: N=50 +0.26% (p=0.35), N=100 -0.18% (p=0.78), N=200 +0.01% (p=1), N=500 -0.06% (p=0.4), N=1000 -0.02% (p=1)

- main/financial: GM-HRL vs HEFT by N: N=50 +0.00% (p=1), N=100 -0.06% (p=1), N=200 +0.04% (p=0.37), N=500 +0.28% (p=0.00016), N=1000 +0.03% (p=0.12)

- gap to bound (GM-HRL / HEFT / best): genomic N=50: 5.4/5.4/4.9 (opt 100%); genomic N=100: 17.1/18.1/16.7 (opt 37%); genomic N=200: 13.1/16.8/12.4 (opt 2%); video N=50: 6.4/5.9/5.9 (opt 100%); video N=100: 23.1/22.8/22.8 (opt 2%); video N=200: 14.9/18.3/14.6 (opt 0%); ai_ft N=50: 2.4/2.1/2.1 (opt 98%); ai_ft N=100: 2.9/3.1/2.9 (opt 90%); ai_ft N=200: 4.3/4.3/4.2 (opt 67%); financial N=50: -0.2/-0.2/-0.2 (opt 100%); financial N=100: 0.6/0.6/0.6 (opt 100%); financial N=200: 3.1/3.1/3.1 (opt 67%)

- ablation (genomic N=200): GM-HRL -3.17% vs HEFT, +0.00% vs full (p=1), SLA 33.7%; no_hierarchy -3.78% vs HEFT, -0.62% vs full (p=0.00013), SLA 32.5%; no_gnn -2.92% vs HEFT, +0.26% vs full (p=0.032), SLA 35.0%; no_dynamic_graph -3.39% vs HEFT, -0.21% vs full (p=0.19), SLA 32.1%; kappa0.3 -3.50% vs HEFT, -0.34% vs full (p=0.14), SLA 38.0%; kappa3 -2.84% vs HEFT, +0.34% vs full (p=0.31), SLA 33.7%; kappa10 -3.32% vs HEFT, -0.14% vs full (p=0.65), SLA 29.3%; GA -2.71% vs HEFT, +0.49% vs full (p=0.15), SLA 33.6%

- robustness GM-HRL vs HEFT: no RRD ($p_{rrd}\times 0$): -1.29%; default: -3.17%; RRD $\times 2.67$: -5.22%; load $\rho=0.6$: -0.81%; load $\rho=0.95$: -4.74%; capacity $\times 0.7$: -2.01%

- transfer: genomic $\to$ Video: GA +0.76%, GM-HRL-genomic -2.82%, GM-HRL-in-domain -2.74%; genomic $\to$ AI fine-tuning: GA +0.22%, GM-HRL-genomic +1.73%, GM-HRL-in-domain +0.01%; genomic $\to$ Financial: GA +0.67%, GM-HRL-genomic +0.85%, GM-HRL-in-domain +0.04%; $N{=}100 \to 200$: GA -2.71%, GM-HRL-N100 -3.50%, GM-HRL-in-size -3.17%; $N{=}100 \to 500$: GA -1.76%, GM-HRL-N100 -2.65%, GM-HRL-in-size -2.52%; $N{=}100 \to 1000$: GA -1.98%, GM-HRL-N100 -2.10%

- NOTE: latency from evaluation runs was measured on a loaded machine; use the dedicated measurement for the paper.
