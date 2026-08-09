#!/usr/bin/env python3
"""Descriptive resource-scaling benchmark for dense vs sparse memory representations.
Not part of the inferential hypothesis family; rerun on the target laptop for local numbers.
"""
import argparse, time, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='resource_benchmark'); ap.add_argument('--dim',type=int,default=256)
    ap.add_argument('--sizes',default='1000,5000,10000,25000'); ap.add_argument('--nnz',type=int,default=16); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    sizes=[int(x) for x in args.sizes.split(',')]; rng=np.random.default_rng(20260809); rows=[]
    for n in sizes:
        dense=np.zeros((n,args.dim),dtype=np.float32)
        ri=np.repeat(np.arange(n),args.nnz); ci=np.concatenate([rng.choice(args.dim,args.nnz,replace=False) for _ in range(n)])
        vals=rng.normal(size=n*args.nnz).astype(np.float32); dense[ri,ci]=vals
        csr=sparse.csr_matrix((vals,(ri,ci)),shape=(n,args.dim),dtype=np.float32)
        q=rng.normal(size=args.dim).astype(np.float32)
        # warmup
        _=dense@q; _=csr@q
        def best_time(fn,reps=7):
            times=[]
            for _ in range(reps):
                t=time.perf_counter(); fn(); times.append(time.perf_counter()-t)
            return min(times)
        td=best_time(lambda: dense@q); ts=best_time(lambda: csr@q)
        dense_bytes=dense.nbytes
        sparse_bytes=csr.data.nbytes+csr.indices.nbytes+csr.indptr.nbytes
        rows.append({'records':n,'dim':args.dim,'nnz_per_record':args.nnz,'dense_bytes':dense_bytes,'sparse_bytes':sparse_bytes,
                     'memory_ratio_sparse_over_dense':sparse_bytes/dense_bytes,'dense_query_ms':td*1000,'sparse_query_ms':ts*1000})
        print(rows[-1])
    df=pd.DataFrame(rows); df.to_csv(out/'resource_scaling.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4)); ax.plot(df.records,df.dense_bytes/1e6,marker='o',label='dense'); ax.plot(df.records,df.sparse_bytes/1e6,marker='o',label='sparse'); ax.set_xlabel('records'); ax.set_ylabel('representation MB'); ax.legend(); fig.tight_layout(); fig.savefig(out/'memory_scaling.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); ax.plot(df.records,df.dense_query_ms,marker='o',label='dense'); ax.plot(df.records,df.sparse_query_ms,marker='o',label='sparse'); ax.set_xlabel('records'); ax.set_ylabel('single query ms'); ax.legend(); fig.tight_layout(); fig.savefig(out/'query_latency_scaling.png',dpi=170); plt.close(fig)
    (out/'README.txt').write_text('Descriptive local benchmark only. Do not compare these numbers directly to the paper\'s earlier illustrative MB values unless the data structures and workload are equivalent.\n')

if __name__=='__main__': main()
