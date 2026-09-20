"""Image-cluster percentile confidence intervals; seed variability is reported separately."""
import numpy as np
SIG=lambda x: np.exp(-np.logaddexp(0.,-np.asarray(x,dtype=float)))

def cluster_ci(values, images, B=5000):
    """Resample image clusters; retain every query in each sampled cluster."""
    images=np.asarray(images); names=sorted(set(images)); v=np.asarray(values,float)
    sums=np.array([v[images==i].sum() for i in names]); ns=np.array([(images==i).sum() for i in names])
    rng=np.random.default_rng(3027); idx=rng.integers(0,len(names),(B,len(names)))
    means=sums[idx].sum(1)/ns[idx].sum(1)
    return np.quantile(means,[.025,.975]).tolist()

