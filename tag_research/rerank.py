"""Deterministic score fusion and greedy maximum marginal relevance."""
import numpy as np

def unit_rows(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('expected finite matrix')
    norms=np.linalg.norm(values,axis=1,keepdims=True)
    if np.any(norms<1e-12):raise ValueError('zero-norm embedding')
    return values/norms

def standardize(scores):
    scores=np.asarray(scores,dtype=np.float64)
    if scores.ndim!=1 or not np.isfinite(scores).all():raise ValueError('expected finite score vector')
    scale=scores.std() if len(scores) else 0
    return (scores-scores.mean())/scale if scale>1e-12 else np.zeros_like(scores)

def fuse(ram_probs,clip_cosines,alpha=0.5):
    ram=np.asarray(ram_probs,dtype=np.float64);clip=np.asarray(clip_cosines,dtype=np.float64)
    if ram.shape!=clip.shape or ram.ndim!=1:raise ValueError('score alignment mismatch')
    if not 0<=alpha<=1 or not np.isfinite(ram).all() or np.any((ram<0)|(ram>1)):raise ValueError('invalid probabilities/alpha')
    # Per-image ranking normalization, not probability calibration.
    return (1-alpha)*standardize(ram)+alpha*standardize(clip)

def select_mmr(relevance,text_embeddings,k,weight=0.8):
    scores=np.asarray(relevance,dtype=np.float64)
    text=unit_rows(text_embeddings)
    if scores.ndim!=1 or len(scores)!=len(text) or not np.isfinite(scores).all():raise ValueError('invalid relevance')
    if type(k) is not int or k<0 or not 0<=weight<=1:raise ValueError('invalid selection parameters')
    pair=text@text.T
    chosen=[];remaining=list(range(len(scores)))
    while remaining and len(chosen)<k:
        if not chosen:best=max(remaining,key=lambda i:(scores[i],-i))
        else:
            best=max(remaining,key=lambda i:(weight*scores[i]-(1-weight)*max(pair[i,j] for j in chosen),-i))
        chosen.append(best);remaining.remove(best)
    return chosen

def metrics(ranking,truth,k=5):
    if type(k) is not int or k<1:raise ValueError('k must be positive')
    if len(set(ranking))!=len(ranking):raise ValueError('duplicate predictions')
    truth=set(truth);top=ranking[:k];hits=sum(x in truth for x in top)
    precision=hits/k
    recall=hits/len(truth) if truth else 0.0
    return {'p_at_k':precision,'recall_at_k':recall,'f1_at_k':2*precision*recall/(precision+recall) if precision+recall else 0.0}
