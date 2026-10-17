import numpy as np
import pytest
from tag_research.rerank import fuse,select_mmr,metrics,unit_rows,standardize

def test_mmr_recomputes_best_remaining_after_each_selection():
    # Mathematical regression fixture, not an evaluation image or model output.
    scores=[1,.99,.8];text=[[1,0],[1,0],[0,1]]
    assert select_mmr(scores,text,2,.5)==[0,2]

def test_identity_fusion_endpoints():
    ram=[.2,.9,.4];clip=[.8,.2,.6]
    assert np.argmax(fuse(ram,clip,0))==1
    assert np.argmax(fuse(ram,clip,1))==0

def test_constant_scores_do_not_make_nan():
    assert np.array_equal(standardize([.5,.5]),[0,0])

def test_precision_keeps_k_denominator():
    assert metrics(['cat'],{'cat'},5)['p_at_k']==.2

def test_no_duplicate_selection():
    assert select_mmr([1,1],[[1,0],[0,1]],10)==[0,1]

@pytest.mark.parametrize('scores,cos,alpha',[([.2],[.1,.2],.5),([float('nan')],[1],.5),([2],[1],.5),([.2],[float('inf')],.5),([.2],[1],2)])
def test_invalid_fusion_is_rejected(scores,cos,alpha):
    with pytest.raises(ValueError):fuse(scores,cos,alpha)

def test_zero_vectors_rejected():
    with pytest.raises(ValueError):unit_rows([[0,0]])

def test_duplicate_predictions_rejected():
    with pytest.raises(ValueError):metrics(['cat','cat'],{'cat'})
