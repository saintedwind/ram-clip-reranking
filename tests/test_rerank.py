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


@pytest.mark.parametrize('scale',[1e160,1e200,1e308])
def test_large_finite_embeddings_preserve_direction_and_mmr(scale):
    # Numerical stress fixture, not an image/model quality evaluation.
    directions=np.array([[1.,0.],[1.,0.],[0.,1.],[-1.,1.]])
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        normalized=unit_rows(directions*scale)
        selected=select_mmr([1,.99,.8],directions[:3]*scale,2,.5)
    np.testing.assert_allclose(normalized,unit_rows(directions),atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(normalized,axis=1),1.)
    assert selected==[0,2]


def test_embedding_norm_threshold_and_empty_shapes():
    with pytest.raises(ValueError,match='zero-norm'):
        unit_rows([[1e-13,0.]])
    with pytest.raises(ValueError,match='zero-norm'):
        unit_rows(np.empty((2,0)))
    np.testing.assert_allclose(unit_rows([[1e-12,0.]]),[[1.,0.]])
    np.testing.assert_allclose(unit_rows([[8e-13,8e-13]]),[[2**-.5,2**-.5]])
    assert unit_rows(np.empty((0,3))).shape==(0,3)
