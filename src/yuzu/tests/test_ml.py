import pytest
import yuzu as yz
import math
import yuzu.torch as yz_torch

@pytest.fixture(params=["torch"])
def backend(request):
    backend = request.param
    if backend == 'torch':
        yz_torch.init()
    yield yz.Backend

def test_categorical_sample(backend):
    yz.set_seed(55)
    probs = [0.1, 0.3, 0.6]
    logits = yz.from_value([math.log(p) for p in probs])
    dist = yz.Categorical(logits)
    N = 20000
    cnt = [0] * 3
    for i in range(N):
        j = (dist.sample((1,)).item())
        cnt[j] += 1
    dprobs = [c/N for c in cnt]
    assert dprobs == pytest.approx(probs, rel=0.01)

def test_mse_loss():
    # Test cases
    x = yz.from_value([[1, 2, 3], [4, 5, 6]]).float()
    y = yz.from_value([[1, 2, 3], [4, 5, 6]]).float()
    z = yz.from_value([[2, 2, 2], [2, 2, 2]]).float()
    
    # Test 'mean' reduction
    result_mean = yz.mse_loss(x, y, reduction='mean')
    expected_mean = yz.from_value([0, 0])
    assert (result_mean == expected_mean).all()

    result_mean_diff = yz.mse_loss(x, z, reduction='mean')
    expected_mean_diff = yz.from_value([2/3, 29/3])
    assert result_mean_diff.is_close(expected_mean_diff).all()

    # Test 'sum' reduction
    result_sum = yz.mse_loss(x, y, reduction='sum')
    expected_sum = yz.from_value([0, 0]) 
    assert (result_sum == expected_sum).all()

    result_sum_diff = yz.mse_loss(x, z, reduction='sum')
    expected_sum_diff = yz.from_value([2, 29]) 
    assert (result_sum_diff == expected_sum_diff).all()

    # Test 'none' reduction
    result_none = yz.mse_loss(x, y, reduction='none')
    expected_none = yz.from_value([[0, 0, 0], [0, 0, 0]]) 
    assert (result_none == expected_none).all()

    result_none_diff = yz.mse_loss(x, z, reduction='none')
    expected_none_diff = yz.from_value([[1, 0, 1], [4, 9, 16]])
    assert (result_none_diff == expected_none_diff).all()

    # Test shape mismatch
    with pytest.raises(Exception, match="mse_loss: shapes mismatch"):
        yz.mse_loss(x, yz.from_value([1, 2, 3]))

    # Test invalid reduction mode
    with pytest.raises(Exception, match="mse_loss: invalid reduction mode"):
        yz.mse_loss(x, y, reduction='invalid')
