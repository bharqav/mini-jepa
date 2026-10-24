"""Weight-free tests for the JEPA linear/kNN probe math."""

import torch

from scripts.evaluate_jepa import (
    effective_rank,
    knn_predict,
    rankdata,
    ridge_fit,
    ridge_predict,
    spearman,
    standardize,
)


def test_ridge_classifier_separates_two_clusters():
    torch.manual_seed(0)
    train = torch.cat([torch.randn(40, 4) + 2.0, torch.randn(40, 4) - 2.0])
    labels = torch.cat([torch.zeros(40, dtype=torch.long), torch.ones(40, dtype=torch.long)])
    weights = ridge_fit(train, labels, 2, alpha=0.1)
    pred = ridge_predict(weights, train, 2)
    assert float((pred == labels).float().mean()) > 0.95


def test_knn_classifier_separates_two_clusters():
    torch.manual_seed(0)
    train = torch.cat([torch.randn(40, 4) + 2.0, torch.randn(40, 4) - 2.0])
    labels = torch.cat([torch.zeros(40, dtype=torch.long), torch.ones(40, dtype=torch.long)])
    pred = knn_predict(train, labels, train, k=5, num_classes=2)
    assert float((pred == labels).float().mean()) > 0.95


def test_spearman_perfect_and_reversed():
    x = torch.arange(10).float()
    assert spearman(x, x) == 1.0
    assert spearman(x, -x) == -1.0


def test_rankdata_handles_ties():
    ranks = rankdata(torch.tensor([2.0, 1.0, 1.0, 3.0]).numpy())
    assert ranks.tolist() == [3.0, 1.5, 1.5, 4.0]


def test_standardize_uses_train_statistics():
    train = torch.tensor([[1.0], [3.0]])
    test = torch.tensor([[5.0]])
    z_train, z_test = standardize(train, test)
    assert torch.allclose(z_train.mean(), torch.tensor(0.0))
    assert z_test.item() > 1.0


def test_effective_rank_is_one_for_constant_and_high_for_isotropic():
    const = torch.ones(64, 8)
    assert effective_rank(const) < 1.01
    isotropic = torch.randn(4096, 64)
    assert effective_rank(isotropic) > 20.0
