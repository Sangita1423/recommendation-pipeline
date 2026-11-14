from src.evaluation import metrics as M


def test_precision_recall():
    rec, rel = ["a", "b", "c", "d"], {"b", "d", "z"}
    assert M.precision_at_k(rec, rel, 4) == 0.5
    assert abs(M.recall_at_k(rec, rel, 4) - 2 / 3) < 1e-9


def test_ndcg_perfect_and_zero():
    assert abs(M.ndcg_at_k(["a", "b"], {"a", "b"}, 2) - 1.0) < 1e-9
    assert M.ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_coverage_and_diversity():
    recs = {"u1": ["a", "b"], "u2": ["b", "c"]}
    assert M.catalog_coverage(recs, 4) == 0.75
    genres = {"a": {"Drama"}, "b": {"Drama"}, "c": {"Comedy"}}
    assert M.intra_list_diversity(recs, genres, 2) == 0.5   # u1: 0, u2: 1


def test_rmse_mae():
    assert M.rmse([1, 3], [1, 5]) == 2 ** 0.5
    assert M.mae([1, 3], [1, 5]) == 1.0
