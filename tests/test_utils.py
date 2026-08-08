from src.utils import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_preprocessor,
    load_params,
)


def test_feature_columns_have_no_overlap():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert FEATURE_COLUMNS == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_load_params_has_expected_keys():
    params = load_params()
    assert "random_state" in params
    assert params["data"]["subsample_size"] > 0
    assert params["training"]["n_trials"] > 0


def test_build_preprocessor_has_both_transformers():
    preprocessor = build_preprocessor()
    names = [name for name, _, _ in preprocessor.transformers]
    assert names == ["numeric", "categorical"]
