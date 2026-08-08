from src.predict import predict_one, risk_label


def test_risk_label_boundaries():
    assert risk_label(0.0) == "Low"
    assert risk_label(0.29) == "Low"
    assert risk_label(0.3) == "Medium"
    assert risk_label(0.59) == "Medium"
    assert risk_label(0.6) == "High"
    assert risk_label(1.0) == "High"


def test_predict_one_returns_expected_shape(valid_application):
    result = predict_one(valid_application)
    assert set(result.keys()) == {"prediction", "default_probability", "risk_label"}
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["default_probability"] <= 1.0
    assert result["risk_label"] in ("Low", "Medium", "High")
