import pandas as pd

from src.utils import load_config
from src.validation import rules as R
from src.validation.validate import apply_rules

V = load_config()["validation"]


def test_interaction_rules_quarantine_bad_rows():
    df = pd.DataFrame({
        "user_id":  ["U00001", "U00001", "U99999", "U00001", "U00001", None],
        "item_id":  ["I00001", "I00002", "I00001", "I00001", "I00003", "I00001"],
        "rating":   ["4",      "7",      "3",      "4",      "five",   "2"],
        "timestamp": ["2024-01-01 10:00:00", "2024-01-02 10:00:00", "2024-01-03 10:00:00",
                      "2024-01-01 10:00:00", "not_a_date", "2024-01-05 10:00:00"],
    })
    ok, errors, _, _ = apply_rules(df, R.interaction_rules(V, {"U00001"}, {"I00001", "I00002", "I00003"}))
    assert list(ok.index) == [0]
    reasons = dict(zip(errors.index, errors.error_reason))
    assert "rating_out_of_range" in reasons[1]
    assert "orphan_user_id" in reasons[2]
    assert "exact_duplicate" in reasons[3]
    assert "rating_wrong_type" in reasons[4] and "timestamp_unparseable" in reasons[4]
    assert "missing_user_id" in reasons[5]


def test_user_warnings_are_kept():
    df = pd.DataFrame({"user_id": ["U00001", "U00001"], "age": ["230", "30"],
                       "gender": [" male", "F"], "country": [None, "India"],
                       "signup_date": ["2024-01-01", "2024-01-01"]})
    ok, errors, warns, _ = apply_rules(df, R.user_rules(V))
    assert len(ok) == 1 and len(errors) == 1            # duplicate id quarantined
    assert "age_out_of_range" in warns.warning_reason.iat[0]
