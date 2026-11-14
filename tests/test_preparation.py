import pandas as pd

from src.preparation.prepare import clean_interactions, time_split


def test_dedup_keeps_latest_and_split_is_temporal():
    df = pd.DataFrame({"user_id": ["U1", "U1", "U2", "U2"], "item_id": ["I1", "I1", "I1", "I2"],
                       "rating": ["2", "5", "3", "4"],
                       "timestamp": ["2024-01-01 00:00:00", "2024-02-01 00:00:00",
                                     "2024-03-01 00:00:00", "2024-04-01 00:00:00"]})
    x = clean_interactions(df)
    assert len(x) == 3
    assert x[(x.user_id == "U1")].rating.iat[0] == 5.0
    train, test, cutoff = time_split(x, 0.34)
    assert train.timestamp.max() <= cutoff < test.timestamp.min()
