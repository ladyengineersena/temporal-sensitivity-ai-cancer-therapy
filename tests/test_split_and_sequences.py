import numpy as np
import pandas as pd

from training.train_model import DEFAULT_FEATURES, make_sequences, split_patients


def test_split_patients_disjoint_and_complete():
    patient_ids = np.arange(30)
    train_pids, val_pids, test_pids = split_patients(
        patient_ids=patient_ids,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=123,
    )

    train_set = set(train_pids.tolist())
    val_set = set(val_pids.tolist())
    test_set = set(test_pids.tolist())

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    all_ids = train_set.union(val_set).union(test_set)
    assert all_ids == set(patient_ids.tolist())


def test_make_sequences_sorts_by_time():
    # İki patient için temporal_sensitivity = time olacak şekilde kuruyoruz.
    rows = []
    for pid in [0, 1]:
        for t in range(5):
            rows.append(
                {
                    "patient_id": pid,
                    "time": t,
                    "treatment_intensity": float(t),
                    "tumor_burden": float(2 * t),
                    "toxicity": float(3 * t),
                    "temporal_sensitivity": float(t),
                }
            )

    df = pd.DataFrame(rows)
    # Satırları karıştır: make_sequences içindeki sort_values("time") çalışmalı.
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    X, y, pids = make_sequences(df=df, features=DEFAULT_FEATURES, seq_len=3)

    # timesteps=5, seq_len=3 -> (5-3)=2 sequence per patient -> toplam 4
    assert X.shape == (4, 3, 3)
    assert y.shape == (4,)
    assert pids.shape == (4,)

    # pid 0 için y: time 3 ve 4; pid 1 için aynı.
    assert y.tolist() == [3.0, 4.0, 3.0, 4.0]
    assert pids.tolist() == [0, 0, 1, 1]

