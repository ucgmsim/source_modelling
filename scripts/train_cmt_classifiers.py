"""Train the nodal plane classifier.

Run with the ``ml`` dependency group::

    uv run --group ml scripts/train_cmt_classifiers.py

The model is an L2-regularised logistic regression without intercept on the
plane 1 - plane 2 CFM misfits
(`source_modelling.focal_mechanism.NODAL_PLANE_FEATURE_NAMES`), fitted on
both plane orderings so that swapping the planes flips the prediction.

The labels (``tests/data/nodal_plane_labels.csv``, plane 1 preferred) are
human-picked fault planes for GeoNet CMT solutions. They combine Felipe's
review of GeoNet CMTs, Robin Lee's picks and those of a suite of moderate
crustal events prepared for simulation validation; the ``source`` column
records which sets agree on each event. Where they disagree, the reviewed
GeoNet pick is used.

Accuracy is estimated by repeated grouped cross-validation, grouping events
by 1-degree cell so that an earthquake sequence never straddles the
train/test split. The mapped crustal faults say less about deeper events,
so the log-odds are then scaled by a depth temperature
(`source_modelling.focal_mechanism.depth_temperature`) fitted to the
out-of-fold log-odds. The weights and temperature are exported to
``source_modelling/NZ_CFM/nodal_plane_model.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from source_modelling import focal_mechanism as fm
from source_modelling.community_fault_model import NodalPlane, get_community_fault_model

REPO = Path(__file__).resolve().parent.parent
REGULARISATION = 0.03
"""Inverse L2 regularisation strength (scikit-learn's ``C``) on standardised features."""


def _fit_weights(differences: np.ndarray) -> np.ndarray:
    """Fit the antisymmetric logistic regression and return weights on the raw differences."""
    # Scale by the spread of each feature so one regularisation strength suits all.
    scale = np.std(np.abs(differences), axis=0) + 1e-9
    x = np.vstack([differences, -differences]) / scale
    y = np.r_[np.ones(len(differences)), np.zeros(len(differences))]
    model = LogisticRegression(C=REGULARISATION, fit_intercept=False).fit(x, y)
    return model.coef_[0] / scale


def _cross_validate(
    differences: np.ndarray, groups: np.ndarray, n_splits: int = 5, repeats: int = 10
) -> np.ndarray:
    """Out-of-fold plane 1 log-odds from grouped cross-validation, one row per repeat."""
    log_odds = np.zeros((repeats, len(differences)))
    for repeat in range(repeats):
        folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        for train, test in folds.split(differences, groups=groups):
            log_odds[repeat, test] = differences[test] @ _fit_weights(
                differences[train]
            )
    return log_odds


def _log_loss(log_odds: np.ndarray, temperature: np.ndarray | float = 1.0) -> float:
    """Mean log-loss of plane 1 log-odds (plane 1 is always the label)."""
    return float(np.mean(np.logaddexp(0.0, -temperature * log_odds)))


def _fit_temperature(log_odds: np.ndarray, depth: np.ndarray) -> tuple[float, float]:
    """Fit the depth temperature's scale and decay rate to out-of-fold log-odds."""
    result = sp.optimize.minimize(
        lambda p: _log_loss(log_odds, fm.depth_temperature(depth, *p)),
        [1.0, 0.0],
        bounds=[(0.0, None), (0.0, None)],
    )
    return float(result.x[0]), float(result.x[1])


def main() -> None:
    """Train and export the nodal plane classifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labelled", type=Path, default=REPO / "tests/data/nodal_plane_labels.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "source_modelling/NZ_CFM"
    )
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    segments = fm.FaultSegmentIndex(get_community_fault_model())
    labelled = pd.read_csv(args.labelled)
    # Plane 1 - plane 2 misfits for every event.
    columns = ["Latitude", "Longitude", "CD"]
    columns += ["strike1", "dip1", "rake1", "strike2", "dip2", "rake2"]
    differences = np.array(
        [
            segments.plane_misfits(lat, lon, depth, NodalPlane(*planes[:3]))
            - segments.plane_misfits(lat, lon, depth, NodalPlane(*planes[3:]))
            for lat, lon, depth, *planes in labelled[columns].to_numpy(float).tolist()
        ]
    )
    groups = (
        labelled.Latitude.round().astype(int).astype(str)
        + "_"
        + labelled.Longitude.round().astype(int).astype(str)
    ).to_numpy()
    print(
        f"Labelled events: {len(differences)} in {len(np.unique(groups))} spatial groups"
    )

    log_odds = _cross_validate(differences, groups)
    scores = np.mean(log_odds >= 0, axis=1)
    weights = _fit_weights(differences)
    in_sample = np.mean(differences @ weights >= 0)
    print(
        f"Grouped 5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}; "
        f"in-sample {in_sample:.3f}"
    )
    print("Weights on plane 1 - plane 2 misfits:")
    for name, weight in zip(fm.NODAL_PLANE_FEATURE_NAMES, weights):
        print(f"  {name:40s} {weight:+.4f}")

    depth = labelled.CD.to_numpy(float)
    scale, depth_rate = _fit_temperature(
        log_odds, np.broadcast_to(depth, log_odds.shape)
    )
    temperature = fm.depth_temperature(depth, scale, depth_rate)
    print(
        f"Depth temperature: {scale:.3f} * exp(-{depth_rate:.5f} * depth); "
        f"CV log-loss {_log_loss(log_odds):.3f} -> {_log_loss(log_odds, temperature):.3f}"
    )
    print("Mean CV confidence by depth (before -> after temperature; accuracy):")
    for low, high in [(0, 20), (20, 40), (40, 100), (100, np.inf)]:
        in_bin = (depth > low) & (depth <= high)
        before = sp.special.expit(np.abs(log_odds[:, in_bin])).mean()
        after = sp.special.expit(
            np.abs(temperature[in_bin] * log_odds[:, in_bin])
        ).mean()
        print(
            f"  {low:3.0f}-{high:<4.0f} km (n={in_bin.sum():3d}): "
            f"{before:.3f} -> {after:.3f}; {np.mean(log_odds[:, in_bin] >= 0):.3f}"
        )

    if not args.no_export:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / "nodal_plane_model.json", "w") as handle:
            json.dump(
                {
                    "feature_names": fm.NODAL_PLANE_FEATURE_NAMES,
                    "weights": weights.tolist(),
                    "temperature": {"scale": scale, "depth_rate": depth_rate},
                },
                handle,
                indent=1,
            )
        print(f"Exported model to {args.output_dir}")


if __name__ == "__main__":
    main()
