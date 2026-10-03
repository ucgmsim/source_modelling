"""Train the nodal plane classifier.

Run with the ``ml`` dependency group::

    uv run --group ml scripts/train_cmt_classifiers.py

The model is an L2-regularised logistic regression without intercept on the
plane 1 - plane 2 CFM misfits
(`source_modelling.focal_mechanism.NODAL_PLANE_FEATURE_NAMES`), fitted on
both plane orderings so that swapping the planes flips the prediction.

The labels (``tests/data/nodal_plane_labels.csv``, plane 1 preferred)
combine two independent sets of human-picked fault planes for GeoNet CMT
solutions: Robin Lee's picks and those of a suite of moderate crustal events
prepared for simulation validation. Events where the two sets disagree are
excluded, and the ``source`` column records which set each event came from.

Accuracy is estimated by repeated grouped cross-validation, grouping events
by 1-degree cell so that an earthquake sequence never straddles the
train/test split. The weights are exported to
``source_modelling/NZ_CFM/nodal_plane_model.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    """Grouped cross-validation accuracy for each repeat."""
    scores = []
    for repeat in range(repeats):
        folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        correct = np.zeros(len(differences), dtype=bool)
        for train, test in folds.split(differences, groups=groups):
            correct[test] = differences[test] @ _fit_weights(differences[train]) >= 0
        scores.append(correct.mean())
    return np.array(scores)


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

    scores = _cross_validate(differences, groups)
    weights = _fit_weights(differences)
    in_sample = np.mean(differences @ weights >= 0)
    print(
        f"Grouped 5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}; "
        f"in-sample {in_sample:.3f}"
    )
    print("Weights on plane 1 - plane 2 misfits:")
    for name, weight in zip(fm.NODAL_PLANE_FEATURE_NAMES, weights):
        print(f"  {name:40s} {weight:+.4f}")

    if not args.no_export:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / "nodal_plane_model.json", "w") as handle:
            json.dump(
                {
                    "feature_names": fm.NODAL_PLANE_FEATURE_NAMES,
                    "weights": weights.tolist(),
                },
                handle,
                indent=1,
            )
        print(f"Exported model to {args.output_dir}")


if __name__ == "__main__":
    main()
