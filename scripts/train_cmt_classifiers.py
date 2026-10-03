"""Train the nodal plane classifier.

Run with the ``ml`` dependency group::

    uv run --group ml scripts/train_cmt_classifiers.py

The nodal plane model is an L2-regularised logistic regression on the
difference between the two planes' CFM misfits
(`source_modelling.focal_mechanism.NODAL_PLANE_FEATURE_NAMES`), fitted to
human-picked GeoNet solutions (``tests/data/GeoNet_Test_Solutions.csv``,
where plane 1 is the preferred plane). It has no intercept and is fitted on
both plane orderings, so swapping the planes exactly flips the prediction.

It is evaluated with repeated grouped cross-validation, grouping events by
1-degree cell so that an earthquake sequence never straddles the
train/test split. The 98 events fall in only about 26 groups, so the
cross-validated accuracy has a standard error of a few percent: more
flexible models (random forests on these and slab, Andersonian and
magnitude features) were no more accurate under the same procedure.

The weights are exported to ``source_modelling/NZ_CFM/nodal_plane_model.json``.
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


def misfit_differences(
    segments: fm.FaultSegmentIndex, frame: pd.DataFrame
) -> np.ndarray:
    """Plane 1 - plane 2 CFM misfits for every row of a GeoNet-format data frame."""
    columns = ["Latitude", "Longitude", "CD"]
    columns += ["strike1", "dip1", "rake1", "strike2", "dip2", "rake2"]
    return np.array(
        [
            segments.plane_misfits(lat, lon, depth, NodalPlane(*planes[:3]))
            - segments.plane_misfits(lat, lon, depth, NodalPlane(*planes[3:]))
            for lat, lon, depth, *planes in frame[columns].to_numpy(float).tolist()
        ]
    )


def fit_weights(differences: np.ndarray) -> np.ndarray:
    """Fit the antisymmetric logistic regression and return weights on raw differences.

    Plane 1 is the preferred plane, so each difference has label 1 and its
    negation (the swapped ordering) label 0. Features are scaled by the
    spread of the absolute differences so one regularisation strength suits
    all of them.
    """
    scale = np.std(np.abs(differences), axis=0) + 1e-9
    x = np.vstack([differences, -differences]) / scale
    y = np.r_[np.ones(len(differences)), np.zeros(len(differences))]
    model = LogisticRegression(C=REGULARISATION, fit_intercept=False).fit(x, y)
    return model.coef_[0] / scale


def accuracy(weights: np.ndarray, differences: np.ndarray) -> float:
    """Fraction of events for which plane 1 is predicted."""
    return float(np.mean(differences @ weights >= 0))


def cross_validate(
    differences: np.ndarray, groups: np.ndarray, n_splits: int = 5, repeats: int = 10
) -> np.ndarray:
    """Grouped cross-validation accuracy for each repeat."""
    scores = []
    for repeat in range(repeats):
        folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        correct = np.zeros(len(differences), dtype=bool)
        for train, test in folds.split(differences, groups=groups):
            weights = fit_weights(differences[train])
            correct[test] = differences[test] @ weights >= 0
        scores.append(correct.mean())
    return np.array(scores)


def main() -> None:
    """Train and export the nodal plane classifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labelled", type=Path, default=REPO / "tests/data/GeoNet_Test_Solutions.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "source_modelling/NZ_CFM"
    )
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    segments = fm.FaultSegmentIndex(get_community_fault_model())
    labelled = pd.read_csv(args.labelled)
    differences = misfit_differences(segments, labelled)
    groups = (
        labelled.Latitude.round().astype(int).astype(str)
        + "_"
        + labelled.Longitude.round().astype(int).astype(str)
    ).to_numpy()
    n_groups = len(np.unique(groups))
    print(f"Labelled events: {len(differences)} in {n_groups} spatial groups")

    scores = cross_validate(differences, groups)
    weights = fit_weights(differences)
    report = {
        "n_labelled": len(differences),
        "n_groups": n_groups,
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "in_sample_accuracy": accuracy(weights, differences),
    }
    print(
        f"Grouped 5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}; "
        f"in-sample {report['in_sample_accuracy']:.3f}"
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
        with open(args.output_dir / "training_report.json", "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"Exported model to {args.output_dir}")


if __name__ == "__main__":
    main()
