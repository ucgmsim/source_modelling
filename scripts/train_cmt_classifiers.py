"""Train the nodal plane and tectonic type classifiers.

Run with the ``ml`` dependency group::

    uv run --group ml scripts/train_cmt_classifiers.py

The nodal plane model is a random forest trained on human-picked GeoNet
solutions (``tests/data/GeoNet_Test_Solutions.csv``, where plane 1 is the
preferred plane), optionally augmented with synthetic solutions generated
from the CFM. It is evaluated with grouped cross-validation, grouping events
by 1-degree cell so that an earthquake sequence never straddles the
train/test split.

The tectonic type model is a random forest trained on the NZ NSHM 2022 rule
applied to the full GeoNet CMT catalogue under Monte Carlo perturbation of
the depth below the interface, so that it predicts class probabilities that
account for centroid and Slab2 depth uncertainty. The hard classification
in the package follows the rule directly; the rule and the model are both
checked against literature-classified events
(``tests/data/tectonic_type_literature_labels.csv``).

Both forests are exported to JSON in ``source_modelling/NZ_CFM`` and evaluated
with numpy at runtime (see `source_modelling.focal_mechanism.Forest`).
"""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

from source_modelling import focal_mechanism as fm
from source_modelling.community_fault_model import NodalPlane, get_community_fault_model

CATALOGUE_URL = "https://raw.githubusercontent.com/GeoNet/data/refs/heads/main/moment-tensor/GeoNet_CMT_solutions.csv"
REPO = Path(__file__).resolve().parent.parent
RANDOM_STATE = 20260908


def forest_to_json(
    forest: RandomForestClassifier | ExtraTreesClassifier, feature_names: list[str]
) -> dict:
    """Export a fitted scikit-learn forest to the JSON layout used by `fm.Forest`."""
    trees = []
    for estimator in forest.estimators_:
        tree = estimator.tree_
        value = tree.value[:, 0, :]
        value = value / np.maximum(value.sum(axis=1, keepdims=True), 1e-12)
        trees.append(
            {
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": np.round(tree.threshold, 4).tolist(),
                "value": np.round(value, 3).tolist(),
            }
        )
    classes = [c.item() if hasattr(c, "item") else c for c in forest.classes_]
    return {"feature_names": feature_names, "classes": classes, "trees": trees}


def solution_features(
    classifier: fm.CMTClassifier, frame: pd.DataFrame
) -> list[fm.EventFeatures]:
    """Compute features for every row of a GeoNet-format data frame."""
    return [
        classifier.features(
            np.array([row.Latitude, row.Longitude, row.CD]),
            NodalPlane(row.strike1, row.dip1, row.rake1),
            NodalPlane(row.strike2, row.dip2, row.rake2),
            row.Mw,
        )
        for row in frame.itertuples()
    ]


def nodal_plane_design(
    features: list[fm.EventFeatures],
) -> tuple[np.ndarray, np.ndarray]:
    """Antisymmetric design matrix: both plane orderings for each event.

    Plane 1 is the preferred plane, so ordering 0 has label 1 and the swapped
    ordering has label 0.
    """
    x = np.array(
        [f.nodal_plane_vector(0) for f in features]
        + [f.nodal_plane_vector(1) for f in features]
    )
    y = np.array([1] * len(features) + [0] * len(features))
    return x, y


def antisymmetric_probability(
    model: fm.Forest | RandomForestClassifier | ExtraTreesClassifier,
    features: list[fm.EventFeatures],
) -> np.ndarray:
    """Probability that plane 1 is preferred, averaged over both orderings."""
    x0 = np.array([f.nodal_plane_vector(0) for f in features])
    x1 = np.array([f.nodal_plane_vector(1) for f in features])
    classes = list(model.classes if isinstance(model, fm.Forest) else model.classes_)
    column = classes.index(1)
    p0 = model.predict_proba(x0)[:, column]
    p1 = model.predict_proba(x1)[:, column]
    return 0.5 * (p0 + 1.0 - p1)


def make_forest(random_state: int = RANDOM_STATE) -> ExtraTreesClassifier:
    """The shallow randomised-tree ensemble used for the nodal plane model.

    Extremely randomised trees were the most accurate and the most stable
    configuration under grouped cross-validation (see the module docstring
    for the comparison procedure).
    """
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )


def make_tectonic_forest(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """The forest used for tectonic type probabilities."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )


def perturbed_rule_labels(
    features: list[fm.EventFeatures],
    rng: np.random.Generator,
    samples: int,
    depth_sigma_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo tectonic labels under centroid and interface depth uncertainty.

    Each event is replicated ``samples`` times; in each replicate the depth
    below the slab is perturbed by Gaussian noise (combined centroid and
    Slab2 depth uncertainty) before applying the NSHM rule. The features are
    left unperturbed, so a forest trained on these labels learns
    P(type | features) marginalised over the depth uncertainty.
    """
    x, y = [], []
    index = fm.EVENT_FEATURE_NAMES.index("depth_below_slab")
    for feature in features:
        vector = feature.tectonic_vector()
        for _ in range(samples):
            perturbed = fm.EventFeatures(
                feature.plane_features,
                feature.event_features.copy(),
                feature.slab,
                feature.interface_like,
            )
            if feature.slab.present:
                perturbed.event_features[index] += rng.normal(0.0, depth_sigma_km)
            x.append(vector)
            y.append(fm.tectonic_type_rule(perturbed).value)
    return np.array(x), np.array(y)


def cross_validate_nodal_plane(
    labelled: list[fm.EventFeatures],
    groups: np.ndarray,
    synthetic: list[fm.EventFeatures],
    n_splits: int = 5,
    repeats: int = 5,
) -> float:
    """Grouped cross-validation accuracy of the nodal plane model on labelled events."""
    correct = []
    x_syn, y_syn = (
        nodal_plane_design(synthetic) if synthetic else (np.empty((0, 0)), np.empty(0))
    )
    for repeat in range(repeats):
        # Shuffle group assignment by permuting group labels.
        rng = np.random.default_rng(repeat)
        unique = np.unique(groups)
        permuted = dict(zip(unique, rng.permutation(len(unique))))
        shuffled_groups = np.array([permuted[g] for g in groups])
        for train, test in GroupKFold(n_splits=n_splits).split(
            groups, groups=shuffled_groups
        ):
            x_train, y_train = nodal_plane_design([labelled[i] for i in train])
            if synthetic:
                x_train = np.vstack([x_train, x_syn])
                y_train = np.concatenate([y_train, y_syn])
            model = make_forest(repeat).fit(x_train, y_train)
            probability = antisymmetric_probability(model, [labelled[i] for i in test])
            correct.append(probability >= 0.5)
    return float(np.concatenate(correct).mean())


def load_catalogue(path: Path) -> pd.DataFrame:
    """Load (downloading if needed) the GeoNet CMT catalogue."""
    if not path.exists():
        print(f"Downloading GeoNet CMT catalogue to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CATALOGUE_URL, path)
    return pd.read_csv(path)


def main() -> None:
    """Train and export both classifiers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue", type=Path, default=Path("/tmp/GeoNet_CMT_solutions.csv")
    )
    parser.add_argument(
        "--labelled", type=Path, default=REPO / "tests/data/GeoNet_Test_Solutions.csv"
    )
    parser.add_argument(
        "--literature",
        type=Path,
        default=REPO / "tests/data/tectonic_type_literature_labels.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "source_modelling/NZ_CFM"
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=300,
        help="Synthetic CFM solutions added to nodal plane training (0 to disable)",
    )
    parser.add_argument(
        "--depth-samples",
        type=int,
        default=8,
        help="Monte Carlo depth samples per catalogue event for the tectonic model",
    )
    parser.add_argument(
        "--depth-sigma",
        type=float,
        default=7.0,
        help="Standard deviation (km) of the depth-below-interface uncertainty",
    )
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    faults = get_community_fault_model()
    classifier = fm.CMTClassifier(
        fm.FaultSegmentIndex(faults), fm.SlabModel.load(), None, None
    )

    # ---------------------------------------------------------------- nodal plane
    labelled_frame = pd.read_csv(args.labelled)
    labelled = solution_features(classifier, labelled_frame)
    groups = np.array(
        [
            f"{round(r.Latitude)}_{round(r.Longitude)}"
            for r in labelled_frame.itertuples()
        ]
    )
    print(
        f"Labelled events: {len(labelled)} in {len(np.unique(groups))} spatial groups"
    )

    rng = np.random.default_rng(RANDOM_STATE)
    synthetic_solutions = (
        fm.synthetic_solutions(faults, args.n_synthetic, rng)
        if args.n_synthetic
        else []
    )
    synthetic = [
        classifier.features(s.centroid, s.fault_plane, s.auxiliary_plane)
        for s in synthetic_solutions
    ]

    report: dict = {"n_labelled": len(labelled), "n_synthetic": len(synthetic)}
    for key, name, augmentation in [
        ("cv_accuracy_labelled_only", "labelled only", []),
        (
            "cv_accuracy_with_synthetic",
            f"labelled + {len(synthetic)} synthetic",
            synthetic,
        ),
    ]:
        if not augmentation and key == "cv_accuracy_with_synthetic":
            continue
        report[key] = cross_validate_nodal_plane(labelled, groups, augmentation)
        print(f"Grouped 5-fold CV accuracy ({name}): {report[key]:.3f}")

    x, y = nodal_plane_design(labelled)
    if synthetic:
        x_syn, y_syn = nodal_plane_design(synthetic)
        x, y = np.vstack([x, x_syn]), np.concatenate([y, y_syn])
    nodal_model = make_forest().fit(x, y)
    in_sample = (antisymmetric_probability(nodal_model, labelled) >= 0.5).mean()
    report["in_sample_accuracy"] = float(in_sample)
    print(f"In-sample accuracy on labelled events: {in_sample:.3f}")
    if synthetic:
        synthetic_holdout = [
            classifier.features(s.centroid, s.fault_plane, s.auxiliary_plane)
            for s in fm.synthetic_solutions(faults, 300, np.random.default_rng(1))
        ]
        holdout = (
            antisymmetric_probability(nodal_model, synthetic_holdout) >= 0.5
        ).mean()
        report["synthetic_holdout_accuracy"] = float(holdout)
        print(f"Accuracy on 300 held-out synthetic solutions: {holdout:.3f}")
    importance = sorted(
        zip(nodal_model.feature_importances_, fm.NODAL_PLANE_FEATURE_NAMES),
        reverse=True,
    )
    print("Feature importances:")
    for value, name in importance:
        print(f"  {name:40s} {value:.3f}")

    # -------------------------------------------------------------- tectonic type
    catalogue = load_catalogue(args.catalogue)
    catalogue_features = solution_features(classifier, catalogue)
    x_tec = np.array([f.tectonic_vector() for f in catalogue_features])
    y_rule = np.array([fm.tectonic_type_rule(f).value for f in catalogue_features])
    print(
        "Rule-based tectonic labels on catalogue:",
        {k: int(v) for k, v in zip(*np.unique(y_rule, return_counts=True))},
    )
    x_mc, y_mc = perturbed_rule_labels(
        catalogue_features,
        np.random.default_rng(RANDOM_STATE),
        args.depth_samples,
        args.depth_sigma,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    agreement = []
    for train, test in folds.split(x_tec, y_rule):
        train_rows = np.concatenate(
            [
                np.arange(i * args.depth_samples, (i + 1) * args.depth_samples)
                for i in train
            ]
        )
        model = make_tectonic_forest().fit(x_mc[train_rows], y_mc[train_rows])
        agreement.append(model.predict(x_tec[test]) == y_rule[test])
    print(
        f"Tectonic model 5-fold CV: most probable class agrees with rule for {np.concatenate(agreement).mean():.3f} of events"
    )
    tectonic_model = make_tectonic_forest().fit(x_mc, y_mc)

    literature = pd.read_csv(args.literature).merge(labelled_frame, on="PublicID")
    literature_features = solution_features(classifier, literature)
    rule_labels = np.array(
        [fm.tectonic_type_rule(f).value for f in literature_features]
    )
    probabilities = tectonic_model.predict_proba(
        np.array([f.tectonic_vector() for f in literature_features])
    )
    model_labels = tectonic_model.classes_[probabilities.argmax(axis=1)]
    truth = literature.tectonic_type.to_numpy()
    report["tectonic_literature_rule_accuracy"] = float(np.mean(rule_labels == truth))
    report["tectonic_literature_model_accuracy"] = float(np.mean(model_labels == truth))
    print(
        f"Literature events: {len(truth)}; rule accuracy {np.mean(rule_labels == truth):.3f}; model most-probable accuracy {np.mean(model_labels == truth):.3f}"
    )
    for row, rule, predicted, probability in zip(
        literature.itertuples(), rule_labels, model_labels, probabilities
    ):
        if (
            row.tectonic_type != "crustal"
            or rule != row.tectonic_type
            or predicted != row.tectonic_type
        ):
            p = {
                str(c): float(v)
                for c, v in zip(tectonic_model.classes_, probability.round(2))
            }
            print(
                f"  {row.PublicID:12s} {row.event:35s} truth={row.tectonic_type:9s} rule={rule:9s} P={p}"
            )

    if not args.no_export:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / "nodal_plane_model.json", "w") as handle:
            json.dump(
                forest_to_json(nodal_model, fm.NODAL_PLANE_FEATURE_NAMES),
                handle,
                separators=(",", ":"),
            )
        with open(args.output_dir / "training_report.json", "w") as handle:
            json.dump(report, handle, indent=1)
        with open(args.output_dir / "tectonic_type_model.json", "w") as handle:
            json.dump(
                forest_to_json(tectonic_model, fm.TECTONIC_FEATURE_NAMES),
                handle,
                separators=(",", ":"),
            )
        print(f"Exported models to {args.output_dir}")


if __name__ == "__main__":
    main()
