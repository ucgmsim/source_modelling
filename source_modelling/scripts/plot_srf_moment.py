"""Utility script to plot moment over time for an SRF."""

import collections
import functools
from pathlib import Path
from typing import Annotated, Generator, Optional

import numpy as np
import pandas as pd
import scipy as sp
import typer
from matplotlib import pyplot as plt

from source_modelling import srf
from workflow import realisations
from workflow.realisations import RupturePropagationConfig, SourceConfig


def tree_nodes_in_order(
    tree: dict[str, str],
) -> Generator[str, None, None]:
    """Generate faults in topologically sorted order.

    Parameters
    ----------
    faults : list[RealisationFault]
        List of RealisationFault objects.

    Yields
    ------
    RealisationFault
        The next fault in the topologically sorted order.
    """
    tree_child_map = collections.defaultdict(list)
    for cur, parent in tree.items():
        if parent:
            tree_child_map[parent].append(cur)

    def in_order_traversal(
        node: str,
    ) -> Generator[str, None, None]:
        yield node
        for child in tree_child_map[node]:
            yield from in_order_traversal(child)

    initial_fault = next(cur for cur, parent in tree.items() if not parent)
    yield from in_order_traversal(initial_fault)


def moment_over_time_from_srf(srf_points: pd.DataFrame, mu: float) -> pd.DataFrame:
    non_neglible_patches = srf_points[srf_points["slipt1"].apply(len) > 0]
    time_values = non_neglible_patches.apply(
        (
            lambda row: np.concatenate(
                (
                    np.array([row["tinit"]]),
                    np.full((len(row["slipt1"]) - 1,), row["dt"]),
                )
            ).cumsum()
        ),
        axis=1,
    )
    moment_rate = (
        non_neglible_patches["area"]
        / (100 * 100)
        * non_neglible_patches["dt"]
        * (non_neglible_patches["slipt1"] / 100).apply(
            functools.partial(
                sp.ndimage.convolve1d, weights=[1 / 2, 1 / 2], mode="constant", cval=0
            )
        )
    )
    patch_displacements = pd.DataFrame(
        {
            "t": np.concatenate(time_values.values),
            "moment": np.concatenate(moment_rate.values),
        }
    )
    patch_displacements["t"] = patch_displacements["t"].round(2)
    binned_and_summed_moments = patch_displacements.groupby("t").sum()
    binned_and_summed_moments = mu * binned_and_summed_moments
    return binned_and_summed_moments


def plot_srf_moment(
    srf_ffp: Annotated[
        Path,
        typer.Argument(
            help="SRF filepath to plot", exists=True, readable=True, dir_okay=False
        ),
    ],
    output_png_ffp: Annotated[
        Path, typer.Argument(help="Output plot path", writable=True, dir_okay=False)
    ],
    mu: Annotated[float, typer.Option(help="Shear rigidity constant")] = 3.3e10,
    dpi: Annotated[
        int, typer.Option(help="Plot image pixel density (higher = better)", min=300)
    ] = 300,
    realisation_ffp: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to realisation, used to plot individual fault contribution."
        ),
    ] = None,
):
    """Plot released moment for an SRF over time."""
    srf_data = srf.read_srf(srf_ffp)

    magnitude = (
        2
        / 3
        * np.log10(
            mu * (srf_data.points["area"] * srf_data.points["slip1"] / (100**3)).sum()
        )
        - 6.03333
    )
    overall_moment = moment_over_time_from_srf(srf_data.points, mu)
    plt.plot(
        overall_moment.index.values, overall_moment["moment"], label="Overall Moment"
    )

    if realisation_ffp:
        source_config: SourceConfig = realisations.read_config_from_realisation(
            SourceConfig, realisation_ffp
        )
        rupture_propogation_config: RupturePropagationConfig = (
            realisations.read_config_from_realisation(
                RupturePropagationConfig, realisation_ffp
            )
        )
        segment_counter = 0
        point_counter = 0
        for fault_name in tree_nodes_in_order(
            rupture_propogation_config.rupture_causality_tree
        ):
            plane_count = len(source_config.source_geometries[fault_name].planes)
            segments = srf_data.header.iloc[
                segment_counter : segment_counter + plane_count
            ]
            num_points = (segments["nstk"] * segments["ndip"]).sum()
            individual_moment = moment_over_time_from_srf(
                srf_data.points.iloc[point_counter : point_counter + num_points], mu
            )
            plt.plot(
                individual_moment.index.values,
                individual_moment["moment"],
                label=fault_name,
            )
            segment_counter += plane_count
            point_counter += num_points

    plt.ylabel("Moment (Nm)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.title(f"Moment over Time (Total Mw: {magnitude:.2f})")

    plt.savefig(output_png_ffp, dpi=dpi)


def main():
    typer.run(plot_srf_moment)


if __name__ == "__main__":
    main()