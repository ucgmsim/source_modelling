"""Utility script to plot moment over time for an SRF."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from matplotlib import pyplot as plt

from source_modelling import moment, rupture_propagation, srf
from workflow import realisations
from workflow.realisations import RupturePropagationConfig, SourceConfig


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

    magnitude = moment.moment_to_magnitude(
        mu * (srf_data.points["area"] * srf_data.points["slip"] / (100**3)).sum()
    )

    dt = srf_data.points["dt"].iloc[0]

    overall_moment_rate = moment.moment_rate_over_time_from_slip(
        srf_data.points["area"], srf_data.slip, dt, srf_data.nt, mu
    )
    plt.plot(
        overall_moment_rate.index.values,
        overall_moment_rate["moment_rate"],
        label="Overall Moment Rate",
    )

    if realisation_ffp:
        source_config = SourceConfig.read_from_realisation(realisation_ffp)
        rupture_propogation_config = RupturePropagationConfig.read_from_realisation(realisation_ffp)
        segment_counter = 0
        point_counter = 0
        for fault_name in rupture_propagation.tree_nodes_in_order(
            rupture_propogation_config.rupture_causality_tree
        ):
            plane_count = len(source_config.source_geometries[fault_name].planes)
            segments = srf_data.header.iloc[
                segment_counter : segment_counter + plane_count
            ]
            num_points = (segments["nstk"] * segments["ndip"]).sum()
            individual_moment_rate = moment.moment_rate_over_time_from_slip(
                srf_data.points["area"]
                .iloc[point_counter : point_counter + num_points]
                .to_numpy(),
                srf_data.slip[point_counter : point_counter + num_points],
                dt,
                srf_data.nt,
                mu,
            )
            plt.plot(
                individual_moment_rate.index.values,
                individual_moment_rate["moment_rate"],
                label=fault_name,
            )
            segment_counter += plane_count
            point_counter += num_points

    plt.ylabel("Moment Rate (Nm/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.title(f"Moment over Time (Total Mw: {magnitude:.2f})")

    plt.savefig(output_png_ffp, dpi=dpi)


def main():
    typer.run(plot_srf_moment)


if __name__ == "__main__":
    main()
