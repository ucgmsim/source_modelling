from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from matplotlib import pyplot as plt

from source_modelling import moment, rupture_propagation, srf
from workflow import realisations
from workflow.realisations import (
    RealisationMetadata,
    RupturePropagationConfig,
    SourceConfig,
)


def plot_mw_contributions(
    srf_ffp: Annotated[
        Path, typer.Argument(help="Path to SRF file", exists=True, dir_okay=False)
    ],
    realisation_ffp: Annotated[
        Path,
        typer.Argument(help="Realisation filepath", dir_okay=False, exists=True),
    ],
    output_ffp: Annotated[
        Path, typer.Argument(help="Output plot path.", writable=True, dir_okay=False)
    ],
    mu: Annotated[float, typer.Option(help="Shear rigidity constant")] = moment.MU,
    dpi: Annotated[
        float, typer.Option(help="Output plot DPI (higher is better).")
    ] = 300,
) -> None:
    source_config: SourceConfig = realisations.read_config_from_realisation(
        SourceConfig, realisation_ffp
    )
    rupture_propogation_config: RupturePropagationConfig = (
        realisations.read_config_from_realisation(
            RupturePropagationConfig, realisation_ffp
        )
    )
    realisation_metadata: RealisationMetadata = (
        realisations.read_config_from_realisation(RealisationMetadata, realisation_ffp)
    )
    total_area = sum(fault.area() for fault in source_config.source_geometries.values())
    smallest_area = min(
        fault.area() for fault in source_config.source_geometries.values()
    )
    area = np.linspace(smallest_area, total_area)
    # Mw = log(area) + 3.995 is the Leonard2014 magnitude scaling relation
    # for average rake.
    plt.plot(
        area, np.log10(area) + 3.995, label="Leonard 2014 Interplate (Average Rake)"
    )

    srf_data = srf.read_srf(srf_ffp)
    total_magnitude = moment.moment_to_magnitude(
        mu * (srf_data.points["area"] * srf_data.points["slip"] / (100**3)).sum()
    )
    plt.scatter(total_area, total_magnitude, label="Total Magnitude")

    segment_counter = 0
    point_counter = 0
    for fault_name in rupture_propagation.tree_nodes_in_order(
        rupture_propogation_config.rupture_causality_tree
    ):
        plane_count = len(source_config.source_geometries[fault_name].planes)
        segments = srf_data.header.iloc[segment_counter : segment_counter + plane_count]

        num_points = (segments["nstk"] * segments["ndip"]).sum()
        individual_area = source_config.source_geometries[fault_name].area()

        # get all points associated with all segments in the current fault
        segment_points = srf_data.points.iloc[
            point_counter : point_counter + num_points
        ]
        individual_magnitude = moment.moment_to_magnitude(
            (mu * segment_points["area"] * segment_points["slip"] / (100**3)).sum()
        )
        plt.scatter(individual_area, individual_magnitude, label=fault_name)

        # advance segment counter and point counter to skip all points from the current point
        segment_counter += plane_count
        point_counter += num_points

    plt.xlabel("Area (m^2)")
    plt.ylabel("Mw")
    plt.xscale("log")
    plt.legend()
    plt.title(f"Log Area vs Magnitude ({realisation_metadata.name})")
    plt.savefig(output_ffp, dpi=dpi)


def main():
    typer.run(plot_mw_contributions)


if __name__ == "__main__":
    main()
