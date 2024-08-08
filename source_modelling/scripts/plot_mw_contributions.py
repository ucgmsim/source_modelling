import collections
from pathlib import Path
from typing import Annotated, Generator

import numpy as np
import typer
from matplotlib import pyplot as plt

from source_modelling import srf
from workflow import realisations
from workflow.realisations import (
    RealisationMetadata,
    RupturePropagationConfig,
    SourceConfig,
)


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


def moment_to_mw(moment: float):
    return 2 / 3 * np.log10(moment) - 6.03333


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
    mu: Annotated[float, typer.Option(help="Shear rigidity constant")] = 3.3e10,
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
    log_area = np.linspace(np.log10(smallest_area), np.log10(total_area))
    # Mw = log(area) + 3.995 is the Leonard2014 magnitude scaling relation
    # for average rake.
    plt.plot(log_area, log_area + 3.995, label="Leonard 2014 Interplate (Average Rake)")

    srf_data = srf.read_srf(srf_ffp)
    srf_data.points["slip"] = np.sqrt(
        srf_data.points["slip1"] ** 2
        + srf_data.points["slip2"] ** 2
        + srf_data.points["slip3"] ** 2
    )
    total_magnitude = moment_to_mw(
        mu * (srf_data.points["area"] * srf_data.points["slip"] / (100**3)).sum()
    )
    plt.scatter(np.log10(total_area), total_magnitude, label="Total Magnitude")

    segment_counter = 0
    point_counter = 0
    for fault_name in tree_nodes_in_order(
        rupture_propogation_config.rupture_causality_tree
    ):
        plane_count = len(source_config.source_geometries[fault_name].planes)
        segments = srf_data.header.iloc[segment_counter : segment_counter + plane_count]
        num_points = (segments["nstk"] * segments["ndip"]).sum()
        individual_area = source_config.source_geometries[fault_name].area()
        segment_points = srf_data.points.iloc[
            point_counter : point_counter + num_points
        ]
        individual_magnitude = moment_to_mw(
            (mu * segment_points["area"] * segment_points["slip"] / (100**3)).sum()
        )
        plt.scatter(np.log10(individual_area), individual_magnitude, label=fault_name)
        segment_counter += plane_count
        point_counter += num_points
    plt.xlabel("Log Area")
    plt.ylabel("Mw")
    plt.legend()
    plt.title(f"Log Area vs Magnitude ({realisation_metadata.name})")
    plt.savefig(output_ffp, dpi=dpi)


def main():
    typer.run(plot_mw_contributions)


if __name__ == "__main__":
    main()
