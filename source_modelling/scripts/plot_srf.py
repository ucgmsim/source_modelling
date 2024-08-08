"""Plot multi-segment rupture with time-slip-rise."""

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer

from pygmt_helper import plotting
from qcore import coordinates
from source_modelling import srf
from workflow import realisations
from workflow.realisations import RupturePropagationConfig, SourceConfig


def plot_srf(
    srf_ffp: Annotated[
        Path, typer.Argument(help="Path to SRF file to plot.", exists=True)
    ],
    output_ffp: Annotated[
        Path, typer.Argument(help="Output plot image.", dir_okay=False)
    ],
    dpi: Annotated[
        float, typer.Option(help="Plot output DPI (higher is better)")
    ] = 300,
    title: Annotated[str, typer.Option(help="Plot title to use")] = "Title",
    levels: Annotated[
        float,
        typer.Option(
            help="Plot time as contours of every LEVELS seconds", metavar="LEVELS"
        ),
    ] = 1,
    realisation_ffp: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to realisation, used to mark jump points.", exists=True
        ),
    ] = None,
    latitude_pad: Annotated[
        float, typer.Option(help="Latitude padding to apply (degrees)")
    ] = 0,
    longitude_pad: Annotated[
        float, typer.Option(help="longitude padding to apply (degrees)")
    ] = 0,
    annotations: Annotated[bool, typer.Option(help="Label contours")] = True,
):
    srf_data = srf.read_srf(srf_ffp)
    region = (
        srf_data.points["lon"].min() - longitude_pad,
        srf_data.points["lon"].max() + longitude_pad,
        srf_data.points["lat"].min() - latitude_pad,
        srf_data.points["lat"].max() + latitude_pad,
    )

    srf_data.points["slip"] = np.sqrt(
        srf_data.points["slip1"] ** 2
        + srf_data.points["slip2"] ** 2
        + srf_data.points["slip3"] ** 2
    )
    slip_quantile = srf_data.points["slip"].quantile(0.98)
    slip_cb_max = max(int(np.round(slip_quantile, -1)), 10)
    cmap_limits = (0, slip_cb_max, slip_cb_max / 10)

    fig = plotting.gen_region_fig(title, region=region, map_data=None)

    for (_, segment), segment_points in zip(
        srf_data.header.iterrows(), srf_data.segments
    ):
        nstk = segment["nstk"]
        ndip = segment["ndip"]

        cur_grid = plotting.create_grid(
            segment_points,
            "slip",
            grid_spacing="5e/5e",
            region=(
                segment_points["lon"].min(),
                segment_points["lon"].max(),
                segment_points["lat"].min(),
                segment_points["lat"].max(),
            ),
            set_water_to_nan=False,
        )
        plotting.plot_grid(
            fig,
            cur_grid,
            "hot",
            cmap_limits,
            ("white", "black"),
            transparency=0,
            reverse_cmap=True,
            plot_contours=False,
            cb_label="slip",
            continuous_cmap=True,
        )
        time_grid = plotting.create_grid(
            segment_points,
            "tinit",
            grid_spacing="5e/5e",
            region=(
                segment_points["lon"].min(),
                segment_points["lon"].max(),
                segment_points["lat"].min(),
                segment_points["lat"].max(),
            ),
            set_water_to_nan=False,
        )
        fig.grdcontour(levels=1, grid=time_grid, pen="0.1p")

        corners = segment_points.iloc[[0, nstk - 1, -1, (ndip - 1) * nstk]]
        fig.plot(
            x=corners["lon"].iloc[list(range(len(corners))) + [0]].to_list(),
            y=corners["lat"].iloc[list(range(len(corners))) + [0]].to_list(),
            pen="0.5p,black,-",
        )
        fig.plot(
            x=corners["lon"].iloc[:2].to_list(),
            y=corners["lat"].iloc[:2].to_list(),
            pen="0.8p,black",
        )
        if not annotations:
            continue
        tinit_max = int(np.round(segment_points["tinit"].max()))
        tinit_min = int(np.round(segment_points["tinit"].min()))
        if tinit_max - tinit_min >= 1:
            for j in range(tinit_min, tinit_max):
                min_delta = (segment_points["tinit"] - j).abs().min()
                if min_delta < 0.1:
                    closest_point = segment_points[
                        (segment_points["tinit"] - j).abs() == min_delta
                    ].iloc[0]
                    fig.text(
                        text=f"{j}",
                        x=closest_point["lon"],
                        y=closest_point["lat"],
                        font="5p",
                        fill="white",
                    )

    hypocentre = srf_data.points[
        srf_data.points["tinit"] == srf_data.points["tinit"].min()
    ].iloc[0]
    fig.plot(
        x=hypocentre["lon"],
        y=hypocentre["lat"],
        style="a0.4c",
        pen="1p,black",
        fill="white",
    )
    if realisation_ffp:
        rupture_propagation_config: RupturePropagationConfig = (
            realisations.read_config_from_realisation(
                RupturePropagationConfig, realisation_ffp
            )
        )
        source_config: SourceConfig = realisations.read_config_from_realisation(
            SourceConfig, realisation_ffp
        )
        for fault_name, jump_point in rupture_propagation_config.jump_points.items():
            parent_name = rupture_propagation_config.rupture_causality_tree[fault_name]
            if not parent_name:
                continue
            source = source_config.source_geometries[fault_name]
            parent = source_config.source_geometries[parent_name]
            from_point = parent.fault_coordinates_to_wgs_depth_coordinates(
                jump_point.from_point
            )[:2]
            closest_from_point_distance_idx = (
                coordinates.distance_between_wgs_depth_coordinates(
                    srf_data.points[["lat", "lon"]].to_numpy(), from_point
                )
            ).argmin()
            srf_jump_point = srf_data.points.iloc[closest_from_point_distance_idx]
            to_point = source.fault_coordinates_to_wgs_depth_coordinates(
                jump_point.to_point
            )[:2]
            fig.plot(
                x=from_point[1],
                y=from_point[0],
                style="t0.4c",
                pen="1p,black",
                fill="white",
            )
            fig.text(
                x=srf_jump_point["lon"],
                y=srf_jump_point["lat"] - 0.01,
                font="5p",
                fill="white",
                text=f"t_jump = {srf_jump_point['tinit']:.2f}",
            )
            fig.plot(
                x=to_point[1],
                y=to_point[0],
                style="i0.4c",
                pen="1p,black",
                fill="white",
            )

    fig.savefig(
        output_ffp,
        dpi=dpi,
        anti_alias=True,
    )


def main():
    typer.run(plot_srf)


if __name__ == "__main__":
    main()
