"""Plot multi-segment rupture with time-slip-rise"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from pygmt_helper import plotting
from source_modelling import srf


def plot_rise(
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
):
    """Plot multi-segment rupture with time-slip-rise"""
    srf_data = srf.read_srf(srf_ffp)
    region = (
        srf_data.points["lon"].min() - 0.5,
        srf_data.points["lon"].max() + 0.5,
        srf_data.points["lat"].min() - 0.25,
        srf_data.points["lat"].max() + 0.25,
    )

    srf_data.points["trise"] = (
        srf_data.points[["slipt1", "slipt2", "slipt3"]].map(len).max(axis=1)
        * srf_data.points["dt"]
    )
    trise_cb_max = srf_data.points["trise"].max()
    cmap_limits = (0, trise_cb_max, trise_cb_max / 10)

    fig = plotting.gen_region_fig(title, region=region, map_data=None)
    i = 0
    for _, segment in srf_data.header.iterrows():
        nstk = int(segment["nstk"])
        ndip = int(segment["ndip"])
        point_count = nstk * ndip
        segment_points = srf_data.points.iloc[i : i + point_count]
        cur_grid = plotting.create_grid(
            segment_points,
            "trise",
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
            cb_label="trise",
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
        fig.grdcontour(
            levels=0.5,
            annotation=1,
            grid=time_grid,
            pen="0.1p",
        )
        corners = segment_points.iloc[[0, nstk - 1, -1, (ndip - 1) * nstk]]
        fig.plot(
            x=corners["lon"].iloc[list(range(len(corners))) + [0]].to_list(),
            y=corners["lat"].iloc[list(range(len(corners))) + [0]].to_list(),
            pen="0.5p,black,-",
        )

        i += point_count

    fig.savefig(
        output_ffp,
        dpi=dpi,
        anti_alias=True,
    )


def main():
    typer.run(main)


if __name__ == "__main__":
    main()
