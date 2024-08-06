"""Plot multi-segment rupture with time-slip-rise"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from pygmt_helper import plotting

from source_modelling import srf


def main(
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
    ] = 3,
):
    srf_data = srf.read_srf(srf_ffp)
    region = (
        srf_data.points["lon"].min() - 0.5,
        srf_data.points["lon"].max() + 0.5,
        srf_data.points["lat"].min() - 0.25,
        srf_data.points["lat"].max() + 0.25,
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
    i = 0
    tinit_max = srf_data.points["tinit"].max()
    for _, segment in srf_data.header.iterrows():
        nstk = int(segment["nstk"])
        ndip = int(segment["ndip"])
        point_count = nstk * ndip
        segment_points = srf_data.points.iloc[i : i + point_count]
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
        fig.grdcontour(
            levels=levels,
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


if __name__ == "__main__":
    typer.run(main)
