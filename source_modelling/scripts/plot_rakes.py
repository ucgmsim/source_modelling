"""Plot multi-segment rupture with time-slip-rise"""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from pygmt_helper import plotting

from source_modelling import srf


def plot_rakes(
    srf_ffp: Annotated[
        Path, typer.Argument(help="Path to SRF file to plot.", exists=True)
    ],
    output_ffp: Annotated[
        Path, typer.Argument(help="Output plot image.", dir_okay=False)
    ],
    dpi: Annotated[
        float, typer.Option(help="Plot output DPI (higher is better).")
    ] = 300,
    title: Annotated[str, typer.Option(help="Plot title to use.")] = "Title",
    sample_size: Annotated[
        int, typer.Option(help="Number of points to sample for rake.")
    ] = 200,
    vector_length: Annotated[
        float, typer.Option(help="Length of rake vectors (cm).")
    ] = 0.2,
):
    srf_data = srf.read_srf(srf_ffp)
    region = (
        srf_data.points["lon"].min() - 0.5,
        srf_data.points["lon"].max() + 0.5,
        srf_data.points["lat"].min() - 0.25,
        srf_data.points["lat"].max() + 0.25,
    )

    fig = plotting.gen_region_fig(title, region=region, map_data=None)
    i = 0
    vectors = srf_data.points[["lon", "lat", "rake"]].sample(sample_size)
    vectors["rake"] = (vectors["rake"] + 90) % 360
    vectors["length"] = vector_length
    fig.plot(
        data=vectors.values.tolist(), style="v0.1c+e+a30", pen="0.2p", fill="black"
    )
    for _, segment in srf_data.header.iterrows():
        nstk = int(segment["nstk"])
        ndip = int(segment["ndip"])
        point_count = nstk * ndip
        segment_points = srf_data.points.iloc[i : i + point_count]
        breakpoint()
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
    typer.run(plot_rakes)


if __name__ == "__main__":
    main()
