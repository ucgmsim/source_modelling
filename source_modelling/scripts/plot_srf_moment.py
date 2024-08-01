"""Utility script to plot moment over time for an SRF."""

import functools
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import scipy as sp
import typer
from matplotlib import pyplot as plt

from source_modelling import srf


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
):
    """Plot released moment for an SRF over time."""
    srf_data = srf.read_srf(srf_ffp)

    non_neglible_patches = srf_data.points[srf_data.points["slipt1"].apply(len) > 0]
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
    magnitude = (
        2
        / 3
        * np.log10(
            mu * (srf_data.points["area"] * srf_data.points["slip1"] / (100**3)).sum()
        )
        - 6.03333
    )
    plt.plot(
        binned_and_summed_moments.index.values, binned_and_summed_moments["moment"]
    )
    plt.ylabel("Moment (Nm)")
    plt.xlabel("Time (s)")
    plt.title(f"Moment over Time (Total Mw: {magnitude:.2f})")
    plt.savefig(output_png_ffp, dpi=dpi)


def main():
    typer.run(plot_srf_moment)


if __name__ == "__main__":
    main()
