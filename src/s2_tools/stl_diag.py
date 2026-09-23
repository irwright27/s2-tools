from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from s2_tools.stl import decompose_stl, stl_parameters


def compare_ns(
    series: pd.Series,
    ns_values: Sequence[int],
    period: int = 365,
    inner_iter: int = 2,
    outer_iter: int = 5,
) -> dict[int, pd.DataFrame]:
    """
    Run STL decomposition for multiple candidate seasonal
    smoothing parameters.

    All STL parameters other than ns are held constant or
    calculated according to the rules implemented in stl.py.

    Parameters
    ----------
    series : pandas.Series
        Complete, regularly spaced time series.

    ns_values : sequence of int
        Candidate seasonal smoothing parameters.

    period : int, default 365
        Number of observations in one seasonal cycle.

    inner_iter : int, default 2
        Number of STL inner-loop iterations.

    outer_iter : int, default 5
        Number of STL robustness iterations.

    Returns
    -------
    dict[int, pandas.DataFrame]
        Dictionary keyed by ns containing the decomposition
        returned by decompose_stl().
    """
    if not ns_values:
        raise ValueError("ns_values cannot be empty.")

    results = {}

    for ns in ns_values:
        results[ns] = decompose_stl(
            series=series,
            ns=ns,
            period=period,
            inner_iter=inner_iter,
            outer_iter=outer_iter,
        )

    return results


def ns_parameter_table(
    ns_values: Sequence[int],
    period: int = 365,
    inner_iter: int = 2,
    outer_iter: int = 5,
) -> pd.DataFrame:
    """
    Return the complete STL parameter set associated with
    each candidate ns value.
    """
    rows = []

    for ns in ns_values:
        params = stl_parameters(
            ns=ns,
            period=period,
            inner_iter=inner_iter,
            outer_iter=outer_iter,
        )

        rows.append(
            {
                "ns": ns,
                "np": params["period"],
                "nt": params["trend"],
                "nl": params["low_pass"],
                "ni": params["inner_iter"],
                "no": params["outer_iter"],
            }
        )

    return pd.DataFrame(rows)


def plot_ns_components(
    results: dict[int, pd.DataFrame],
    component: str = "seasonal",
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """
    Plot the same STL component for multiple candidate ns values.

    Parameters
    ----------
    results : dict
        Output from compare_ns().

    component : {"trend", "seasonal", "remainder"}
        STL component to compare.

    figsize : tuple
        Figure size.
    """
    valid_components = {
        "trend",
        "seasonal",
        "remainder",
    }

    if component not in valid_components:
        raise ValueError(
            f"component must be one of {sorted(valid_components)}."
        )

    n = len(results)

    if n == 0:
        raise ValueError("results is empty.")

    fig, axes = plt.subplots(
        n,
        1,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    if n == 1:
        axes = [axes]

    for ax, (ns, result) in zip(
        axes,
        sorted(results.items()),
    ):
        ax.plot(
            result.index,
            result[component],
            linewidth=0.8,
        )

        ax.set_ylabel(f"ns={ns}")

    axes[0].set_title(
        f"STL {component}: comparison of seasonal smoothing"
    )

    axes[-1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()


def seasonal_by_year(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reshape an STL seasonal component into a day-of-year by year
    table.

    Rows represent day of year and columns represent years.

    February 29 is removed so that all years use a common
    365-day seasonal coordinate.
    """
    if "seasonal" not in result:
        raise ValueError(
            "result must contain a 'seasonal' column."
        )

    seasonal = result["seasonal"].copy()

    frame = seasonal.to_frame()

    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    frame["day"] = frame.index.day

    # Remove leap day so all years share a 365-day cycle.
    frame = frame[
        ~(
            (frame["month"] == 2)
            & (frame["day"] == 29)
        )
    ]

    frame["doy"] = frame.index.dayofyear

    # Correct day-of-year after February in leap years.
    leap_year = frame.index.is_leap_year
    after_february = frame.index.month > 2

    frame.loc[
        leap_year & after_february,
        "doy",
    ] -= 1

    return frame.pivot(
        index="doy",
        columns="year",
        values="seasonal",
    )


def plot_seasonal_by_year(
    result: pd.DataFrame,
    figsize: tuple[float, float] = (12, 5),
) -> None:
    """
    Plot the estimated seasonal component for each year
    against a common day-of-year axis.

    This makes year-to-year changes in the seasonal component
    directly visible.
    """
    seasonal = seasonal_by_year(result)

    fig, ax = plt.subplots(
        figsize=figsize,
    )

    for year in seasonal.columns:
        ax.plot(
            seasonal.index,
            seasonal[year],
            label=str(year),
            linewidth=1.0,
        )

    ax.set(
        title="STL seasonal component by year",
        xlabel="Day of year",
        ylabel="Seasonal component",
    )

    ax.legend(
        title="Year",
        ncol=2,
    )

    plt.tight_layout()
    plt.show()


def plot_ns_seasonal_by_year(
    results: dict[int, pd.DataFrame],
    figsize: tuple[float, float] = (12, 10),
) -> None:
    """
    Compare year-to-year seasonal behavior across candidate
    ns values.

    Each panel represents one candidate ns. Within each panel,
    individual lines represent years.
    """
    n = len(results)

    if n == 0:
        raise ValueError("results is empty.")

    fig, axes = plt.subplots(
        n,
        1,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    if n == 1:
        axes = [axes]

    for ax, (ns, result) in zip(
        axes,
        sorted(results.items()),
    ):
        seasonal = seasonal_by_year(result)

        for year in seasonal.columns:
            ax.plot(
                seasonal.index,
                seasonal[year],
                linewidth=0.8,
                label=str(year),
            )

        ax.set_ylabel(f"ns={ns}")

    axes[0].set_title(
        "Seasonal component by year across candidate ns values"
    )

    axes[-1].set_xlabel("Day of year")

    axes[0].legend(
        title="Year",
        ncol=2,
    )

    plt.tight_layout()
    plt.show()


def remainder_stats(
    results: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarize remainder magnitude for each candidate ns.

    These statistics are descriptive diagnostics only and
    should not be used alone to select ns.
    """
    rows = []

    for ns, result in sorted(results.items()):
        remainder = result["remainder"]

        rows.append(
            {
                "ns": ns,
                "remainder_mean": remainder.mean(),
                "remainder_sd": remainder.std(),
                "remainder_mae": remainder.abs().mean(),
                "remainder_rmse": np.sqrt(
                    np.mean(remainder**2)
                ),
            }
        )

    return pd.DataFrame(rows)