"""Magnitude scaling relationships for fault dimensions."""

import functools
import warnings
from enum import Enum, StrEnum, auto
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp


class RakeType(Enum):
    """Enumeration of rake types."""

    NORMAL = auto()
    REVERSE = auto()
    STRIKE_SLIP = auto()
    REVERSE_OBLIQUE = auto()
    NORMAL_OBLIQUE = auto()


class ScalingRelation(StrEnum):
    """Enumeration of scaling relations."""

    LEONARD2014 = auto()
    CONTRERAS_INTERFACE2017 = auto()
    CONTRERAS_SLAB2020 = auto()


MAGNITUDE_BOUNDS = {
    ScalingRelation.LEONARD2014: (4.0, 9.0),
    ScalingRelation.CONTRERAS_INTERFACE2017: (6.0, 9.0),
    ScalingRelation.CONTRERAS_SLAB2020: (5.9, 7.8),
}

Array = float | npt.NDArray[np.floating] | pd.Series | pd.DataFrame
TArray = TypeVar("TArray", bound=Array)


def rake_type(rake: float) -> RakeType:
    """Determine the rake type of a fault given its rake.

    Parameters
    ----------
    rake : float
        Rake of the fault.

    Returns
    -------
    RakeType
        Type of rake of the fault.
    """
    rake %= 360
    if (
        (0 <= rake <= 30) or (150 <= rake <= 210) or (330 <= rake < 360)
    ):  # Use < 360 because modulo maps 360 to 0
        return RakeType.STRIKE_SLIP
    elif 60 <= rake <= 120:
        return RakeType.REVERSE
    elif 240 <= rake <= 300:
        return RakeType.NORMAL
    elif (210 < rake < 240) or (300 < rake < 330):
        return RakeType.NORMAL_OBLIQUE
    else:
        return RakeType.REVERSE_OBLIQUE


def leonard_area_to_magnitude(area: float, rake: float, random: bool = False) -> float:
    """Convert area to magnitude using the leonard scaling relationship [0]_.

    Parameters
    ----------
    area : float
        Area of the fault (km^2).
    rake : float
        Rake of the fault (degrees).
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the *best-fit* values. Default is False.

    Returns
    -------
    float
        Moment magnitude of the fault.

    Notes
    -----
    If the rake is not strike-slip, the uncertainties are assymetric.
    Hence, setting `random = False` is not equivalent to using the
    mean value for the parameters for all faults. Parameter values are
    found in Table 4 of [0]_.

    References
    ----------
    .. [0] Leonard, Mark. "Self‐consistent earthquake fault‐scaling
           relations: Update and extension to stable continental strike‐slip
           faults." Bulletin of the Seismological Society of America 104.6
           (2014): 2953-2965.
    """

    if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
        return np.log10(area) + (
            sp.stats.norm(loc=3.99, scale=0.26).rvs() if random else 3.99
        )

    # Leonard quotes assymetric uncertainties for the other rake types.
    return np.log10(area) + (
        sp.stats.norm(loc=4.03, scale=0.3).rvs() if random else 4.0
    )


def leonard_magnitude_to_area(
    magnitude: TArray, rake: float, random: bool = False
) -> TArray:
    """Convert magnitude to area using the Leonard scaling relationship [0]_.

    Parameters
    ----------
    magnitude : TArray
        Moment magnitude of the fault.
    rake : float
        Rake of the fault (degrees).
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the *best-fit* values. Default is False.

    Returns
    -------
    TArray
        Area of the fault. (km^2)

    Notes
    -----
    If the rake is not strike-slip, the uncertainties are assymetric.
    Hence, setting `random = False` is not equivalent to using the
    mean values.

    References
    ----------
    .. [0] Leonard, Mark. "Self‐consistent earthquake fault‐scaling
           relations: Update and extension to stable continental strike‐slip
           faults." Bulletin of the Seismological Society of America 104.6
           (2014): 2953-2965.
    """
    if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
        return 10 ** (
            magnitude - (sp.stats.norm(loc=3.99, scale=0.26).rvs() if random else 3.99)
        )

    # Leonard quotes asymmetric uncertainties for the other rake types.
    return 10 ** (
        magnitude - (sp.stats.norm(loc=4.03, scale=0.3).rvs() if random else 4.0)
    )


def leonard_magnitude_to_length(
    magnitude: TArray,
    rake: float,
    random: bool = False,
) -> TArray:
    """Convert magnitude to length using the Leonard scaling relationship [0]_.

    Parameters
    ----------
    magnitude : TArray
            Moment magnitude of the fault.
    rake : float
            Rake of the fault (degrees).
    random : bool, optional
            If True, sample parameters according to uncertainties in the
            paper, otherwise use the mean values. Default is False.

    Returns
    -------
    TArray
            Length of the fault. (km)

    References
    ----------
    .. [0] Leonard, Mark. "Self‐consistent earthquake fault‐scaling
           relations: Update and extension to stable continental strike‐slip
           faults." Bulletin of the Seismological Society of America 104.6
           (2014): 2953-2965.

    """
    a_strike_slip_small = sp.stats.norm(loc=4.16, scale=0.39).rvs() if random else 4.17
    b_strike_slip_small = 1.667
    a_strike_slip_large = 5.27
    b_strike_slip_large = 1.0
    length: TArray
    if rake_type(rake) == RakeType.STRIKE_SLIP:
        length = 10 ** ((magnitude - a_strike_slip_small) / b_strike_slip_small)
        length = np.where(
            length > 45.0,
            10 ** ((magnitude - a_strike_slip_large) / b_strike_slip_large),
            length,
        )
    else:
        a_dip_slip_small = 4
        b_dip_slip_small = 2
        a_dip_slip_large = 4.24
        b_dip_slip_large = 1.667

        length = 10 ** ((magnitude - a_dip_slip_small) / b_dip_slip_small)
        length = np.where(
            length > 5.4,
            10 ** ((magnitude - a_dip_slip_large) / b_dip_slip_large),
            length,
        )

    return length


def leonard_magnitude_to_width(
    magnitude: TArray,
    rake: float,
    random: bool = False,
) -> TArray:
    """Convert magnitude to width using the Leonard scaling relationship [0]_.

    Parameters
    ----------
    magnitude : TArray
            Moment magnitude of the fault.
    rake : float
            Rake of the fault (degrees).
    random : bool, optional
            If True, sample parameters according to uncertainties in the
            paper, otherwise use the mean values. Default is False.

    Returns
    -------
    TArray
            Width of the fault. (km)

    Warns
    -----
    UserWarning
            If the width is out of range for the Leonard model.

    References
    ----------
    .. [0] Leonard, Mark. "Self‐consistent earthquake fault‐scaling
           relations: Update and extension to stable continental strike‐slip
           faults." Bulletin of the Seismological Society of America 104.6
           (2014): 2953-2965.

    """
    a_strike_slip_small = (
        sp.stats.norm(loc=3.885, scale=0.065).rvs() if random else 3.88
    )
    b_strike_slip_small = 2.5
    width: TArray
    if rake_type(rake) == RakeType.STRIKE_SLIP:
        width = 10 ** ((magnitude - a_strike_slip_small) / b_strike_slip_small)
        if np.any((width > 19.0) | (width < 3.4)):
            warnings.warn("Width out of range for Leonard model.")
    else:
        a_dip_slip_small = sp.stats.norm(loc=3.67, scale=0.06).rvs() if random else 3.63
        b_dip_slip_small = 2.5
        width = 10 ** ((magnitude - a_dip_slip_small) / b_dip_slip_small)
        if np.any(width <= 5.4):
            warnings.warn("Width out of range for Leonard model.")

    return width


def leonard_magnitude_to_length_width(
    magnitude: TArray,
    rake: float,
    random: bool = False,
) -> tuple[TArray, TArray]:
    """Convert magnitude to length and width using the Leonard scaling relationship [0]_.

    Parameters
    ----------
    magnitude : TArray
        Moment magnitude of the fault.
    rake : float
        Rake of the fault (degrees).
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Returns
    -------
    tuple[TArray, TArray]
        Length and width of the fault.

    References
    ----------
    .. [0] Leonard, Mark. "Self‐consistent earthquake fault‐scaling
           relations: Update and extension to stable continental strike‐slip
           faults." Bulletin of the Seismological Society of America 104.6
           (2014): 2953-2965.
    """
    area = leonard_magnitude_to_area(magnitude, rake, random)
    length = leonard_magnitude_to_length(magnitude, rake, random)
    width = leonard_magnitude_to_width(magnitude, rake, random)
    aspect_ratio = np.maximum(length / width, 1.0)
    return area_aspect_ratio_to_length_width(area, aspect_ratio)


def area_aspect_ratio_to_length_width(
    area: TArray, aspect_ratio: TArray
) -> tuple[TArray, TArray]:
    """Convert area and aspect ratio to length and width.

    Parameters
    ----------
    area : TArray
        Area of the fault (km^2).
    aspect_ratio : TArray
        Aspect ratio of the fault (length / width).

    Returns
    -------
    tuple[TArray, TArray]
        Length and width of the fault.
    """
    width = np.sqrt(area / aspect_ratio)
    length = area / width
    return length, width


def contreras_interface_area_to_magnitude(area: TArray, random: bool = False) -> TArray:
    """Convert area to magnitude using the Contreras scaling relationship [0]_.

    Parameters
    ----------
    area : TArray
        Area of the fault (km^2).
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Returns
    -------
    TArray
        Moment magnitude of the fault.

    Warns
    -----
    UserWarning
        If the area is less than the minimum area of 137.76 km^2.

    Notes
    -----
    This scaling relationship is for *interface* events. For slab events,
    use the original Strasser scaling relationship [1]_.

    References
    ----------
    .. [0] Contreras, Victor, et al. "NGA-Sub source and path database." Earthquake Spectra 38.2 (2022): 799-840.
    .. [1] Strasser, Fleur O., M. C. Arango, and Julian J. Bommer.
           "Scaling of the source dimensions of interface and intraslab
           subduction-zone earthquakes with moment magnitude." Seismological
           Research Letters 81.6 (2010): 941-950.
    """
    if np.any(area < 137.76):
        warnings.warn(
            "Area out of range for Contreras model, minimum area is 137.76 km^2"
        )
    sigma_a = sp.stats.norm(loc=0, scale=0.73).rvs() if random else 0
    a_1 = -8.890
    a_2 = np.log(10)
    return 1 / a_2 * (np.log(area) - a_1 - sigma_a)


def contreras_interface_magnitude_to_area(
    magnitude: TArray, random: bool = False
) -> TArray:
    """Convert magnitude to area using the Contreras scaling relationship [0]_.

    Parameters
    ----------
    magnitude : TArray
        Moment magnitude of the fault.
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Returns
    -------
    TArray
        Area of the fault. (km^2)

    Warns
    -----
    UserWarning
        If the magnitude is less than the minimum magnitude of 6.

    Notes
    -----
    This relationship is for *interface* events. For slab events, use the
    Strasser scaling relationship [1]_.

    References
    ----------
    .. [0] Contreras, Victor, et al. "NGA-Sub source and path database." Earthquake Spectra 38.2 (2022): 799-840.
    .. [1] Strasser, Fleur O., M. C. Arango, and Julian J. Bommer.
           "Scaling of the source dimensions of interface and intraslab
           subduction-zone earthquakes with moment magnitude." Seismological
           Research Letters 81.6 (2010): 941-950.
    """
    if np.any(magnitude < 6):
        warnings.warn(
            "Magnitude out of range for Contreras model, minimum magnitude is 6"
        )
    sigma_a = sp.stats.norm(loc=0, scale=0.73).rvs() if random else 0
    a_1 = -8.890
    a_2 = np.log(10)
    return np.exp(a_1 + a_2 * magnitude + sigma_a)


def contreras_interface_magnitude_to_aspect_ratio(
    magnitude: TArray, random: bool = False
) -> TArray:
    """Convert magnitude to aspect ratio (L/W) using the Contreras scaling relationship [0]_.

    Parameters
    ----------
    magnitude : TArray
        Moment magnitude of the fault.
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Returns
    -------
    TArray
        Aspect ratio of the fault.

    Warns
    -----
    UserWarning
        If the magnitude is less than the minimum magnitude of 6.

    References
    ----------
    .. [0] Contreras, Victor, et al. "NGA-Sub source and path database." Earthquake Spectra 38.2 (2022): 799-840.
    """
    if magnitude < 6:
        warnings.warn(
            "Magnitude out of range for Contreras model, minimum magnitude is 6"
        )
    a_3 = 0.6248
    m_1 = 7.25
    sigma_1 = 0.32
    sigma_2 = 0.47
    if magnitude < m_1:
        return np.exp(sp.stats.norm(loc=0, scale=sigma_1).rvs() if random else 0)

    return np.exp(
        a_3 * (magnitude - m_1)
        + (sp.stats.norm(loc=0, scale=sigma_2).rvs() if random else 0)
    )


def contreras_interface_magnitude_to_length_width(
    magnitude: TArray, random: bool = False
) -> tuple[TArray, TArray]:
    """Convert magnitude to length and width using the Contreras scaling relationship [0]_.

    Parameters
    ----------
    magnitude : TArray
            Moment magnitude of the fault.
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Returns
    -------
    tuple[TArray, TArray]
            Length and width of the fault.

    References
    ----------
    .. [0] Contreras, Victor, et al. "NGA-Sub source and path database." Earthquake Spectra 38.2 (2022): 799-840.

    """
    area = contreras_interface_magnitude_to_area(magnitude, random)
    aspect_ratio = contreras_interface_magnitude_to_aspect_ratio(magnitude, random)
    return area_aspect_ratio_to_length_width(area, aspect_ratio)


def strasser_slab_area_to_magnitude(area: TArray, random: bool = False) -> TArray:
    """Convert area to magnitude using the Strasser scaling relationship [0]_.

    Parameters
    ----------
    area : TArray
        Area of the fault (km^2).
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Returns
    -------
    TArray
        Moment magnitude of the fault.

    Warns
    -----
    UserWarning
        If the area is not between the minimum and maximum area of the Strasser model, estimated at 130km^2 and 5212km^2.

    References
    ----------
    .. [0] Strasser, Fleur O., M. C. Arango, and Julian J. Bommer.
           "Scaling of the source dimensions of interface and intraslab
           subduction-zone earthquakes with moment magnitude." Seismological
           Research Letters 81.6 (2010): 941-950.
    """

    # lower bound and upper bound for the area are estimated from the minimum and maximum magnitude of the Strasser model.
    lower_bound = 130
    upper_bound = 5212
    if np.any((area < lower_bound) | (area > upper_bound)):
        warnings.warn(
            f"Area out of range for Strasser model, area must be between {lower_bound} and {upper_bound} km^2."
        )

    a = 4.054
    b = 0.981
    sigma_a = sp.stats.norm(loc=0, scale=0.288).rvs() if random else 0
    sigma_b = sp.stats.norm(loc=0, scale=0.093).rvs() if random else 0

    return a + sigma_a + (b + sigma_b) * np.log10(area)


def strasser_slab_magnitude_to_area(magnitude: TArray, random: bool = False) -> TArray:
    """Convert magnitude to area using the Strasser scaling relationship [0]_.

    Parameters
    ----------
    magnitude : TArray
        Moment magnitude of the fault.
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Returns
    -------
    TArray
        Area of the fault. (km^2)

    Warns
    -----
    UserWarning
        If the magnitude is less than the minimum magnitude of 5.9 or
        larger than the maximum magnitude of 7.8.

    References
    ----------
    .. [0] Strasser, Fleur O., M. C. Arango, and Julian J. Bommer.
           "Scaling of the source dimensions of interface and intraslab
           subduction-zone earthquakes with moment magnitude." Seismological
           Research Letters 81.6 (2010): 941-950.
    """
    if np.any((magnitude < 5.9) | (magnitude > 7.8)):
        warnings.warn(
            "Magnitude out of range for Strasser model, magnitude must be between 5.9 and 7.8"
        )
    a = -3.225
    sigma_a = sp.stats.norm(loc=0, scale=0.598).rvs() if random else 0
    b = 0.890
    sigma_b = sp.stats.norm(loc=0, scale=0.085).rvs() if random else 0
    return 10 ** (a + sigma_a + (b + sigma_b) * magnitude)  # type: ignore[invalid-return-type]


def contreras_slab_magnitude_to_aspect_ratio(
    magnitude: TArray, random: bool = False
) -> TArray:
    """Convert magnitude to aspect ratio (L/W) using the Contreras scaling relationship [0]_.

    Parameters
    ----------
    magnitude : TArray
        Moment magnitude of the fault.
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Returns
    -------
    TArray
        Aspect ratio of the fault.

    Warns
    -----
    UserWarning
        If the magnitude is less than the minimum magnitude of 5.

    References
    ----------
    .. [0] Bozorgnia, Y., & Stewart, J.P. (2020). Data Resources for
           NGA-Subduction Project (Report No. 2020/02). Pacific Earthquake
           Engineering Research Center (PEER).
           https://doi.org/10.55461/RDWC6463
    """
    if np.any(magnitude < 5):
        warnings.warn(
            "Magnitude out of range for Contreras model, minimum magnitude is 5"
        )
    a_3 = 0.216
    m_1 = 6.5
    sigma_1 = 0.24
    sigma_2 = 0.38
    return np.where(
        magnitude < m_1,
        np.exp(
            sp.stats.norm(loc=0, scale=sigma_1).rvs(size=magnitude.size)
            if random
            else 0
        ),
        np.exp(
            a_3 * (magnitude - m_1)
            + (sp.stats.norm(loc=0, scale=sigma_2).rvs(magnitude.size) if random else 0)
        ),
    )


def contreras_slab_magnitude_to_length_width(
    magnitude: TArray, random: bool = False
) -> tuple[TArray, TArray]:
    """Convert magnitude to length and width using the Contreras scaling relationship [0]_.

    Parameters
    ----------
    magnitude : TArray
        Moment magnitude of the fault.
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Returns
    -------
    tuple[TArray, TArray]
        Length and width of the fault.

    References
    ----------
    .. [0] Bozorgnia, Y., & Stewart, J.P. (2020). Data Resources for
           NGA-Subduction Project (Report No. 2020/02). Pacific Earthquake
           Engineering Research Center (PEER).
           https://doi.org/10.55461/RDWC6463
    """
    area = strasser_slab_magnitude_to_area(magnitude, random)
    aspect_ratio = contreras_slab_magnitude_to_aspect_ratio(magnitude, random)
    return area_aspect_ratio_to_length_width(area, aspect_ratio)


def magnitude_to_length_width(
    scaling_relation: ScalingRelation,
    magnitude: TArray,
    rake: float | None = None,
    random: bool = False,
) -> tuple[TArray, TArray]:
    """Convert magnitude to length and width using a scaling relationship.

    Parameters
    ----------
    scaling_relation : ScalingRelation
        Scaling relation to use.
    magnitude : TArray
        Moment magnitude of the fault.
    rake : float, optional
        Rake of the fault (degrees). Required for Leonard scaling.
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Returns
    -------
    tuple[TArray, TArray]
            Length and width of the fault estimated by the scaling relation.
    """
    if scaling_relation == ScalingRelation.LEONARD2014 and rake is None:
        warnings.warn("Rake must be specified for Leonard scaling.")
    scaling_relations_map = {
        ScalingRelation.LEONARD2014: functools.partial(
            leonard_magnitude_to_length_width, rake=rake, random=random
        ),
        ScalingRelation.CONTRERAS_INTERFACE2017: functools.partial(
            contreras_interface_magnitude_to_length_width, random=random
        ),
        ScalingRelation.CONTRERAS_SLAB2020: functools.partial(
            contreras_slab_magnitude_to_length_width, random=random
        ),
    }
    return scaling_relations_map[scaling_relation](magnitude)


def magnitude_to_area(
    scaling_relation: ScalingRelation,
    magnitude: TArray,
    rake: float | None = None,
    random: bool = False,
) -> TArray:
    """Convert magnitude to area using a scaling relationship.

    Parameters
    ----------
    scaling_relation : ScalingRelation
        Scaling relation to use.
    magnitude : TArray
        Moment magnitude of the fault.
    rake : float, optional
        Rake of the fault (degrees). Required for Leonard scaling.
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Raises
    ------
    ValueError
        If `scaling_relation` is `LEONARD2014` and `rake` is not specified.

    Returns
    -------
    TArray
        Area of the fault estimated by the scaling relation.
    """
    if scaling_relation == ScalingRelation.LEONARD2014 and rake is None:
        raise ValueError("Rake must be specified for Leonard scaling.")
    scaling_relations_map = {
        ScalingRelation.LEONARD2014: functools.partial(
            leonard_magnitude_to_area, rake=rake, random=random
        ),
        ScalingRelation.CONTRERAS_INTERFACE2017: functools.partial(
            contreras_interface_magnitude_to_area, random=random
        ),
        ScalingRelation.CONTRERAS_SLAB2020: functools.partial(
            strasser_slab_magnitude_to_area, random=random
        ),
    }
    return scaling_relations_map[scaling_relation](magnitude)


def area_to_magnitude(
    scaling_relation: ScalingRelation,
    area: TArray,
    rake: float | None = None,
    random: bool = False,
) -> TArray:
    """Convert area to magnitude using a scaling relationship.

    Parameters
    ----------
    scaling_relation : ScalingRelation
        Scaling relation to use.
    area : TArray
        Area of the fault (km^2).
    rake : float, optional
        Rake of the fault (degrees). Required for Leonard scaling.
    random : bool, optional
        If True, sample parameters according to uncertainties in the
        paper, otherwise use the mean values. Default is False.

    Returns
    -------
    TArray
        Moment magnitude of the fault estimated by the scaling relation.
    """
    scaling_relations_map = {
        ScalingRelation.LEONARD2014: functools.partial(
            leonard_area_to_magnitude, rake=rake, random=random
        ),
        ScalingRelation.CONTRERAS_INTERFACE2017: functools.partial(
            contreras_interface_area_to_magnitude, random=random
        ),
        ScalingRelation.CONTRERAS_SLAB2020: functools.partial(
            strasser_slab_area_to_magnitude, random=random
        ),
    }
    return scaling_relations_map[scaling_relation](area)
