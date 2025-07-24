import pandas as pd

from source_modelling import moment

# def dyne_cm_to_newton_metre(dyne_cm: float) -> float:
#     """Convert dyne-cm units to standard SI units Nm.

#     Parameters
#     ----------
#     dyne_cm : float
#         Force in dyne-cm.

#     Returns
#     -------
#     float
#         Force in Nm.
#     """
#     _dyne_to_newton = 1e-5
#     _cm_to_m = 1e-2
#     return dyne_cm * _dyne_to_newton * _cm_to_m


# def newton_metre_to_dyne_cm(newton_metre: float) -> float:
#     """Convert newton-metre units to dyne-cm.

#     Parameters
#     ----------
#     newton_metre : float
#         Force in newton-metre.

#     Returns
#     -------
#     float
#         Force in dyne-cm.
#     """
#     # Use inverse of dyne_cm_to_newton_metre: if 1 dyne-cm = X Nm, then 1 Nm = 1/X dyne-cm
#     return newton_metre / dyne_cm_to_newton_metre(1.0)


# def old_point_source_slip(
#     moment_dyne_cm: float,
#     fault_area_km2: float,
#     velocity_model_df: pd.DataFrame,
#     source_depth_km: float,
# ) -> float:
#     """Calculate slip for a point source.

#     This calculation is the same as in the old workflow:
#     https://github.com/ucgmsim/Pre-processing/blob/6572ea8be0963da4f7ac6a503ab07dd2519296e5/srf_generation/input_file_generation/realisation_to_srf.py#L321C9-L321C57

#     Parameters
#     ----------
#     moment_dyne_cm : float
#         The seismic moment in dyne-cm.
#     fault_area_km2 : float
#         The area of the fault in square kilometers.
#     velocity_model_df : pd.DataFrame
#       columns:
#         - "depth_km": The depth in kilometers.
#         - "rho_g_per_cm3": The density of the fault in grams per cubic centimeter.
#         - "vs_km_per_s": The shear wave velocity in kilometers per second.
#     source_depth_km : float
#         The depth of the source in kilometers.

#     Returns
#     -------
#     float
#         The calculated slip in cm.
#     """

#     # Find the index of the closest depth in the velocity model
#     idx = np.argmin(np.abs(velocity_model_df["depth_km"] - source_depth_km))
#     vs_km_per_s = velocity_model_df.iloc[idx]["Vs"]
#     rho_g_per_cm3 = velocity_model_df.iloc[idx]["rho"]

#     # The factor of 1.0e-20 converts the combination of input units to cm.
#     return (moment_dyne_cm * 1.0e-20) / (
#         fault_area_km2 * rho_g_per_cm3 * vs_km_per_s**2
#     )


# def moment_to_magnitude(moment: float) -> float:
#     """Convert moment to magnitude.

#     NOTE: the qcore mag_scaling module does not have this expression.

#     Parameters
#     ----------
#     moment : float
#         The moment of the rupture.

#     Returns
#     -------
#     float
#         Rupture magnitude
#     """
#     return 2 / 3 * np.log10(moment) - 6.03333


# def magnitude_to_moment(magnitude: float) -> float:
#     """Convert magnitude to moment.

#     NOTE: the qcore mag_scaling module does not have this expression.

#     Parameters
#     ----------
#     magnitude : float
#         The magnitude of the rupture.

#     Returns
#     -------
#     float
#         moment
#     """
#     return 10 ** ((magnitude + 6.03333) * 3 / 2)


# def point_source_slip(
#     moment_newton_metre: float,
#     fault_area_km2: float,
#     velocity_model_df: pd.DataFrame,
#     source_depth_km: float,
# ) -> float:
#     """Calculate slip for a point source.

#     This calculation is based on the old workflow:
#     https://github.com/ucgmsim/Pre-processing/blob/6572ea8be0963da4f7ac6a503ab07dd2519296e5/srf_generation/input_file_generation/realisation_to_srf.py#L321C9-L321C57

#     Parameters
#     ----------
#     moment_newton_metre : float
#         The seismic moment in newton-metre.
#     fault_area_km2 : float
#         The area of the fault in square kilometers.
#     velocity_model_df : pd.DataFrame
#       columns:
#         - "depth_km": The depth in kilometers.
#         - "rho_g_per_cm3": The density of the fault in grams per cubic centimeter.
#         - "vs_km_per_s": The shear wave velocity in kilometers per second.
#     source_depth_km : float
#         The depth of the source in kilometers.

#     Returns
#     -------
#     float
#         The calculated slip in cm.
#     """

#     # Find the index of the closest depth in the velocity model
#     idx = np.argmin(np.abs(velocity_model_df["depth_km"] - source_depth_km))
#     vs_km_per_s = velocity_model_df.iloc[idx]["Vs"]
#     rho_g_per_cm3 = velocity_model_df.iloc[idx]["rho"]

#     _vs_m_per_s = vs_km_per_s * 1e3  # Convert km/s to m/s
#     _rho_kg_per_m3 = rho_g_per_cm3 * 1e3  # Convert g/cm³ to kg/m³
#     _fault_area_m2 = fault_area_km2 * 1e6  # Convert km² to m²

#     _slip_m = moment_newton_metre / (_fault_area_m2 * _rho_kg_per_m3 * _vs_m_per_s**2)

#     return _slip_m * 1e2  # Convert m to cm


# moment_dyne_cm = 5e23
# moment_newton_metre = dyne_cm_to_newton_metre(moment_dyne_cm)
# check_moment_dyne_cm = newton_metre_to_dyne_cm(moment_newton_metre)

# fault_area_km2 = 100.0
# source_depth_km = 2.0

# old_slip_result_cm = old_point_source_slip(
#     moment_dyne_cm=moment_dyne_cm,
#     fault_area_km2=fault_area_km2,
#     velocity_model_df=velocity_model_df,
#     source_depth_km=source_depth_km,
# )

# slip_result_cm = point_source_slip(
#     moment_newton_metre=moment_newton_metre,
#     fault_area_km2=fault_area_km2,
#     velocity_model_df=velocity_model_df,
#     source_depth_km=source_depth_km,
# )


# test_magnitude = 6

# test_moment = moment.magnitude_to_moment(test_magnitude)

# check_magnitude = moment.moment_to_magnitude(test_moment)

# source_depth_km = 0.01
# source_depth_km = 2


# source_depth_km = 5
# fault_area_km2 = 100.0
# magnitude = 6

fault_area_km2 = 100

# moment_newton_metre = moment.magnitude_to_moment(magnitude)

# moment_newton_metre = moment.dyne_cm_to_newton_metre(2.2387211385683286e25)
# check_mag = moment.moment_to_magnitude(moment_newton_metre)


velocity_model_df = pd.DataFrame(
    {
        "depth_km": [0.05, 2, 1038.9999],
        "thickness": [0.05, 0.2, 999.9999],
        "Vp": [1.8, 3.27, 8.1],
        "Vs": [0.5, 1.64, 4.6],
        "rho": [1.81, 2.28, 3.33],
        "Qp": [38, 164, 460],
        "Qs": [19, 82, 230],
    }
)


moment_newton_metre = moment.magnitude_to_moment(5)
source_depth_km = 1040

calculated_slip = moment.point_source_slip(
    moment_newton_metre=moment_newton_metre,
    fault_area_km2=fault_area_km2,
    velocity_model_df=velocity_model_df,
    source_depth_km=source_depth_km,
)

print(calculated_slip)

print()
