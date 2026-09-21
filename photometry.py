import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from photutils.centroids import centroid_2dg, centroid_sources

def calculate_optimal_aperture_snr(image_data, error_map, center, r_min=7.5, r_max=13, step=0.5):
    """
    Calculates the optimal aperture radius by maximizing Signal-to-Noise Ratio (SNR).
    
    Parameters:
    -----------
    image_data : 2D numpy array
        The science image containing the source.
    error_map : 2D numpy array
        The error/uncertainty map corresponding to image_data.
    center : tuple (x, y)
        Sub-pixel position of the target source centroid.
    r_min, r_max, step : float
        Aperture radius search space parameters.
        
    Returns:
    --------
    best_radius : float
        Optimal aperture radius maximizing SNR.
    max_snr : float
        Maximum SNR value obtained.
    radii : list
        Tested aperture radii.
    snr_values : list
        Calculated SNR values for each radius.
    """
    radii = np.arange(r_min, r_max + step, step)
    snr_values = []
    
    for r in radii:
        # Define background annulus parameters
        r_in = r + 6
        r_out = r_in + 8
        annulus = CircularAnnulus(center, r_in=r_in, r_out=r_out)
    
        # Calculate background stats
        annulus_mask = annulus.to_mask(method='center')
        annulus_data = annulus_mask.multiply(image_data)
        bg_vals = annulus_data[annulus_mask.data > 0]
        bg_mean = np.median(bg_vals)
        
        aperture = CircularAperture(center, r=r)
        
        # Photometry on Science Image and Error Map
        phot_table = aperture_photometry(image_data, aperture)
        err_table = aperture_photometry(error_map**2, aperture)  # Sum of variances
        
        # Net flux calculation (subtract background)
        raw_flux = phot_table['aperture_sum'][0]
        area = aperture.area
        net_flux = raw_flux - (bg_mean * area)
        
        # Total variance inside the aperture
        variance = err_table['aperture_sum'][0]
        
        # Calculate SNR
        if net_flux > 0 and variance > 0:
            snr = net_flux / np.sqrt(variance)
        else:
            snr = 0.0
            
        snr_values.append(snr)
        
    best_radius = radii[np.argmax(snr_values)]
    max_snr = np.max(snr_values)
    
    return best_radius, max_snr, radii, snr_values

"""
def estimate_fwhm(data, x, y, box_size=15):
   # Estimates PSF FWHM around a target location using 2D Gaussian moments.
    half_box = box_size // 2
    x_int, y_int = int(round(x)), int(round(y))

    # Extract sub-image cutout around star
    sub_img = data[
        max(0, y_int - half_box) : min(data.shape[0], y_int + half_box + 1),
        max(0, x_int - half_box) : min(data.shape[1], x_int + half_box + 1),
    ]

    # Calculate second moments to estimate sigma
    y_idx, x_idx = np.indices(sub_img.shape)
    total = np.sum(np.maximum(sub_img - np.median(sub_img), 0))
    if total <= 0:
        return 4.0  # Fallback standard FWHM in pixels

    x_mean = np.sum(x_idx * np.maximum(sub_img - np.median(sub_img), 0)) / total
    y_mean = np.sum(y_idx * np.maximum(sub_img - np.median(sub_img), 0)) / total

    x_var = (
        np.sum((x_idx - x_mean) ** 2 * np.maximum(sub_img - np.median(sub_img), 0))
        / total
    )
    y_var = (
        np.sum((y_idx - y_mean) ** 2 * np.maximum(sub_img - np.median(sub_img), 0))
        / total
    )

    sigma = np.sqrt((x_var + y_var) / 2.0)
    fwhm = sigma / gaussian_fwhm_to_sigma
    # Constrain FWHM within physical bounds (e.g., 2 to 10 pixels)
    return np.clip(fwhm, 2.0, 10.0)
"""

def run_improved_photometry(
    file_pattern="2026/red/AE_UMa-*_calibrated_mef.fits",
    initial_target_xy=(739.2, 717.7),
    initial_ref_xy=(1128.9, 509.1),
    ref_mag=11.32,
    ref_mag_err=0.08,
    min_flux_fraction=0.8,  # Quality Cut: Drop frames where ref flux drops below 80% median
    csv_output="AE_UMa_lightcurve_cleaned.csv",
):
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching pattern: {file_pattern}")

    print(f"Processing {len(files)} files with 2D Gaussian Centroiding & Dynamic Apertures...")

    results = []
    curr_target_xy = initial_target_xy
    curr_ref_xy = initial_ref_xy

    # -------------------------------------------------------------------------
    # 1. First Pass: Photometry & Centroiding with Dynamic Growth-Matched Radii
    # -------------------------------------------------------------------------
    for filepath in files:
        filename = os.path.basename(filepath)

        with fits.open(filepath) as hdul:
            sci_hdu = hdul["SCI"] if "SCI" in hdul else hdul[1]
            err_hdu = hdul["ERR"] if "ERR" in hdul else hdul[2]

            data = sci_hdu.data.astype(np.float64)
            error = err_hdu.data.astype(np.float64)
            header = sci_hdu.header

            jd_helio = (
                header.get("JD-HELIO")
                or header.get("HJD")
                or header.get("JD")
                or np.nan
            )

        # Validate input coordinates before passing to centroid_sources
        h, w = data.shape
        x_in = np.array([curr_target_xy[0], curr_ref_xy[0]])
        y_in = np.array([curr_target_xy[1], curr_ref_xy[1]])

        # Skip frame if initial coordinates are out of bounds or NaN
        if (
            np.isnan(x_in).any()
            or np.isnan(y_in).any()
            or (x_in < 0).any()
            or (x_in >= w - 1).any()
            or (y_in < 0).any()
            or (y_in >= h - 1).any()
        ):
            print(f"Skipping {filename}: Initial coordinates out of bounds.")
            continue

        # Robust 2D Gaussian Centroiding with Fallback
        try:
            x_cen, y_cen = centroid_sources(
                data,
                x_in,
                y_in,
                box_size=35,
                centroid_func=centroid_2dg,
            )
        except Exception as e:
            print(f"Centroiding failed on {filename}: {e}")
            continue

        # If centroiding returns NaN, fallback to previous good coordinates
        if np.isnan(x_cen[0]) or np.isnan(y_cen[0]):
            x_cen[0], y_cen[0] = curr_target_xy
        else:
            curr_target_xy = (x_cen[0], y_cen[0])

        if np.isnan(x_cen[1]) or np.isnan(y_cen[1]):
            x_cen[1], y_cen[1] = curr_ref_xy
        else:
            curr_ref_xy = (x_cen[1], y_cen[1])

        exact_positions = np.transpose([x_cen, y_cen])
        
        # Robust 2D Gaussian Centroiding
        # approx_coords = np.array([curr_target_xy, curr_ref_xy])
        # x_cen, y_cen = centroid_sources(
        #    data,
        #    approx_coords[:, 0],
        #    approx_coords[:, 1],
        #    box_size=25,
        #    centroid_func=centroid_2dg,  # 2D Gaussian Fit
        # )

        # curr_target_xy = (x_cen[0], y_cen[0])
        # curr_ref_xy = (x_cen[1], y_cen[1])
        # exact_positions = np.transpose([x_cen, y_cen])
        
        # Growth-Matched Aperture Geometry:
        best_r, max_snr, radii, snrs = calculate_optimal_aperture_snr(data, error, exact_positions[0])
        ap_radius=best_r
        r_in = ap_radius + 6.0
        r_out = r_in + 8.0

        # Apertures & Background Annuli
        apertures = CircularAperture(exact_positions, r=ap_radius)
        annuli = CircularAnnulus(exact_positions, r_in=r_in, r_out=r_out)

        # Background Measurement
        annulus_stats = ApertureStats(data, annuli, error=error)
        bkg_per_pixel = annulus_stats.median
        bkg_std_per_pixel = annulus_stats.std

        # Measure Photometry
        phot_table = aperture_photometry(data, apertures, error=error)
        aperture_area = apertures.area
        total_bkg = bkg_per_pixel * aperture_area
        net_flux = phot_table["aperture_sum"] - total_bkg

        # Uncertainty Propagation
        ap_sum_err = phot_table["aperture_sum_err"]
        total_flux_err = np.sqrt(
            ap_sum_err**2 + (np.sqrt(aperture_area) * bkg_std_per_pixel) ** 2
        )

        f_target, f_ref = net_flux[0], net_flux[1]
        err_target, err_ref = total_flux_err[0], total_flux_err[1]

        if f_target <= 0 or f_ref <= 0:
            continue

        flux_ratio = f_target / f_ref
        diff_mag = ref_mag - 2.5 * np.log10(flux_ratio)

        mag_factor = 2.5 / np.log(10)
        diff_mag_err = mag_factor * np.sqrt(
            (err_target / f_target) ** 2 + (err_ref / f_ref) ** 2
        )

        results.append({
            "Filename": filename,
            "JD_HELIO": jd_helio,
            "Target_X": x_cen[0],
            "Target_Y": y_cen[0],
            "Ref_X": x_cen[1],
            "Ref_Y": y_cen[1],
            "Max_SNR": max_snr,
            "Ap_Radius": ap_radius,
            "Target_Flux": f_target,
            "Target_Flux_Err": err_target,
            "Ref_Flux": f_ref,
            "Ref_Flux_Err": err_ref,
            "Diff_Mag": diff_mag,
            "Diff_Mag_Err": diff_mag_err,
        })

    df = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 2. Apply Cloud Outlier Quality Cut & Sigma-Clipping
    # -------------------------------------------------------------------------
    # Calculate baseline reference star flux using 15-frame rolling median
    ref_flux_baseline = df["Ref_Flux"].rolling(15, center=True, min_periods=1).median()
    df["Ref_Flux_Fraction"] = df["Ref_Flux"] / ref_flux_baseline

    # Quality Cut 1: Reject cloud extinction frames (Reference flux < 80% baseline)
    cloud_mask = df["Ref_Flux_Fraction"] >= min_flux_fraction

    # Quality Cut 2: 3-Sigma outlier rejection relative to local median light curve
    diff_mag_baseline = df["Diff_Mag"].rolling(3, center=True, min_periods=1).median()
    residuals = np.abs(df["Diff_Mag"] - diff_mag_baseline)
    sigma_mask = residuals <= (3 * df["Diff_Mag_Err"].median())

    # Combined clean mask
    df_clean = df[cloud_mask & sigma_mask].copy()

    df_clean.to_csv(csv_output, index=False)
    print(
        f"Photometry complete! Kept {len(df_clean)} / {len(df)} frames "
        f"({len(df) - len(df_clean)} cloud/outlier frames removed). Saved to {csv_output}"
    )

    return df_clean


def plot_cleaned_light_curve(df):
    """Plots the cleaned, outlier-free light curve."""
    plt.figure(figsize=(10, 6))

    plt.errorbar(
        df["JD_HELIO"],
        df["Diff_Mag"],
        yerr=df["Diff_Mag_Err"],
        fmt="o",
        color="darkblue",
        ecolor="lightgray",
        capsize=3,
        capthick=1,
        markersize=3.5,
        label="AE UMa (Cleaned)",
    )

    plt.gca().invert_yaxis()  # Brighter stars at top
    plt.xlabel("Heliocentric Julian Date (JD-HELIO)", fontsize=12)
    plt.ylabel("Differential Magnitude (mag)", fontsize=12)
    plt.title(
        "Cleaned Light Curve of AE UMa (Growth-Matched Aperture & 2D Gaussian Centroiding)",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# Run pipeline
df_clean = run_improved_photometry()
plot_cleaned_light_curve(df_clean)
