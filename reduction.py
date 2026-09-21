import glob
import os
import re
import numpy as np
from astropy.io import fits


def reduce_single_frame(
    science_path, master_bias, master_dark, master_flat
):
    """Calibrates a single science frame and computes its electron error map."""
    with fits.open(science_path) as hdul:
        sci_hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        raw_data = sci_hdu.data.astype(np.float64)
        sci_hdr = sci_hdu.header.copy()

    # Extract header values safely
    sci_exptime = float(sci_hdr.get("EXPTIME", 1.0))

    # Search common header aliases for Gain and Readout Noise
    gain = float(
        sci_hdr.get("EGAIN")
        or sci_hdr.get("GAIN")
        or sci_hdr.get("GAINVAL")
        or 1.0
    )

    rdnoise = float(
        sci_hdr.get("RDNOISE")
        or sci_hdr.get("READNOIS")
        or sci_hdr.get("RON")
        or 0.0
    )

    # 1. Calibration math
    scaled_master_dark = master_dark * sci_exptime
    calibrated_adu = (raw_data - master_bias - scaled_master_dark) / master_flat
    calibrated_e = calibrated_adu * gain

    # 2. Error Map Calculation (in e-)
    raw_signal_e = np.maximum(
        (raw_data - master_bias - scaled_master_dark) * gain, 0.0
    )
    poisson_var_e = raw_signal_e
    readout_var_e = rdnoise**2
    error_map_e = np.sqrt(poisson_var_e + readout_var_e) / master_flat

    # 3. Build MEF Structure
    primary_hdr = fits.Header()
    primary_hdr["OBJECT"] = sci_hdr.get("OBJECT", "AE UMa")
    primary_hdr["COMMENT"] = "Calibrated Multi-Extension FITS (MEF) file"
    primary_hdu = fits.PrimaryHDU(header=primary_hdr)

    # Extension 1: Science Image
    sci_hdr["BUNIT"] = ("electron", "Physical units of data")
    sci_hdr["CALIB"] = ("DONE", "Bias, Dark, and Flat calibrated")
    sci_hdr["GAIN_APL"] = (gain, "Gain applied (e-/ADU)")
    science_hdu = fits.ImageHDU(
        data=calibrated_e.astype(np.float32), header=sci_hdr, name="SCI"
    )

    # Extension 2: Error Map
    err_hdr = fits.Header()
    err_hdr["BUNIT"] = ("electron", "Physical units of uncertainty")
    err_hdr["EXTNAME"] = ("ERR", "Error Extension")
    err_hdr["RDNOISE"] = (rdnoise, "Readout noise used in e-")
    err_hdr["EGAIN"] = (gain, "Gain used in e-/ADU")
    error_hdu = fits.ImageHDU(
        data=error_map_e.astype(np.float32), header=err_hdr, name="ERR"
    )

    return fits.HDUList([primary_hdu, science_hdu, error_hdu])


def batch_reduce_ae_uma(
    input_dir="2026/sci",
    output_dir="2026/red",
    master_bias_path="2026/calib/master_bias.fits",
    master_dark_path="2026/calib/master_dark.fits",
    master_flat_path="2026/calib/master_flat.fits",
):
    """Discovers AE_UMa-NNNN.fit files in numerical order and processes them."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Calibration Master Frames once into memory
    print("Loading master calibration frames...")
    with fits.open(master_bias_path) as h:
        master_bias = h[0].data.astype(np.float64)
    with fits.open(master_dark_path) as h:
        master_dark = h[0].data.astype(np.float64)
    with fits.open(master_flat_path) as h:
        master_flat = h[0].data.astype(np.float64)

    # 2. Match AE_UMa_NNNN.fit files using regex for strict numerical sorting
    pattern = re.compile(r"AE_UMa-(\d+)\.fit$", re.IGNORECASE)
    raw_files = glob.glob(os.path.join(input_dir, "AE_UMa-*.fit"))

    # Extract frame numbers and sort numerically (so 2 comes before 10)
    matched_files = []
    for fpath in raw_files:
        match = pattern.search(fpath)
        if match:
            frame_num = int(match.group(1))
            matched_files.append((frame_num, fpath))

    matched_files.sort(key=lambda x: x[0])

    if not matched_files:
        print(f"No files matching 'AE_UMa-NNNN.fit' found in {input_dir}")
        return

    print(f"Found {len(matched_files)} science frames to process.\n")

    # 3. Process each frame
    for frame_num, filepath in matched_files:
        filename = os.path.basename(filepath)
        out_filename = f"AE_UMa-{frame_num:04d}_calibrated_mef.fits"
        out_path = os.path.join(output_dir, out_filename)

        print(f"Processing frame {frame_num:04d}: {filename} -> {out_filename}")

        try:
            mef_hdul = reduce_single_frame(
                filepath, master_bias, master_dark, master_flat
            )
            mef_hdul.writeto(out_path, overwrite=True)
        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")

    print("\nBatch calibration complete!")


# Example Execution:
batch_reduce_ae_uma(input_dir="2026/sci", output_dir="2026/red")
