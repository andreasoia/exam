import glob
import numpy as np
from astropy.io import fits


def get_exptime(filepath):
    """Extracts exposure time from FITS header, looking for EXPTIME or EXPOSURE keys."""
    with fits.open(filepath) as hdul:
        header = hdul[0].header if hdul[0].data is not None else hdul[1].header
        # Look for common exposure time keywords
        if "EXPTIME" in header:
            return float(header["EXPTIME"])
        elif "EXPOSURE" in header:
            return float(header["EXPOSURE"])
        else:
            raise KeyError(
                f"Neither EXPTIME nor EXPOSURE found in header of {filepath}"
            )


def stack_frames(file_list):
    """Opens a list of FITS files and stacks them into a 3D numpy array along axis 0."""
    data_list = []
    for filepath in file_list:
        with fits.open(filepath) as hdul:
            data = hdul[0].data if hdul[0].data is not None else hdul[1].data
            data_list.append(data.astype(np.float64))
    return np.array(data_list)


def create_master_calibration(data_dir="2026/calib"):
    # --- 1. MASTER BIAS ---
    bias_files = sorted(glob.glob(f"{data_dir}/*bias.fit"))
    if not bias_files:
        raise FileNotFoundError("No bias files found matching the pattern.")

    print(f"Combining {len(bias_files)} bias frames...")
    bias_stack = stack_frames(bias_files)
    master_bias = np.median(bias_stack, axis=0)

    fits.writeto(
        "2026/calib/master_bias.fits", master_bias.astype(np.float32), overwrite=True
    )
    print("Master Bias created successfully.")

    # --- 2. MASTER DARK (Normalized by Exposure Time) ---
    dark_files = sorted(glob.glob(f"{data_dir}/dark*.fit"))
    if not dark_files:
        raise FileNotFoundError("No dark files found matching the pattern.")

    # Fetch exposure time from the first dark frame
    dark_exptime = get_exptime(dark_files[0])
    print(
        f"Combining {len(dark_files)} dark frames (Exposure Time: {dark_exptime}s)..."
    )

    dark_stack = stack_frames(dark_files)
    bias_subtracted_darks = dark_stack - master_bias

    # Divide by exposure time to get dark current in counts/second
    master_dark_per_sec = (
        np.median(bias_subtracted_darks, axis=0) / dark_exptime
    )

    # Save Normalized Master Dark (ADU/sec)
    fits.writeto(
        "2026/calib/master_dark.fits",
        master_dark_per_sec.astype(np.float32),
        overwrite=True,
    )
    print("Normalized Master Dark (ADU/s) created successfully.")

    # --- 3. MASTER FLAT ---
    flat_files = sorted(glob.glob(f"{data_dir}/*flat.fit"))
    if not flat_files:
        raise FileNotFoundError("No flat files found matching the pattern.")
        
    print(f"Combining {len(flat_files)} flat frames...")
    flat_stack = stack_frames(flat_files)

    # Subtract bias and dark
    calibrated_flats = flat_stack - master_bias - np.median(bias_subtracted_darks, axis=0)

    combined_flat = np.median(calibrated_flats, axis=0)

    # Normalize Master Flat response around 1.0
    flat_median = np.median(combined_flat)
    master_flat = combined_flat / flat_median

    # Save Master Flat
    fits.writeto(
        "2026/calib/master_flat.fits", master_flat.astype(np.float32), overwrite=True
    )
    print("Master Flat created successfully.")

    return master_bias, master_dark_per_sec, master_flat


# Execution
master_bias, master_dark, master_flat = create_master_calibration(
    "2026/calib"
)
