import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize


def inspect_fits(fits_filepath):
    # Open the FITS file safely
    with fits.open(fits_filepath) as hdul:
        # FITS primary data is usually in HDU 0, but some instruments use HDU 1
        primary_hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        header = primary_hdu.header
        data = primary_hdu.data

        # Safely extract header metadata using .get() to handle missing keys
        exptime = header.get("EXPTIME", "N/A")
        filter_name = header.get("FILTER", "N/A")
        date_obs = header.get("DATE-OBS", "N/A")
        object_name = header.get("OBJECT", "Unknown Target")
        readout_noise = header.get("RDNOISE", "N/A")
        julian_date = header.get("JD-HELIO", "N/A")

        print("=== FITS Header Summary ===")
        print(f"Target:   {object_name}")
        print(f"Date-Obs: {date_obs}")
        print(f"JD:       {julian_date}")
        print(f"Exp Time: {exptime} s")
        print(f"Filter:   {filter_name}")
        print(f"Readout Noise: {readout_noise} e-")
        print("===========================")

        # Calculate display limits using ZScaleInterval
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data)

        # Alternative method using ImageNormalize:
        # norm = ImageNormalize(data, interval=ZScaleInterval())

        # Plot the image
        plt.figure(figsize=(10, 8))
        plt.imshow(
            data,
            origin="lower",  # Standard astronomical convention (0,0 at bottom-left)
            cmap="inferno",  # Common choices: 'gray', 'viridis', or 'inferno'
            vmin=vmin,
            vmax=vmax,
        )

        plt.colorbar(label="Counts (ADU)")
        plt.title(
            f"{object_name} | Filter: {filter_name} | Exp: {exptime}s",
            fontsize=12,
        )
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.tight_layout()
        plt.show()


# Example Usage:
inspect_fits('2026/sci/AE_UMa-0001.fit')
