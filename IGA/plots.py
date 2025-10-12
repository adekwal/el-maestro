import matplotlib.pyplot as plt
import numpy as np
import string

def plot_charts(charts, titles=None, suptitle=None, cbar_label=None):
    n_charts = len(charts)
    n_cols = min(2, n_charts) # max 2 plots per row
    n_rows = int(np.ceil(n_charts / 2)) # calculate the number of rows

    # Calculate global vmin and vmax for consistent scaling
    vmin = min(chart.min() for chart in charts)
    vmax = max(chart.max() for chart in charts)

    # Create subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig.subplots_adjust(right=0.85)  # Leave space for colorbar

    # Ensure axs is always a 2D array for indexing, even with a single row
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)

    # Generate letters for annotations
    annotations = string.ascii_lowercase

    # Plot each chart
    for idx, chart in enumerate(charts):
        row, col = divmod(idx, n_cols)
        im = axs[row][col].imshow(chart, vmin=vmin, vmax=vmax)

        # Add title if available
        axs[row][col].set_title(
            titles[idx] if titles and idx < len(titles) else f"Chart {idx + 1}",
            fontsize=16
        )

        # Add annotation
        axs[row][col].text(
            0.02, 0.98, f"({annotations[idx]})",
            transform=axs[row][col].transAxes,
            fontsize=16,
            color='white',
            ha='left',
            va='top',
            bbox=dict(facecolor='black', alpha=0, edgecolor='none')
        )

    # Hide unused subplots if any
    for idx in range(n_charts, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axs[row][col].axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label(cbar_label, fontsize=12)

    # Add overall title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
        plt.savefig(suptitle)

    plt.show()