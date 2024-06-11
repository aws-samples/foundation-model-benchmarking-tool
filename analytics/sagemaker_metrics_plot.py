import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_sm_utilization_metrics(endpoint_metrics_df : pd.DataFrame) -> sns.FacetGrid:
    '''
    This function reads a CSV file containing endpoint metrics data, processes the data,
    and generates a FacetGrid line plot to visualize utilization metrics for different 
    concurrency levels and instance types. The function returns the FacetGrid object.

    Parameters:
    endpoint_metrics_df (pd.DataFrame): Dataframe containing the endpoint metrics data.

    Returns:
    FacetGrid: The FacetGrid object containing the plotted data.

    Example Usage:
    ```
    sm_utilization_metrics_plot = plot_sm_utilization_metrics('sagemaker_endpoint_metrics.csv')
    sm_utilization_metrics_plot.savefig('sm_utilization_metrics_plot.png')
    ```
    '''
    # Load the data from CSV file

    logger.info("======================================")
    logger.info(f"loaded dataframe, shape is: {endpoint_metrics_df.shape}")
    logger.info("======================================")
    

    # Ensure Timestamp is in datetime format
    endpoint_metrics_df['Timestamp'] = pd.to_datetime(endpoint_metrics_df['Timestamp'])

    # Melt the DataFrame to long format
    utilization_metrics_melted_data = endpoint_metrics_df.melt(
        id_vars=['Timestamp', 'instance_type', 'concurrency'], 
        value_vars=['CPUUtilization', 'DiskUtilization', 'GPUMemoryUtilization', 'GPUUtilization', 'MemoryUtilization'],
        var_name='Metric', value_name='Value'
    )

    logger.info("======================================")
    logger.info(f"Melted dataframe, new shape is :  {utilization_metrics_melted_data.shape}")
    logger.info("======================================")

    # Define markers for the line plots
    markers = {"CPUUtilization": "o", "DiskUtilization": "s", "GPUMemoryUtilization": "X", "GPUUtilization": "^", "MemoryUtilization": 'v'}

    # Create the FacetGrid for line plot
    utilization_metrics_plot = sns.FacetGrid(
        utilization_metrics_melted_data, col='instance_type', row='concurrency', hue='Metric', 
        palette='muted', height=4, aspect=1.25, sharex=False
    )
    utilization_metrics_plot.map(sns.lineplot, 'Timestamp', 'Value', dashes=False)

    # Update markers
    for ax in utilization_metrics_plot.axes.flat:
        lines = ax.get_lines()
        for line, (metric, marker) in zip(lines, markers.items()):
            line.set_marker(marker)

    # Add legend
    utilization_metrics_plot.add_legend()

    # Create a subtitle
    with sns.plotting_context('paper', font_scale=1.3):
        utilization_metrics_plot.figure.suptitle("Utilization metrics for different concurrency levels and instance types")
        utilization_metrics_plot.set_titles(row_template="concurrency={row_name}", col_template="instance={col_name}", size=8)

    # Bold the concurrency titles
    for ax in utilization_metrics_plot.axes.flat:
        title_text = ax.get_title()
        if "concurrency" in title_text:
            ax.set_title(title_text, fontsize=10, fontweight='bold')

    # Set x and y labels for this chart
    utilization_metrics_plot.set_ylabels("Utilization (%)")
    utilization_metrics_plot.set_xlabels("Timestamp")
    #top below controls the spacing for title of the plot, hspace and wspace control the spacing between the grid
    utilization_metrics_plot.figure.subplots_adjust(top=.93, hspace=0.5, wspace=0.2)

    # Rotate x-axis labels for better readability
    for ax in utilization_metrics_plot.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    logger.info(f'\nPlotting Complete, returning FacetGrid Object\n')
    return utilization_metrics_plot