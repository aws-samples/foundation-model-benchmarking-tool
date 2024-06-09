import re
import ast
import logging
import pandas as pd
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def _pre_process_df(summary_payload_df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Pre-processes the summary payload DataFrame to extract relevant data for plotting.

    Args:
        summary_payload_df (pd.DataFrame): Input DataFrame containing instance data.

    Returns:
        tuple: A tuple containing the processed DataFrame and a list of RPM values.
    """
    logger.info("======================================")
    logger.info(f"Loaded dataframe, shape is: {summary_payload_df.shape}")
    logger.info("======================================")
    
    rpm_values = []
    # Dynamically extract RPM values based on DataFrame columns
    for name in summary_payload_df.columns:
        match = re.search(r'(\d+)_rpm', name)
        if match:
            rpm_values.append(int(match.group(1)))
    
    logger.info("======================================")
    logger.info(f"RPM Values extracted: {rpm_values}")
    logger.info("======================================")
    
    rows = []
    # Extract and reformat data for each RPM value
    for index, row in summary_payload_df.iterrows():
        for value in rpm_values:
            column = f'instance_count_and_cost_{value}_rpm'
            count, cost = ast.literal_eval(row[column])
            rows.append({'instance_type': row['instance_type'], 'instance_count': count, 'RPM': value, 'cost': cost, 'TPM': row['transactions_per_minute']})
    
    plot_df = pd.DataFrame(rows)
    
    logger.info("======================================")
    logger.info(f"Completed pre-processing the data, new dataframe shape: {plot_df.shape}")
    logger.info("======================================")
    
    return plot_df, rpm_values

def plot_best_cost_instance_heatmap(summary_payload_df: pd.DataFrame) -> go.Figure:
    """
    Creates a heatmap plot to visualize the cost of running different instance types at various RPM values.

    Args:
        summary_payload_df (pd.DataFrame): Input DataFrame containing instance data.

    Returns:
        go.Figure: Plotly heatmap figure object.
    """
    

    plot_df, rpm_values = _pre_process_df(summary_payload_df)

    logger.info("======================================")
    logger.info(f"Processed dataframe, shape is: {plot_df.shape}")
    logger.info(f"RPM Values loaded are: {rpm_values}")
    logger.info("======================================")

    heatmap_data = plot_df.pivot(index="RPM", columns="instance_type", values="cost")
    heatmap_data.index = heatmap_data.index.astype(str)

    # Sorting instance types by cost in ascending order for RPM = 1
    sorted_columns = heatmap_data.loc['1'].sort_values().index
    heatmap_data = heatmap_data[sorted_columns]

    heatmap_data_instance_count = plot_df.pivot(index="RPM", columns="instance_type", values="instance_count")
    heatmap_data_instance_count = heatmap_data_instance_count[sorted_columns]

    # Create hover text for the heatmap
    hover_text_combined = []
    for i in range(heatmap_data.shape[0]):
        row = []
        for j in range(heatmap_data.shape[1]):
            rpm = heatmap_data.index[i]
            instance_type = heatmap_data.columns[j]
            cost = heatmap_data.iloc[i, j]
            instance_count = heatmap_data_instance_count.iloc[i, j]
            hover_text = (
                f'Instance Type: {instance_type}<br>'
                f'Instance Count: {instance_count}<br>'
                f'Cost: ${cost}<br>'
                f'RPM: {rpm}'
            )
            row.append(hover_text)
        hover_text_combined.append(row)

    text_arr = heatmap_data.values.copy().astype(str)

    logger.info("======================================")
    logger.info("Adding annotations")
    logger.info("======================================")

    # Annotate the heatmap with additional information
    for i, (cost_row, instance_count_row) in enumerate(zip(heatmap_data.values, heatmap_data_instance_count.values)):
        min_cost_idx = cost_row.argmin()
        min_inst_idx = instance_count_row.argmin()
        normalized_val = (cost_row / cost_row.max()) + (instance_count_row / instance_count_row.max())
        normalized_idx = normalized_val.argmin()

        text_arr[i, min_cost_idx] = f"<b>{cost_row[min_cost_idx]:.2f}<br>(least cost)"
        text_arr[i, min_inst_idx] = f"<b>{cost_row[min_inst_idx]:.2f}<br>(fewest instances)</b>"
        text_arr[i, normalized_idx] = f"<b>{cost_row[normalized_idx]:.2f}<br>(best choice)</b>"

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Dense',
        colorbar=dict(title="Cost"),
        zmin=heatmap_data.values.min(),
        zmax=heatmap_data.values.max(),
        text=text_arr,
        texttemplate="$%{text}",
        hovertext=hover_text_combined,
        hovertemplate='%{hovertext}<extra></extra>'
    ))

    logger.info("======================================")
    logger.info("Updating layout for better visualization")
    logger.info("======================================")

    # Update layout for better visualization
    fig.update_layout(
        title="Costs for different requests/minute by instance types",
        xaxis_title="Instance Type",
        yaxis_title="Requests/minute (RPM)",
        autosize=True,
        width=1500,
        height=700,
    )

    fig.update_traces(textfont_size=10)

    logger.info("======================================")
    logger.info("Heatmap plotting completed")
    logger.info("======================================")

    return fig