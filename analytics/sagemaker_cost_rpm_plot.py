import re
import logging
import pandas as pd
from typing import List
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def _pre_process_df(summary_payload_df: pd.DataFrame) -> tuple[pd.DataFrame, List]:
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
        print(name)
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
            count, cost = tuple(v.lstrip("('").rstrip("')") for v in row[column].split(", "))
            rows.append({'instance_type': row['instance_type'],
                         'instance_count': int(count),
                         'RPM': value,
                         'cost': float(cost),
                         'TPM': row['transactions_per_minute'],
                         'TP_Degree': row['tensor_parallel_degree']})

    plot_df = pd.DataFrame(rows)

    logger.info("======================================")
    logger.info(f"Completed pre-processing the data, new dataframe shape: {plot_df.shape}")
    logger.info("======================================")

    return plot_df, rpm_values


def plot_best_cost_instance_heatmap(summary_payload_df: pd.DataFrame,
                                    output_filename: str,
                                    model_id: str,
                                    subtitle: str,
                                    cost_weight: float,
                                    instance_count_weight: float) -> go.Figure:
    """
    Creates a heatmap plot to visualize the cost of running different instance types at various RPM values.

    Args:
        summary_payload_df (pd.DataFrame): Input DataFrame containing instance data.

    Returns:
        go.Figure: Plotly heatmap figure object.
    """

    logger.info(summary_payload_df.columns)
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

    heatmap_data_instance_count = plot_df.pivot(index="RPM", 
                                                columns="instance_type",
                                                values="instance_count")
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
    for i, (cost_row, instance_count_row) in enumerate(zip(heatmap_data.values,
                                                           heatmap_data_instance_count.values)):
        min_cost_idx = cost_row.argmin()
        min_inst_idx = instance_count_row.argmin()
        normalized_val = (cost_weight * (cost_row / cost_row.max())) +\
                         (instance_count_weight * (instance_count_row / instance_count_row.max()))
        normalized_idx = normalized_val.argmin()

        text_arr[i, min_cost_idx] = f"<b>{cost_row[min_cost_idx]:.2f}<br>(least cost)"
        text_arr[i, min_inst_idx] = f"<b>{cost_row[min_inst_idx]:.2f}<br>(fewest instances)"
        text_arr[i, normalized_idx] = f"<b>{cost_row[normalized_idx]:.2f}<br>(best choice)"

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

    num_rows, num_cols = heatmap_data.shape
    dynamic_font_size = _calculate_dynamic_font_size(num_rows, num_cols)
    fig.update_traces(textfont=dict(size=dynamic_font_size))

    logger.info("======================================")
    logger.info("Updating layout for better visualization")
    logger.info("======================================")

    # Update layout for better visualization
    fig.update_layout(
        title=f"Serving costs for \"{model_id}\" for different requests/minute and instance options<br>{subtitle}<br>Hover your mouse over a cell for additional information.",
        xaxis_title="",
        yaxis_title="Requests/minute",
        autosize=True,
        width=1500,
        height=700, 
        font=dict(size=dynamic_font_size)
    )

    fig.update_traces(textfont_size=dynamic_font_size)
    
    fig.add_annotation(
       showarrow=False,
       xanchor='left',
       xref='paper', 
       x=0, 
       yref='paper',
       y=-0.15,
       text=f"Note: <b><i>best choice</i></b> based on {100*cost_weight}% weightage to cost and {100*instance_count_weight}% to number of instances needed. <b><i>least cost</i></b> and <b><i>fewest instances</i></b> called out only when different from <b><i>best choice</i></b>.",
       font=dict(size=max(dynamic_font_size - 2, 8)))

    # Save the figure as an HTML file
    fig.write_html(output_filename)

    logger.info("======================================")
    logger.info(f"Heatmap plotting completed, saved to {output_filename}")
    logger.info("======================================")

    return fig

def _calculate_dynamic_font_size(num_rows: int, num_cols: int):
    """
    Adjust the dynamic font size of the text in the heatmap based on the number of rows
    and columns
    """
    base_size: int = 14
    scale_factor = min(1000 / max(num_rows, num_cols), 1)
    # Ensure minimum font size of 10
    return max(int(base_size * scale_factor), 10)
