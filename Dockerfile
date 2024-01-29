ARG BASE_CONTAINER=jupyter/minimal-notebook
FROM $BASE_CONTAINER
USER root
RUN pip install pandas numpy matplotlib plotly
# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID