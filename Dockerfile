# Use the Amazon Linux 2023 image
FROM public.ecr.aws/amazonlinux/amazonlinux:latest

ENV PYTHONUNBUFFERED=1

# Install necessary packages
RUN yum install -y --skip-broken wget jq aws-cli bzip2 ca-certificates curl git gcc gcc-c++ make openssl-devel libffi-devel zlib-devel && \
    yum clean all
    
# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh && \
    /opt/miniconda/bin/conda clean --all --yes

# Set conda environment variables
ENV PATH="/opt/miniconda/bin:$PATH"
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create and activate the conda environment
RUN conda create --name fmbench_python311 -y python=3.11 ipykernel && \
    echo "source activate fmbench_python311" > ~/.bashrc

# Install fmbench
RUN /opt/miniconda/bin/conda run -n fmbench_python311 pip install -U fmbench



# Set working directory
WORKDIR /app

# Copy the application files
COPY . /app/

# Set the entrypoint to run the fmbench command
ENTRYPOINT ["conda", "run", "-n", "fmbench_python311", "sh", "-c"]
CMD ["fmbench", "--help"]