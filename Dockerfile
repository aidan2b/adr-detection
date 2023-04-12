FROM nvcr.io/nvidia/pytorch:22.08-py3

# Set the working directory
WORKDIR /

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any required packages using pip
RUN pip install --no-cache-dir -r requirements.txt

# Download the en_core_web_md spaCy model
RUN python -m spacy download en_core_web_md

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install necessary packages and R
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        r-base \
        r-base-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev

# Install required R packages
RUN Rscript -e 'install.packages(c("rsconnect", "shiny", "plotly", "tidyr", "dplyr", "jsonlite", "purrr", "httr", "readr", "reticulate"), repos="https://cran.rstudio.com/")'

# Copy your model and other required files into the container
COPY . .

# Set the entrypoint
ENTRYPOINT [ "/bin/bash" ]
