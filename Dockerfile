# Use the iterativeai/cml image with GPU support as the base image
FROM iterativeai/cml:0-dvc2-base1-gpu

# Set the working directory
WORKDIR /

# Copy your model and other required files into the container
COPY . .

# Install any required packages using pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_md

# Install required R packages
RUN apt-get update -qq && \
    apt-get -y --no-install-recommends install \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        libssh2-1-dev \
        libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install R
RUN echo 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/' > /etc/apt/sources.list.d/cran.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    apt-get update && \
    apt-get install -y --no-install-recommends r-base

RUN R -e "install.packages(c('remotes', 'covr', 'rsconnect', 'RColorBrewer', 'Rcpp', 'RcppTOML', 'base64enc', 'bslib', 'cachem', 'cli', 'colorspace', 'commonmark', 'cpp11', 'crosstalk', 'data.table', 'dplyr', 'evaluate', 'fansi', 'farver', 'fontawesome', 'generics', 'ggplot2', 'glue', 'here', 'highr', 'htmlwidgets', 'httpuv', 'knitr', 'later', 'plotly', 'png', 'promises', 'purrr', 'reticulate', 'shiny', 'shinyWidgets', 'tidyr', 'tibble', 'htmltools'), repos='https://cran.rstudio.com/')"

# Define the entry point for the container
ENTRYPOINT ["python", "run_pipeline.py"]

# Make port 5000 available to the world outside this container
EXPOSE 5000

ARG REDDIT_CLIENT_ID
ARG REDDIT_CLIENT_SECRET

ENV REDDIT_CLIENT_ID=$REDDIT_CLIENT_ID
ENV REDDIT_CLIENT_SECRET=$REDDIT_CLIENT_SECRET

# Define environment variable
ENV MEDICATION_NAME=""

# Run the command to start the app
CMD ["sh", "-c", "python run_pipeline.py '$MEDICATION_NAME'"]
