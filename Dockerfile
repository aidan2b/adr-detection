FROM nvcr.io/nvidia/pytorch:22.08-py3

# Set the working directory
WORKDIR /

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any required packages using pip
RUN pip install --no-cache-dir -r requirements.txt

# Download the en_core_web_md spaCy model
RUN python -m spacy download en_core_web_md

# Copy your model and other required files into the container
COPY . .

# Set the entrypoint
ENTRYPOINT [ "/bin/bash" ]
