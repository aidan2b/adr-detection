# Use the iterativeai/cml image with GPU support as the base image
FROM iterativeai/cml:0-dvc2-base1-gpu

# Set the working directory
WORKDIR /

# Copy your model and other required files into the container
COPY . .

# Install any required packages using pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_md

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
