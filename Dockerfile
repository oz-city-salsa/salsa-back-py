FROM gineshidalgo99/openpose:latest

# Install Python & dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    pip3 install fastapi uvicorn numpy opencv-python

# Copy your FastAPI app
COPY app /app
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
