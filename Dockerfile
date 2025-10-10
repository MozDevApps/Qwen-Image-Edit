FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Install system dependencies
RUN apt update && apt install -y python3 python3-pip git

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application code
COPY . /app
WORKDIR /app

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
