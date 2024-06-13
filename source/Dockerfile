FROM svercoutere/mu-python-ml:0.1.0
LABEL maintainer="stefaan.vercoutere@sirus.be"

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any dependencies in the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt