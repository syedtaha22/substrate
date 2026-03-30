# This is only a test Dockerfile. I am unsure if it's correct
# Still navigating docker stuff...

FROM python:3.12-slim

# Required for some HF internal tools
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as user 1000, not root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy files and ensure the 'user' owns them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# Double check: if app.py is in a folder, keep 'app/app.py'. 
# If it's in the root, change it to 'app.py'
CMD ["chainlit", "run", "app/app.py", "--host", "0.0.0.0", "--port", "7860"]