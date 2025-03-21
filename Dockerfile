FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# # Set the working directory in the container
WORKDIR /app

# # Copy the current directory contents into the container at /app
COPY . /app

RUN uv pip install --system --no-cache -r requirements.txt

# # Run the application
# # CMD ["/opt/venv/bin/python", "trader.py"]

# # Start a shell
CMD ["/bin/sh"] 
