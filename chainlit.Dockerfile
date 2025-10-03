FROM python:3.10-slim-bookworm

RUN apt-get update && \
    apt-get install -y git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python - && \
    mv $HOME/.local/bin/poetry /usr/local/bin/poetry && \
    poetry config virtualenvs.create false

WORKDIR /app

COPY app ./

WORKDIR /app/chainlit
#RUN sed -i 's/python-multipart = "\^0.0.9"/python-multipart = "\^0.0.19"/g' backend/pyproject.toml

WORKDIR /app/chainlit/backend
RUN poetry build

RUN pip install dist/*.whl

WORKDIR /app
# COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/chainlit/frontend/dist /frontend



CMD ["chainlit", "run", "app.py", "--headless", "--no-cache"]
