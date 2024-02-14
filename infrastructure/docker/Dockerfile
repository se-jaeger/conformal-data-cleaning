FROM python:3.9.13

WORKDIR /conformal-data-cleaning

# cache depedencies
COPY pyproject.toml poetry.lock .
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root --no-directory --only main --no-interaction --no-ansi

# install source code
COPY . .
RUN poetry install --only main --no-interaction --no-ansi
