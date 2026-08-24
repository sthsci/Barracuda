FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/platform:/app

WORKDIR /app/platform/api

RUN addgroup --system barracuda && adduser --system --ingroup barracuda barracuda

# The scientific package is installed from the exact source revision being
# deployed.  A release deployment can replace this with a hash-pinned wheel.
COPY platform/api/requirements.txt /tmp/api-requirements.txt
COPY pyproject.toml PACKAGE_README.md LICENSE /app/
COPY src /app/src
COPY webapp/core /app/webapp/core
COPY section_1/src /app/section_1/src
COPY section_2/src /app/section_2/src
COPY section_3/src /app/section_3/src
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt \
    && pip install --no-cache-dir /app

COPY platform/api /app/platform/api
COPY platform/worker /app/platform/worker
RUN mkdir -p /work/cache /work/artifacts \
    && chown -R barracuda:barracuda /app /work

USER barracuda
