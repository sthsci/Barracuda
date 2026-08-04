FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    HDF5_USE_FILE_LOCKING=FALSE \
    PYTENSOR_FLAGS=optimizer_excluding=fusion,base_compiledir=/tmp/pytensor

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home appuser

COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser streamlit_app.py ./
COPY --chown=appuser:appuser webapp ./webapp
COPY --chown=appuser:appuser section_1/src ./section_1/src
COPY --chown=appuser:appuser section_2/src ./section_2/src

USER appuser

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
