FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libegl1 libgl1 libgles2 libglib2.0-0 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app

COPY requirements.txt ./
RUN --mount=type=secret,id=pip_ca \
    if [ -s /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=65534:65534 handscribe_xr/ handscribe_xr/
COPY --chown=65534:65534 BiosHackathon/src/egovision/ BiosHackathon/src/egovision/
COPY --chown=65534:65534 web/ web/
COPY --chown=65534:65534 --chmod=755 run.sh serve.sh ./

USER 65534:65534
EXPOSE 8000
CMD ["./serve.sh", "--host", "0.0.0.0"]
