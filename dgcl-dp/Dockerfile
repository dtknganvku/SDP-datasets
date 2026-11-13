FROM python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5
ENV PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 PYTHONPATH=/app/src CUBLAS_WORKSPACE_CONFIG=:4096:8
WORKDIR /app
COPY requirements.lock ./
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.lock
COPY src ./src
COPY manifests ./manifests
ENTRYPOINT ["python", "-m", "dgcl_dp.reproduce"]
CMD ["experiment=demo"]
