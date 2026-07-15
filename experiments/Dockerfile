# Reproducibility image for the SARIMAX.jl experiment artifact (Tier 2).
#
# Build context = the experiments/ directory:
#     docker build -f Dockerfile -t sarimax-artifact .
# Run (regenerates results/ and tables/ into ./out):
#     docker run --rm -v "$PWD/out:/artifact/results" sarimax-artifact
#
# Caveat: this image is linux/amd64; the reported numbers were produced on macOS/arm64.
# The image freezes the *environment* (exact tool and package versions); last-digit numerical
# values and solver timing/termination may differ across platform/BLAS. The paper's claims are
# bounded and do not depend on exact digits.
#
# Requires the tested package commit to be reachable on the public repository (see julia_setup.jl).

# Exact R (rocker pins R itself); Ubuntu base.
FROM rocker/r-ver:4.6.1

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git build-essential xz-utils && \
    rm -rf /var/lib/apt/lists/*

# --- Julia 1.11.9 (pinned) ---
RUN curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.9-linux-x86_64.tar.gz \
      | tar -xz -C /opt && ln -s /opt/julia-1.11.9/bin/julia /usr/local/bin/julia

# --- uv (to pin Python 3.9.6 exactly) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /artifact
COPY . /artifact

# --- Python 3.9.6 env + pinned packages ---
RUN uv venv --python 3.9.6 .venv-benchmarks && \
    . .venv-benchmarks/bin/activate && \
    uv pip install -r requirements.txt

# --- R packages (dated PPM snapshot -> pinned versions) ---
RUN Rscript scripts/setup/install_r.R

# --- Julia env: Sarimax pinned to the tested commit + script deps ---
RUN julia --project=env scripts/setup/julia_setup.jl

# Default: run the full battery and rebuild the tables.
ENTRYPOINT ["bash", "reproduce.sh", "run"]
