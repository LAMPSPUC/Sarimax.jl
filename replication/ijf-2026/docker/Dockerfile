# syntax=docker/dockerfile:1
#
# Replication image for "SARIMAX.jl: Open-Source Optimization-Based SARIMAX Modeling in Julia"
# (International Journal of Forecasting). Build from the repository root:
#
#   ./reproduce.sh build
#
# which is equivalent to
#
#   docker build --platform linux/amd64 -f replication/ijf-2026/docker/Dockerfile \
#     --build-arg SOURCE_FINGERPRINT="$(bash replication/ijf-2026/scripts/fingerprint.sh)" \
#     --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t sarimax-ijf-replication .
#
# Every component is pinned: the base image by digest, Julia and Python by release tarball
# and SHA-256, the Julia packages by Manifest.toml, the Python packages by wheel hashes, and
# the R packages by a dated CRAN snapshot that the build then checks against renv.lock.
# The Sarimax.jl source is the one in the build context (this checkout), never a download.

# R 4.4.1 on Ubuntu 22.04, with CRAN frozen at the 2024-10-30 Posit Package Manager snapshot.
# That snapshot carries forecast 8.23.0, the version the archived auto.arima M4 baseline used.
FROM rocker/r-ver:4.4.1@sha256:78cb94ce2db23aaaf7b546450fcf70b5a3f2ace5a9b5fa1f87217da329211312

ARG TARGETARCH

ARG JULIA_VERSION=1.10.12
ARG JULIA_SHA256_AMD64=b0c50dfe349d76ffc8580d24e0c0be623988a6d91ed57ad8b73c0137fd27bc0d
ARG JULIA_SHA256_ARM64=cc257b21967049642bfe17de1aff7dae17477eba4f423f98cb400c002080e5a2
ARG PYTHON_BUILD=cpython-3.12.14+20260901
ARG PYTHON_RELEASE=20260901
ARG PYTHON_SHA256_AMD64=936c246dfdbbfa7cb22dd01814a21f582a892689fae96b06071a5e433baffa22
ARG PYTHON_SHA256_ARM64=b61b856c3e1a4fc65b8f6e6b0495ef975dd0924f90c59f3ea61b38a079173b84
ARG CRAN_SNAPSHOT=https://p3m.dev/cran/__linux__/jammy/2024-10-30

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# --- Python 3.12 (python-build-standalone), packages from hashed wheels ---------------------
RUN case "${TARGETARCH:-amd64}" in \
        amd64) arch=x86_64;  sha="$PYTHON_SHA256_AMD64" ;; \
        arm64) arch=aarch64; sha="$PYTHON_SHA256_ARM64" ;; \
        *) echo "unsupported architecture ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && f="${PYTHON_BUILD}-${arch}-unknown-linux-gnu-install_only.tar.gz" \
    && curl -fsSL -o /tmp/python.tar.gz \
        "https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/${f//+/%2B}" \
    && echo "${sha}  /tmp/python.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/python && tar -xzf /tmp/python.tar.gz -C /opt/python --strip-components=1 \
    && rm /tmp/python.tar.gz \
    && /opt/python/bin/python3 -m venv /opt/venv
COPY replication/ijf-2026/environments/python/requirements.lock /tmp/requirements.lock
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements.lock \
    && /opt/venv/bin/pip check

# --- R packages from the frozen CRAN snapshot, verified against renv.lock --------------------
COPY replication/ijf-2026/environments/r/renv.lock /opt/ijf-env/renv.lock
COPY replication/ijf-2026/scripts/checks/r_env.R /opt/ijf-env/r_env.R
# The snapshot serves Linux binaries for amd64 only; on arm64 the packages are compiled, and
# the R package curl needs the libcurl headers to build.
RUN if [ "${TARGETARCH:-amd64}" = "arm64" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends libcurl4-openssl-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi \
    && Rscript -e "install.packages(c('forecast', 'jsonlite'), repos = '${CRAN_SNAPSHOT}', Ncpus = 4L)" \
    && Rscript --vanilla /opt/ijf-env/r_env.R /opt/ijf-env/renv.lock

# --- Julia --------------------------------------------------------------------------------
RUN case "${TARGETARCH:-amd64}" in \
        amd64) dir=x64;     arch=x86_64;  sha="$JULIA_SHA256_AMD64" ;; \
        arm64) dir=aarch64; arch=aarch64; sha="$JULIA_SHA256_ARM64" ;; \
        *) echo "unsupported architecture ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && minor="${JULIA_VERSION%.*}" \
    && curl -fsSL -o /tmp/julia.tar.gz \
        "https://julialang-s3.julialang.org/bin/linux/${dir}/${minor}/julia-${JULIA_VERSION}-linux-${arch}.tar.gz" \
    && echo "${sha}  /tmp/julia.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/julia && tar -xzf /tmp/julia.tar.gz -C /opt/julia --strip-components=1 \
    && rm /tmp/julia.tar.gz

ENV PATH=/opt/venv/bin:/opt/julia/bin:$PATH \
    JULIA_DEPOT_PATH=/opt/julia-depot \
    JULIA_PKG_PRECOMPILE_AUTO=0 \
    OPENBLAS_NUM_THREADS=1 \
    LANG=C.UTF-8

# The Julia environment needs the package source (Sarimax is developed from this checkout),
# so the source goes in before the environment is instantiated and precompiled.
COPY Project.toml /opt/sarimax/Project.toml
COPY src /opt/sarimax/src
COPY replication/ijf-2026/environments/julia /opt/sarimax/replication/ijf-2026/environments/julia
COPY replication/ijf-2026/scripts/checks/scip_artifacts.jl /opt/ijf-env/scip_artifacts.jl
# Precompiled code is built for the CPU targets of the official Julia binaries rather than for
# the build machine, so that it stays valid on the reviewer's CPU (otherwise every container
# start would recompile into its throw-away depot).
# SCIP's binaries (SCIP 8.0.0 and SoPlex 6.0, ZIB Academic License) are used to precompile and
# test the environment and then removed in this same layer, so that no layer of the image
# contains them; `./reproduce.sh fetch scip` installs them on the user's machine.
RUN case "${TARGETARCH:-amd64}" in \
        amd64) export JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)" ;; \
        arm64) export JULIA_CPU_TARGET="generic;cortex-a57;thunderx2t99;carmel,clone_all;apple-m1,base(3);neoverse-512tvb,base(3)" ;; \
    esac \
    && julia --project=/opt/sarimax/replication/ijf-2026/environments/julia -e ' \
        using Pkg; Pkg.instantiate(); Pkg.precompile(); \
        using Sarimax, JuMP, Ipopt, SCIP, HiGHS, CSV, DataFrames, JSON, TimeSeries; \
        println("SCIP ", JuMP.MOI.get(SCIP.Optimizer(), JuMP.MOI.SolverVersion()), " loads")' \
    && julia --project=/opt/sarimax/replication/ijf-2026/environments/julia /opt/ijf-env/scip_artifacts.jl \
        record /opt/ijf-env/scip-artifacts.toml \
    && julia --project=/opt/sarimax/replication/ijf-2026/environments/julia /opt/ijf-env/scip_artifacts.jl strip \
    && chmod -R a+rX /opt/julia-depot /opt/ijf-env

# --- The rest of the checkout ----------------------------------------------------------------
# The identity arguments are declared here, not at the top: a build argument invalidates the
# cache of every RUN that follows it, and the fingerprint changes with any replication file.
# Declared here, editing a script rebuilds only this last layer, not R, Python and Julia.
# Sarimax.jl loads SCIP (the default solver of its lower-bounding step), so precompiling and
# checking it here reinstalls the SCIP artifacts; they are removed again before this layer ends.
ARG SOURCE_FINGERPRINT
ARG SOURCE_REVISION=unknown
COPY . /opt/sarimax
RUN test -n "${SOURCE_FINGERPRINT}" \
        || { echo "SOURCE_FINGERPRINT build argument is required (use ./reproduce.sh build)" >&2; exit 1; } \
    && actual="$(bash /opt/sarimax/replication/ijf-2026/scripts/fingerprint.sh /opt/sarimax)" \
    && if [ "$actual" != "$SOURCE_FINGERPRINT" ]; then \
           echo "build context fingerprint $actual does not match SOURCE_FINGERPRINT $SOURCE_FINGERPRINT" >&2; \
           bash /opt/sarimax/replication/ijf-2026/scripts/fingerprint.sh --list /opt/sarimax >&2; exit 1; \
       fi \
    && case "${TARGETARCH:-amd64}" in \
        amd64) export JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)" ;; \
        arm64) export JULIA_CPU_TARGET="generic;cortex-a57;thunderx2t99;carmel,clone_all;apple-m1,base(3);neoverse-512tvb,base(3)" ;; \
       esac \
    && julia --project=/opt/sarimax/replication/ijf-2026/environments/julia -e 'using Pkg; Pkg.precompile()' \
    && julia --project=/opt/sarimax/replication/ijf-2026/environments/julia \
        /opt/sarimax/replication/ijf-2026/scripts/checks/julia_env.jl \
    && julia --project=/opt/sarimax/replication/ijf-2026/environments/julia \
        /opt/sarimax/replication/ijf-2026/scripts/checks/scip_artifacts.jl strip \
    && bash /opt/sarimax/replication/ijf-2026/docker/collect-licenses.sh \
    && chmod -R a+rX /opt/julia-depot /opt/licenses \
    && (id -u ijf >/dev/null 2>&1 || useradd --create-home ijf) \
    && mkdir -p /ijf/output /ijf/artifacts /ijf/data /ijf/julia \
    && chown ijf /ijf/output /ijf/artifacts /ijf/data \
    && chmod 1777 /ijf/julia

# A writable depot goes first so that the read-only image depot still serves the precompiled
# packages when the container runs as an arbitrary host user. /ijf/julia is the depot that
# `fetch scip` installs SCIP's binaries into (reproduce.sh mounts a Docker volume there, so
# they persist between runs); it is world-writable so that a new volume is too.
ENV JULIA_DEPOT_PATH=/tmp/julia-depot:/ijf/julia:/opt/julia-depot \
    IJF_SCIP_DEPOT=/ijf/julia \
    IJF_SCIP_RECORD=/opt/ijf-env/scip-artifacts.toml \
    JULIA_PKG_OFFLINE=true \
    HOME=/tmp \
    IJF_RUNTIME=docker \
    IJF_REPO_ROOT=/opt/sarimax \
    IJF_IMAGE_FINGERPRINT=${SOURCE_FINGERPRINT} \
    IJF_SOURCE_REVISION=${SOURCE_REVISION} \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1

LABEL org.opencontainers.image.title="sarimax-ijf-replication" \
      org.opencontainers.image.description="Replication environment for the SARIMAX.jl IJF paper: Julia 1.10.12 (JuMP, Ipopt; SCIP 8.0.0 fetched on first use, not included), R 4.4.1 (forecast 8.23.0), Python 3.12 (statsmodels 0.14.6)" \
      org.opencontainers.image.source="https://github.com/LAMPSPUC/Sarimax.jl" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.licenses="MIT" \
      io.github.lampspuc.sarimax.ijf.fingerprint="${SOURCE_FINGERPRINT}"

USER ijf
WORKDIR /opt/sarimax
ENTRYPOINT ["python3", "/opt/sarimax/replication/ijf-2026/scripts/ijf.py"]
CMD ["quick"]
