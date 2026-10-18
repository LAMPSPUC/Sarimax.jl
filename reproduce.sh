#!/usr/bin/env bash
# SARIMAX.jl — reproduction entry point for the IJF paper
# "SARIMAX.jl: Open-Source Optimization-Based SARIMAX Modeling in Julia".
#
#   ./reproduce.sh                      check + smoke + tables (the default quick reproduction)
#   ./reproduce.sh check                verify environment, image identity, inputs and checksums
#   ./reproduce.sh smoke                small real computations (Julia/Ipopt/SCIP, R, Python)
#   ./reproduce.sh tables               regenerate tables and figures from archived results
#   ./reproduce.sh full --experiment ID [--jobs N] [--resume] [--limit N | --sample N]
#   ./reproduce.sh full --all [--jobs N] [--resume]
#   ./reproduce.sh list                 experiments and their replication readiness
#   ./reproduce.sh fetch [artifacts|m4|pjme|all]   download (or --from FILE) and verify inputs
#   ./reproduce.sh build                build the container image from this checkout
#   ./reproduce.sh help
#
# Everything scientific runs inside a container image (Julia, R, Python and the solvers are
# preinstalled there); the host only needs bash and Docker. See replication/ijf-2026/README.md.
set -Eeuo pipefail

readonly EXIT_OK=0
# shellcheck disable=SC2034  # returned by the in-container orchestrator; listed for reference
readonly EXIT_FAIL=1        # a verification, smoke test, table comparison or run failed
readonly EXIT_USAGE=2       # bad command line
readonly EXIT_ENV=3         # Docker missing, daemon unreachable, image unavailable or not this checkout
readonly EXIT_INPUT=4       # required input missing or corrupted

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT
readonly REPL_REL="replication/ijf-2026"
readonly REPL_DIR="$REPO_ROOT/$REPL_REL"
# The container engine; IJF_DOCKER_BIN lets CI simulate a host without Docker.
readonly DOCKER="${IJF_DOCKER_BIN:-docker}"

# ---------------------------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    c_bold=$'\e[1m'; c_red=$'\e[31m'; c_green=$'\e[32m'; c_yellow=$'\e[33m'; c_off=$'\e[0m'
else
    c_bold=""; c_red=""; c_green=""; c_yellow=""; c_off=""
fi
info() { printf '%s==>%s %s\n' "$c_bold" "$c_off" "$*" >&2; }   # progress goes to stderr
warn() { printf '%swarning:%s %s\n' "$c_yellow" "$c_off" "$*" >&2; }
err()  { printf '%serror:%s %s\n' "$c_red" "$c_off" "$*" >&2; }
die()  { local code="$1"; shift; err "$*"; exit "$code"; }

usage() {
    sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options (all commands):
  --image REF        use this image reference instead of the one recorded in image.lock
  --build            build the image locally from this checkout instead of pulling it
  --no-build         never build; fail if the recorded image cannot be obtained
  --platform P       image platform (default linux/amd64; linux/arm64 is experimental)
  --artifacts DIR    archived artifact bundle directory (default replication/ijf-2026/artifacts)
  --data-dir DIR     third-party input data, m4/ and pjme/ (default replication/ijf-2026/data/external)
  --output DIR       where generated outputs are written (default replication/ijf-2026/output)
  --offline          never download anything (missing inputs are reported, not fetched)
  --from FILE        `fetch artifacts` (and the default run) take the bundle from this local
                     tarball instead of downloading it; it is verified the same way
  --allow-unverified-image
                     run even if the image was not built from this checkout (results are
                     labelled UNVERIFIED in every report)
  --native           developer/CI mode: run with host Julia/R/Python instead of Docker
                     (not a supported reproduction path; reports are labelled NATIVE)

Options for `full`:
  --experiment ID    experiment identifier from experiment_manifest.csv (repeatable)
  --all              every experiment marked runnable in the manifest
  --jobs N           parallel worker processes (default 1)
  --resume           continue an interrupted run from its checkpoints
  --limit N          run only the first N series per task (execution check, not science)
  --sample N         run N series drawn at random per frequency, the same for every arm
                     (execution check on a spread of series, not science); --seed S changes the draw
  --timeout S        per-series wall-clock limit in seconds (default: the experiment's own)
  --run-id NAME      name of the run directory (default: a timestamp)
  --frequency F      only this M4 frequency (repeatable)
  --arm A            only this arm of the experiment (repeatable)

Exit codes: 0 success, 1 failure, 2 usage, 3 environment, 4 missing/corrupted input.
EOF
}

# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------
command_name="quick"
if [[ $# -gt 0 && "$1" != -* ]]; then
    command_name="$1"
    shift
fi

image_ref=""
force_build=0
no_build=0
platform="linux/amd64"
artifacts_dir="$REPL_DIR/artifacts"
artifacts_explicit=0
from_file=""
data_dir="$REPL_DIR/data/external"
output_dir="$REPL_DIR/output"
offline=0
allow_unverified=0
native=0
passthrough=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) image_ref="${2:?--image needs a value}"; shift 2 ;;
        --build) force_build=1; shift ;;
        --no-build) no_build=1; shift ;;
        --platform) platform="${2:?--platform needs a value}"; shift 2 ;;
        --artifacts) artifacts_dir="${2:?--artifacts needs a value}"; artifacts_explicit=1; shift 2 ;;
        --data-dir) data_dir="${2:?--data-dir needs a value}"; shift 2 ;;
        --output) output_dir="${2:?--output needs a value}"; shift 2 ;;
        --offline) offline=1; shift ;;
        --allow-unverified-image) allow_unverified=1; shift ;;
        --native) native=1; shift ;;
        -h|--help) usage; exit "$EXIT_OK" ;;
        --from)
            [[ $# -ge 2 && -f "$2" ]] || die "$EXIT_INPUT" "--from needs an existing file (the artifact bundle tarball)"
            from_file="$2"; shift 2 ;;
        --experiment|--jobs|--limit|--sample|--seed|--timeout|--run-id|--frequency|--arm|--group)
            [[ $# -ge 2 ]] || die "$EXIT_USAGE" "$1 needs a value"
            passthrough+=("$1" "$2"); shift 2 ;;
        --all|--resume|--keep-going)
            passthrough+=("$1"); shift ;;
        artifacts|m4|pjme|all)
            passthrough+=("$1"); shift ;;
        *) usage >&2; die "$EXIT_USAGE" "unknown argument: $1" ;;
    esac
done

case "$command_name" in
    quick|check|smoke|tables|full|list|fetch|build|shell|help) ;;
    *) usage >&2; die "$EXIT_USAGE" "unknown command: $command_name" ;;
esac
[[ "$command_name" == "help" ]] && { usage; exit "$EXIT_OK"; }
if [[ "$command_name" == "full" ]]; then
    has_target=0
    for a in "${passthrough[@]+"${passthrough[@]}"}"; do
        [[ "$a" == "--experiment" || "$a" == "--all" ]] && has_target=1
    done
    [[ $has_target -eq 1 ]] || die "$EXIT_USAGE" "full needs --experiment ID or --all (see ./reproduce.sh list)"
fi

# Absolute host paths (Docker Desktop on Windows wants C:/... rather than /c/...).
abspath() {
    local p="$1"
    mkdir -p "$p" 2>/dev/null || true
    p="$(cd "$p" && pwd)"
    if command -v cygpath >/dev/null 2>&1; then cygpath -m "$p"; else printf '%s\n' "$p"; fi
}

# ---------------------------------------------------------------------------------------------
# Host prerequisites
# ---------------------------------------------------------------------------------------------
fingerprint() { bash "$REPL_DIR/scripts/fingerprint.sh" "$REPO_ROOT"; }

lock_value() {  # lock_value FILE KEY
    sed -n "s/^$2=//p" "$1" 2>/dev/null | head -n 1 | tr -d '\r'
}

report_resources() {
    local avail
    avail="$(df -Pk "$output_dir" 2>/dev/null | awk 'NR==2 {printf "%.1f", $4/1048576}')"
    info "Free disk at the output directory: ${avail:-unknown} GiB"
    info "Expected footprint: image ~6 GiB; archived artifacts ~1 GiB; quick reproduction outputs < 50 MiB;"
    info "  a full M4 campaign writes up to a few GiB and needs ~2 GiB of RAM per worker (see README)."
}

ensure_docker() {
    if ! command -v "$DOCKER" >/dev/null 2>&1; then
        die "$EXIT_ENV" "Docker is not installed or not on PATH. Install Docker Engine (Linux) or
       Docker Desktop (macOS/Windows): https://docs.docker.com/get-docker/ — then re-run."
    fi
    if ! "$DOCKER" info >/dev/null 2>&1; then
        die "$EXIT_ENV" "Docker is installed but the daemon is not reachable. Start Docker (Desktop),
       or on Linux make sure your user may use it (e.g. member of the 'docker' group)."
    fi
    local mem cpus
    mem="$("$DOCKER" info --format '{{.MemTotal}}' 2>/dev/null || echo 0)"
    cpus="$("$DOCKER" info --format '{{.NCPU}}' 2>/dev/null || echo '?')"
    info "Docker $("$DOCKER" version --format '{{.Server.Version}}' 2>/dev/null || echo '?'): ${cpus} CPUs, $(awk -v m="$mem" 'BEGIN {printf "%.1f", m/1073741824}') GiB RAM available to containers"
    if [[ "$mem" =~ ^[0-9]+$ && "$mem" -gt 0 && "$mem" -lt 4000000000 ]]; then
        warn "less than 4 GiB of RAM is available to containers; smoke tests may run out of memory"
    fi
}

build_image() {
    local tag="$1" revision
    revision="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]]; then
        revision="$revision-dirty"
    fi
    info "Building $tag for $platform from this checkout (first build takes 20–40 minutes)"
    DOCKER_BUILDKIT=1 "$DOCKER" build \
        --platform "$platform" \
        --file "$REPL_DIR/docker/Dockerfile" \
        --build-arg "SOURCE_FINGERPRINT=$host_fp" \
        --build-arg "SOURCE_REVISION=$revision" \
        --tag "$tag" \
        "$REPO_ROOT" || die "$EXIT_ENV" "image build failed (see the build log above)"
}

resolve_image() {
    local host_fp="$1" lock="$REPL_DIR/image.lock" repo digest tag
    repo="$(lock_value "$lock" IMAGE_REPOSITORY)"
    digest="$(lock_value "$lock" IMAGE_DIGEST)"
    tag="$(lock_value "$lock" IMAGE_TAG)"
    local local_tag="sarimax-ijf-replication:local-${host_fp:0:12}"

    if [[ -n "$image_ref" ]]; then
        if ! "$DOCKER" image inspect "$image_ref" >/dev/null 2>&1; then
            info "Pulling $image_ref"
            "$DOCKER" pull --platform "$platform" "$image_ref" || die "$EXIT_ENV" "cannot pull $image_ref"
        fi
        printf '%s\n' "$image_ref"; return
    fi
    if [[ $force_build -eq 1 ]]; then
        build_image "$local_tag" >&2
        printf '%s\n' "$local_tag"; return
    fi
    if [[ -n "$digest" ]]; then
        local ref="$repo@$digest"
        if ! "$DOCKER" image inspect "$ref" >/dev/null 2>&1; then
            info "Pulling the published image $repo:$tag ($digest)" >&2
            if ! "$DOCKER" pull --platform "$platform" "$ref" >&2; then
                [[ $no_build -eq 1 ]] && die "$EXIT_ENV" "cannot pull $ref and --no-build was given"
                warn "cannot pull $ref; falling back to a local build of this checkout"
                build_image "$local_tag" >&2
                printf '%s\n' "$local_tag"; return
            fi
        fi
        printf '%s\n' "$ref"; return
    fi
    # No published image recorded yet (pre-release checkout): use or build a local image.
    if "$DOCKER" image inspect "$local_tag" >/dev/null 2>&1; then
        printf '%s\n' "$local_tag"; return
    fi
    [[ $no_build -eq 1 ]] && die "$EXIT_ENV" "no published image is recorded in $REPL_REL/image.lock and --no-build was given"
    info "No published image is recorded in $REPL_REL/image.lock; building one from this checkout" >&2
    build_image "$local_tag" >&2
    printf '%s\n' "$local_tag"
}

verify_image_identity() {
    local ref="$1" host_fp="$2" image_fp revision
    image_fp="$("$DOCKER" image inspect --format '{{ index .Config.Labels "io.github.lampspuc.sarimax.ijf.fingerprint" }}' "$ref" 2>/dev/null || true)"
    revision="$("$DOCKER" image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "$ref" 2>/dev/null || true)"
    info "Image $ref"
    info "  built from revision ${revision:-unknown}, source fingerprint ${image_fp:-missing}"
    info "  this checkout's source fingerprint                    $host_fp"
    if [[ "$image_fp" == "$host_fp" ]]; then
        printf '%s  image matches this checkout%s\n' "$c_green" "$c_off"
        image_status="VERIFIED"
    elif [[ $allow_unverified -eq 1 ]]; then
        warn "the image was NOT built from this checkout; continuing because of --allow-unverified-image"
        image_status="UNVERIFIED"
    else
        die "$EXIT_ENV" "the image was not built from this checkout (fingerprint mismatch).
       Check out the release the image was built from, or run with --build to build an image
       from this checkout."
    fi
}

# ---------------------------------------------------------------------------------------------
# Running the in-container orchestrator
# ---------------------------------------------------------------------------------------------
image_status="NATIVE"

run_native() {
    local py
    py="$(command -v python3 || command -v python || true)"
    [[ -n "$py" ]] || die "$EXIT_ENV" "--native needs python3 on PATH"
    warn "native mode: host toolchains are used; this is not the supported reproduction path"
    IJF_RUNTIME=native \
    IJF_REPO_ROOT="$REPO_ROOT" \
    IJF_ARTIFACTS_DIR="$artifacts_dir" \
    IJF_DATA_DIR="$data_dir" \
    IJF_OUTPUT_DIR="$output_dir" \
    IJF_IMAGE_STATUS="NATIVE" \
    IJF_HOST_FINGERPRINT="$(fingerprint)" \
    IJF_OFFLINE="$offline" \
        "$py" "$REPL_DIR/scripts/ijf.py" "$@"
}

run_container() {  # run_container REF NETWORK(none|bridge) RW_INPUTS(0|1) ARGS...
    local ref="$1" network="$2" rw_inputs="$3"
    shift 3
    local ro=":ro"
    [[ "$rw_inputs" -eq 1 ]] && ro=""
    local -a args=(run --rm --init --network "$network" --platform "$platform"
        -e IJF_RUNTIME=docker
        -e "IJF_IMAGE_REF=$ref"
        -e "IJF_IMAGE_STATUS=$image_status"
        -e "IJF_HOST_FINGERPRINT=$host_fp"
        -e "IJF_OFFLINE=$offline"
        -v "$(abspath "$output_dir"):/ijf/output"
        -v "$(abspath "$artifacts_dir"):/ijf/artifacts$ro"
        -v "$(abspath "$data_dir"):/ijf/data$ro")
    if [[ -n "$from_file" && "${1:-}" == "fetch" ]]; then   # a local bundle tarball, read-only, for `fetch --from`
        args+=(-v "$(abspath "$(dirname "$from_file")")/$(basename "$from_file"):/ijf/import/$(basename "$from_file"):ro")
        set -- "$@" --from "/ijf/import/$(basename "$from_file")"
    fi
    if command -v id >/dev/null 2>&1 && [[ "$(uname -s)" == Linux ]]; then
        args+=(--user "$(id -u):$(id -g)")
    fi
    [[ -t 0 && -t 1 ]] && args+=(-t)
    [[ "${1:-}" == "shell" ]] && args+=(-i --entrypoint bash) && set --
    # MSYS_NO_PATHCONV stops Git Bash on Windows from rewriting the container paths.
    MSYS_NO_PATHCONV=1 "$DOCKER" "${args[@]}" "$ref" "$@"
}

main() {
    mkdir -p "$output_dir" "$artifacts_dir" "$data_dir"
    host_fp="$(fingerprint)"

    if [[ $native -eq 1 ]]; then
        local rc=0
        case "$command_name" in
            quick)
                run_native quick "${passthrough[@]+"${passthrough[@]}"}" || rc=$? ;;
            build|shell) die "$EXIT_USAGE" "$command_name is not available with --native" ;;
            *)  run_native "$command_name" "${passthrough[@]+"${passthrough[@]}"}" || rc=$? ;;
        esac
        exit "$rc"
    fi

    ensure_docker
    report_resources

    if [[ "$command_name" == "build" ]]; then
        build_image "sarimax-ijf-replication:local-${host_fp:0:12}"
        info "Built sarimax-ijf-replication:local-${host_fp:0:12}"
        exit "$EXIT_OK"
    fi

    local ref
    ref="$(resolve_image "$host_fp")"
    verify_image_identity "$ref" "$host_fp"

    local rc=0
    case "$command_name" in
        fetch) run_container "$ref" bridge 1 fetch "${passthrough[@]+"${passthrough[@]}"}" || rc=$? ;;
        shell) run_container "$ref" none 0 shell || rc=$? ;;
        quick)
            # The archived artifacts are fetched (and verified) once if they are missing; a
            # directory given with --artifacts is used as it is (check verifies it).
            if [[ $offline -eq 0 && $artifacts_explicit -eq 0 && ! -f "$artifacts_dir/.complete" ]]; then
                run_container "$ref" bridge 1 fetch artifacts --if-available || rc=$?
                [[ $rc -eq 0 ]] || warn "the artifact bundle could not be fetched; tables will report it as MISSING"
                rc=0
            fi
            run_container "$ref" none 0 quick || rc=$? ;;
        full)
            # A full run reads the M4 data; fetch and verify it first unless --offline.
            if [[ $offline -eq 0 ]]; then
                run_container "$ref" bridge 1 fetch m4 --if-needed "${passthrough[@]+"${passthrough[@]}"}" || rc=$?
                [[ $rc -eq 0 ]] || exit "$rc"
            fi
            run_container "$ref" none 0 full "${passthrough[@]+"${passthrough[@]}"}" || rc=$? ;;
        *)  run_container "$ref" none 0 "$command_name" "${passthrough[@]+"${passthrough[@]}"}" || rc=$? ;;
    esac
    if [[ $rc -eq 0 ]]; then
        printf '%sreproduce.sh %s: SUCCESS%s (outputs in %s)\n' "$c_green" "$command_name" "$c_off" "$output_dir"
    else
        printf '%sreproduce.sh %s: FAILED (exit %d)%s — see the report in %s/reports\n' "$c_red" "$command_name" "$rc" "$c_off" "$output_dir" >&2
    fi
    exit "$rc"
}

main
