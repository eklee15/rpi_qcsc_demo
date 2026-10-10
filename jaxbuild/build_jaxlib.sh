#!/bin/bash
#SBATCH -p el8-rpi
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 6:00:00
#SBATCH -J jaxlib-ppc64le
#SBATCH -o /gpfs/u/home/QCSC/QCSCnkle/scratch-shared/ek/jaxbuild/build-%j.log
#
# Build CPU-only jaxlib for POWER9 / ppc64le on the RPI DCS cluster.
#
#   sbatch build_jaxlib.sh
#
# Deliberate choices, so future-you knows why:
#   * no `set -u`      - conda compiler deactivation hooks reference unset vars
#   * TEST_TMPDIR      - redirects Bazel's output base off the full $HOME
#                        (default is ~/.cache/bazel, which has ~8MB free)
#   * --exclusive      - without it SLURM hands out 4 CPUs and the build crawls
#   * gcc 12           - gcc 16 breaks on 2024-era Abseil/XLA sources
#   * jax 0.4.35       - pins Bazel 6.5.0 (installable from conda-forge) and
#                        predates XLA's XNNPACK CPU path, which has no POWER support
#
set -eo pipefail

EK=/gpfs/u/home/QCSC/QCSCnkle/scratch-shared/ek
CONDA_SH=/gpfs/u/home/QCSC/QCSCnkle/scratch/miniconda3/etc/profile.d/conda.sh
JAX_TAG=jax-v0.4.35
SRC=$EK/jaxbuild/src/jax
DIST=$EK/jaxbuild/dist

mkdir -p "$EK/jaxbuild"/{tmp,pipcache,bazelcache,dist}

# ---------------------------------------------------------------- environment
source "$CONDA_SH"
conda deactivate 2>/dev/null || true
conda activate jaxbuild

export TMPDIR=$EK/jaxbuild/tmp
export PIP_CACHE_DIR=$EK/jaxbuild/pipcache

# sbatch exports the submitting shell's environment. If TEST_TMPDIR is set
# there, Bazel silently switches to test mode (max_idle_secs=15) and its
# server can die mid-startup. Make sure it is not inherited.
unset TEST_TMPDIR
# Bazel's output base defaults to $HOME/.cache/bazel, and $HOME has ~8MB free.
# Do NOT use TEST_TMPDIR to redirect it: that also puts Bazel in test mode with
# max_idle_secs=15, so the server shuts down mid-startup and you get
# "couldn't connect to server ... exit 37". Overriding HOME for the build call
# is side-effect free and keeps the output base on fast node-local disk.
export BAZEL_HOME=/tmp/bzlhome_$USER
mkdir -p "$BAZEL_HOME"

# If Bazel's server dies, its JVM log is the only place the reason appears.
dump_jvm_log() {
  echo "=== bazel JVM log (server startup failure diagnostics) ==="
  find "$BAZEL_HOME" -name jvm.out 2>/dev/null \
    -exec sh -c 'echo "--- {}"; tail -60 "{}"' \;
}
trap dump_jvm_log ERR

# ------------------------------------------------------------------ preflight
echo "================ preflight ================"
echo "host      : $(hostname)"
echo "arch      : $(uname -m)"
echo "threads   : $(nproc)"
echo "CC        : ${CC:-<unset>}"
echo "CXX       : ${CXX:-<unset>}"
[ -n "${CC:-}" ] && $CC --version | head -1
bazel --version
python --version
echo "free disk : $(df -h "$BAZEL_HOME" | tail -1)"

if [ -z "${CC:-}" ]; then
  echo "FATAL: CC is unset - conda compilers did not activate." >&2
  echo "Run: conda install -n jaxbuild -c conda-forge gcc_linux-ppc64le=12 gxx_linux-ppc64le=12" >&2
  exit 1
fi

case "$($CC -dumpversion)" in
  1[3-9]*|2[0-9]*)
    echo "WARNING: gcc $($CC -dumpversion) is newer than this XLA tree expects." >&2
    echo "         Expect errors in Abseil/protobuf. Pin gcc 12 if it fails." >&2
    ;;
esac

if [ "$(nproc)" -lt 32 ]; then
  echo "FATAL: only $(nproc) threads allocated - --exclusive did not take effect." >&2
  echo "       Check 'scontrol show job \$SLURM_JOB_ID' for the real allocation." >&2
  exit 1
fi

JOBS=$(nproc)
[ "$JOBS" -gt 48 ] && JOBS=48          # SMT-4: 160 threads over 40 real cores
echo "bazel jobs: $JOBS"

# ---------------------------------------------------------------------- source
echo "================ source ================"
cd "$SRC"
git checkout -f "$JAX_TAG"      # -f: discard the WORKSPACE patch from a prior run
git log -1 --oneline
cat .bazelversion

# ------------------------------------------------- proxy workaround: go_sdk
# Nothing in jaxlib compiles Go, but gRPC's grpc_extra_deps() unconditionally
# calls go_register_toolchains(), so Bazel must materialize a Go SDK during
# analysis. Two separate problems with that here:
#
#   1. rules_go discovers SDK filenames/hashes from golang.org/dl, which this
#      site's proxy blocks (403). Passing only `version` does NOT avoid that
#      request. Supplying the `sdks` dict with an explicit filename + sha256
#      skips discovery entirely, and dl.google.com is reachable.
#   2. rules_go hard-fails with "version set after go sdk rule declared" if
#      go_register_toolchains() is given a version once an SDK rule exists.
#      gRPC passes version="1.18.4", so that argument gets stripped below.
GO_VERSION=1.21.5
GO_TGZ=go${GO_VERSION}.linux-ppc64le.tar.gz
GO_DIST=$EK/jaxbuild/godist
mkdir -p "$GO_DIST"
if [ ! -s "$GO_DIST/$GO_TGZ" ]; then
  echo "downloading $GO_TGZ from dl.google.com ..."
  curl -fL --retry 3 -o "$GO_DIST/$GO_TGZ" "https://dl.google.com/go/$GO_TGZ"
fi
export GO_VERSION
export GO_SHA256=$(sha256sum "$GO_DIST/$GO_TGZ" | cut -d' ' -f1)
export GO_TGZ
echo "go sdk: $GO_TGZ  sha256=$GO_SHA256"

python - <<'PY'
import os, re, pathlib
p = pathlib.Path("WORKSPACE")
s = p.read_text()
patch = '''
# --- local patch: pin Go SDK with explicit hash, proxy-allowed mirror -------
load("@io_bazel_rules_go//go:deps.bzl", "go_download_sdk")

go_download_sdk(
    name = "go_sdk",
    sdks = {{"linux_ppc64le": ("{tgz}", "{sha}")}},
    urls = ["https://dl.google.com/go/{{}}"],
    version = "{ver}",
)
# --- end local patch --------------------------------------------------------

'''.format(tgz=os.environ["GO_TGZ"],
           sha=os.environ["GO_SHA256"],
           ver=os.environ["GO_VERSION"])
m = re.search(r'^xla_workspace0\(\)', s, re.M)
if not m:
    raise SystemExit("FATAL: no xla_workspace0() call found in WORKSPACE; "
                     "inspect it and move the patch insertion point")
p.write_text(s[:m.start()] + patch + s[m.start():])
print("WORKSPACE patched: @go_sdk pinned with explicit sha256")
PY

grep -n -A10 'local patch' WORKSPACE

# Strip the version argument from gRPC's toolchain registration. This edits the
# cached external repo in Bazel's output base; Bazel does not re-fetch external
# repos on content change, so the edit sticks for subsequent runs.
GRPC_EXTRA=$(find "$CONDA_PREFIX/share/bazel" \
  -path '*com_github_grpc_grpc/bazel/grpc_extra_deps.bzl' 2>/dev/null | head -1)
if [ -n "$GRPC_EXTRA" ]; then
  sed -i 's/go_register_toolchains(version = "[^"]*")/go_register_toolchains()/' \
    "$GRPC_EXTRA"
  echo "patched: $GRPC_EXTRA"
  grep -n 'go_register_toolchains' "$GRPC_EXTRA"
else
  echo "WARNING: grpc_extra_deps.bzl not found in the output base yet." >&2
  echo "         If the build fails on 'version set after go sdk rule'," >&2
  echo "         re-run this script: the repo will have been fetched by then." >&2
fi

# --------------------------------------------- boringssl: teach it about ppc64le
# BoringSSL's base.h tests for x86_64/aarch64/arm and falls through to
#   #error "Unknown target CPU"
# on POWER, which cascades into hundreds of BN_ULONG errors. gRPC's TLS stack
# is unused in a CPU-only single-process jaxlib, but it still has to compile.
#
# We define OPENSSL_64_BIT only, NOT OPENSSL_PPC64LE: the latter enables
# BoringSSL's POWER assembly (aesp8-ppc, ghashp8-ppc), which this Bazel build
# does not wire in, and would fail at link time. Generic C is what we want.
BSSL_BASE=$(find "$CONDA_PREFIX/share/bazel" \
  -path '*boringssl/src/include/openssl/base.h' 2>/dev/null | head -1)
if [ -n "$BSSL_BASE" ]; then
  BSSL_BASE="$BSSL_BASE" python - <<'PY'
import os, re, pathlib
p = pathlib.Path(os.environ["BSSL_BASE"])
s = p.read_text()
if "PATCHED_PPC64LE" in s:
    print("boringssl base.h already patched")
else:
    # The #else and the #error are separated by a comment block, so anchor on
    # the #else that immediately precedes BoringSSL's "Note ..." comment.
    anchor = '#else\n// Note BoringSSL only supports standard 32-bit and 64-bit'
    insert = ('// PATCHED_PPC64LE: POWER8+ little-endian is a standard 64-bit\n'
              '// two\'s-complement little-endian target, which is exactly what\n'
              '// BoringSSL requires below. Only OPENSSL_64_BIT is defined -- not\n'
              '// OPENSSL_PPC64LE -- so the generic C implementations are used\n'
              '// rather than POWER assembly this build does not link.\n'
              '#elif defined(__PPC64__) || defined(__powerpc64__)\n'
              '#define OPENSSL_64_BIT\n')
    if anchor not in s:
        raise SystemExit("FATAL: could not find the '#else' before BoringSSL's "
                         "'Note ...' comment in base.h; inspect it manually")
    new = s.replace(anchor, insert + anchor, 1)
    n = 1
    p.write_text(new)
    print(f"patched boringssl base.h: {p}")
PY
  grep -n -B2 -A6 'PATCHED_PPC64LE' "$BSSL_BASE" || true
else
  echo "WARNING: boringssl base.h not found; will fail on 'Unknown target CPU'" >&2
fi

# ----------------------------------------------------------------------- build
echo "================ build (started $(date)) ================"
SECONDS=0

HOME="$BAZEL_HOME" python build/build.py \
  --bazel_path="$(which bazel)" \
  --enable_cuda=false \
  --use_clang=false \
  --target_cpu_features=default \
  --bazel_options=--jobs="$JOBS" \
  --bazel_options=--disk_cache="$EK/jaxbuild/bazelcache" \
  --bazel_options=--repo_env=CC="$CC" \
  --bazel_options=--repo_env=CXX="$CXX" \
  --output_path="$DIST"

echo "================ build finished in $((SECONDS / 60)) min ================"
ls -lh "$DIST"

# ---------------------------------------------------------------- smoke test
# Installs into a throwaway env so a broken wheel can't poison anything.
echo "================ smoke test ================"
WHEEL=$(ls -t "$DIST"/jaxlib-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
  echo "No wheel produced - nothing to test." >&2
  exit 1
fi
echo "wheel: $WHEEL"

conda create -y -n jaxtest -c conda-forge python=3.11 numpy scipy
conda activate jaxtest
pip install "$WHEEL"
pip install "jax==0.4.35"          # pure-Python frontend, arch-independent

python - <<'PY'
import numpy as np, jax, jax.numpy as jnp
print("jax     :", jax.__version__)
print("jaxlib  :", jax.lib.__version__)
print("devices :", jax.devices())

f = jax.jit(lambda x: jnp.sum(jnp.sin(x) ** 2))
x = jnp.arange(10.0)
got, want = float(f(x)), float(np.sum(np.sin(np.arange(10.0)) ** 2))
print(f"jit     : {got:.10f} vs numpy {want:.10f}")
assert np.allclose(got, want), "NUMERICAL MISMATCH"

g = jax.grad(lambda x: jnp.sum(jnp.sin(x) ** 2))
print("grad    :", np.allclose(g(x), 2 * np.sin(x) * np.cos(x)))

k = jax.random.PRNGKey(0)
a = jax.random.normal(k, (256, 256))
print("matmul  :", np.allclose(a @ a, np.asarray(a) @ np.asarray(a), atol=1e-3))
print("ALL CHECKS PASSED")
PY

echo "================ done $(date) ================"
echo "To use it in your own env:"
echo "  conda activate rpi_demo   # or a fresh python=3.11 env"
echo "  pip install $WHEEL"
echo "  pip install jax==0.4.35"
