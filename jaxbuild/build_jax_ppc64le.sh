#!/bin/bash
#SBATCH -p el8-rpi
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 6:00:00
#SBATCH -J jaxlib-ppc64le
#SBATCH -o /gpfs/u/home/QCSC/QCSCnkle/scratch-shared/ek/jaxbuild/build-%j.log
#
#==============================================================================
# End-to-end CPU-only jaxlib build for POWER9 / ppc64le  (RPI DCS cluster)
#
#   sbatch build_jax_ppc64le.sh 3.11
#   sbatch build_jax_ppc64le.sh 3.12
#
# Creates its own build env, patches the five things that break on this
# platform, builds the wheel, installs it into a runtime env, and verifies
# numerical correctness against numpy. Idempotent and safe to re-run.
#
# Everything lands under $EK/jaxbuild:
#   src/jax            jax source checkout (shared between versions)
#   dist/py3.NN        the built wheels, per Python version
#   bazelcache         shared Bazel disk cache (survives jobs; big win)
#   godist             downloaded Go SDK tarball
#   build-<jobid>.log  this log
#
# Conda envs created:
#   jaxbuild3NN        build toolchain (gcc 12, bazel 6.5.0, python 3.NN)
#   jax3NN             runtime env with jax installed and verified
#
#------------------------------------------------------------------------------
# WHY EACH WORKAROUND EXISTS  (see also: memory/jaxlib-ppc64le-build.md)
#
# 1. No jaxlib for ppc64le exists in conda-forge, Open-CE, or PyPI. Source
#    build is the only option. Note that `conda search jax` shows noarch hits
#    on every architecture and proves nothing -- jaxlib is the compiled part.
#
# 2. jax 0.4.35 is the chosen version. conda-forge's newest ppc64le Bazel is
#    6.6.0, and jaxlib pins an exact Bazel in .bazelversion: 0.4.30-0.5.0 pin
#    6.5.0, while newer jax needs Bazel 7.x which is unavailable here. Also
#    jax 0.5+ routes CPU ops through XNNPACK, which has no POWER support.
#
# 3. gcc is pinned to 12. Unpinned conda-forge gives gcc 16, which fails on
#    2024-era Abseil/XLA sources; the system gcc 8.4.1 is too old for XLA's
#    C++17. Also build.py defaults to clang, which isn't installed here, hence
#    --use_clang=false.
#
# 4. Bazel's output base defaults to ~/.cache/bazel and $HOME on this cluster
#    is a 10GB fileset reporting 100% full with ~8MB free. HOME is overridden
#    for the build call only. Do NOT redirect via TEST_TMPDIR: that also puts
#    Bazel in test mode (max_idle_secs=15) and the server dies mid-startup
#    with "couldn't connect to server ... exit 37".
#
# 5. gRPC's grpc_extra_deps() unconditionally registers a Go toolchain even
#    though jaxlib compiles no Go. rules_go resolves SDKs via golang.org,
#    which this site's proxy blocks (dl.google.com is allowed). Passing only
#    `version` does NOT avoid the blocked request -- the `sdks` dict with an
#    explicit sha256 is required. rules_go then refuses a version argument
#    once an SDK rule exists, so that argument is stripped from gRPC's call.
#
# 6. BoringSSL's base.h has no __PPC64__ branch and hits
#    #error "Unknown target CPU", cascading into hundreds of BN_ULONG errors.
#    We add a branch defining only OPENSSL_64_BIT -- not OPENSSL_PPC64LE,
#    which would pull in POWER assembly this build doesn't link.
#
# Patches 5b and 6 edit files inside Bazel's output base. On a FRESH build env
# those repos aren't fetched until Bazel runs, so the build loop below retries
# up to three times, re-applying patches against newly fetched repos. That is
# expected behaviour on a first run, not a failure.
#==============================================================================

set -eo pipefail   # NOT -u: conda's compiler deactivation hooks use unset vars

#--------------------------------------------------------------- configuration
PYVER="${1:-3.11}"
case "$PYVER" in
  3.11|3.12) ;;
  *) echo "usage: $0 [3.11|3.12]   (got '$PYVER')" >&2; exit 2 ;;
esac
PYTAG="${PYVER//./}"                      # 3.11 -> 311

EK=/gpfs/u/home/QCSC/QCSCnkle/scratch-shared/ek
CONDA_SH=/gpfs/u/home/QCSC/QCSCnkle/scratch/miniconda3/etc/profile.d/conda.sh
JAX_TAG=jax-v0.4.35
JAX_PIP_VERSION=0.4.35
GO_VERSION=1.21.5

BUILD_ENV=jaxbuild${PYTAG}
RUN_ENV=jax${PYTAG}
ROOT=$EK/jaxbuild
SRC=$ROOT/src/jax
DIST=$ROOT/dist/py${PYVER}

mkdir -p "$ROOT"/{tmp,pipcache,bazelcache,godist,src} "$DIST"

echo "=============================================================="
echo " jaxlib ppc64le build"
echo "   python      : $PYVER"
echo "   jax         : $JAX_TAG"
echo "   build env   : $BUILD_ENV"
echo "   runtime env : $RUN_ENV"
echo "   wheels ->   : $DIST"
echo "   started     : $(date)"
echo "=============================================================="

#------------------------------------------------------------------ build env
source "$CONDA_SH"
conda deactivate 2>/dev/null || true

if ! conda env list | grep -qE "^${BUILD_ENV}\s"; then
  echo "--- creating $BUILD_ENV (one time, a few minutes) ---"
  conda create -y -n "$BUILD_ENV" -c conda-forge \
    "python=${PYVER}" numpy scipy wheel setuptools packaging build \
    bazel=6.5.0 gcc_linux-ppc64le=12 gxx_linux-ppc64le=12
else
  echo "--- reusing existing $BUILD_ENV ---"
fi
conda activate "$BUILD_ENV"

export TMPDIR=$ROOT/tmp
export PIP_CACHE_DIR=$ROOT/pipcache
unset TEST_TMPDIR                 # sbatch inherits the submitting shell's env
export BAZEL_HOME=/tmp/bzlhome_$USER
mkdir -p "$BAZEL_HOME"

#------------------------------------------------------------------- preflight
echo "--- preflight ---"
echo "host    : $(hostname)"
echo "arch    : $(uname -m)"
echo "threads : $(nproc)"
echo "python  : $(python -V 2>&1)"
echo "CC      : ${CC:-<unset>}"
[ -n "${CC:-}" ] && $CC --version | head -1
bazel --version
echo "scratch : $(df -h "$BAZEL_HOME" | tail -1)"

[ -n "${CC:-}" ] || { echo "FATAL: CC unset; conda compilers not active" >&2; exit 1; }

case "$(python -V 2>&1)" in
  *"$PYVER"*) ;;
  *) echo "FATAL: env python is not $PYVER" >&2; exit 1 ;;
esac

if [ "$(nproc)" -lt 32 ]; then
  echo "FATAL: only $(nproc) threads; --exclusive did not take effect." >&2
  echo "       check: scontrol show job $SLURM_JOB_ID" >&2
  exit 1
fi

JOBS=$(nproc); [ "$JOBS" -gt 48 ] && JOBS=48   # SMT-4: 160 threads, 40 cores
echo "jobs    : $JOBS"

#---------------------------------------------------------------------- source
echo "--- source ---"
if [ ! -d "$SRC/.git" ]; then
  git clone --filter=blob:none https://github.com/jax-ml/jax.git "$SRC"
fi
cd "$SRC"
git fetch --tags --quiet || true
git checkout -f "$JAX_TAG"
git log -1 --oneline

#-------------------------------------------------------------------- Go SDK
GO_TGZ=go${GO_VERSION}.linux-ppc64le.tar.gz
if [ ! -s "$ROOT/godist/$GO_TGZ" ]; then
  echo "--- downloading $GO_TGZ (dl.google.com is reachable; golang.org is not) ---"
  curl -fL --retry 3 -o "$ROOT/godist/$GO_TGZ" "https://dl.google.com/go/$GO_TGZ"
fi
export GO_TGZ GO_VERSION
export GO_SHA256=$(sha256sum "$ROOT/godist/$GO_TGZ" | cut -d' ' -f1)
echo "go sdk  : $GO_TGZ  sha256=$GO_SHA256"

#--------------------------------------------------------------------- patches
apply_patches() {
  echo "--- applying patches ---"

  # (a) WORKSPACE: declare @go_sdk with explicit hash before xla_workspace0()
  python - <<'PY'
import os, re, pathlib
p = pathlib.Path("WORKSPACE")
s = p.read_text()
if "PATCHED_GO_SDK" in s:
    print("  WORKSPACE: already patched")
else:
    patch = '''
# --- PATCHED_GO_SDK: pin Go SDK with explicit hash via a reachable mirror ---
# rules_go would otherwise query golang.org/dl, which this site's proxy blocks.
load("@io_bazel_rules_go//go:deps.bzl", "go_download_sdk")

go_download_sdk(
    name = "go_sdk",
    sdks = {{"linux_ppc64le": ("{tgz}", "{sha}")}},
    urls = ["https://dl.google.com/go/{{}}"],
    version = "{ver}",
)
# --- end patch --------------------------------------------------------------

'''.format(tgz=os.environ["GO_TGZ"], sha=os.environ["GO_SHA256"],
           ver=os.environ["GO_VERSION"])
    m = re.search(r'^xla_workspace0\(\)', s, re.M)
    if not m:
        raise SystemExit("FATAL: no xla_workspace0() in WORKSPACE")
    p.write_text(s[:m.start()] + patch + s[m.start():])
    print("  WORKSPACE: @go_sdk pinned")
PY

  # (b) gRPC: drop the version argument from go_register_toolchains()
  local grpc_extra
  grpc_extra=$(find "$CONDA_PREFIX/share/bazel" \
    -path '*com_github_grpc_grpc/bazel/grpc_extra_deps.bzl' 2>/dev/null | head -1)
  if [ -n "$grpc_extra" ]; then
    sed -i 's/go_register_toolchains(version = "[^"]*")/go_register_toolchains()/' \
      "$grpc_extra"
    echo "  grpc_extra_deps.bzl: $(grep -c 'go_register_toolchains()' "$grpc_extra") call(s) stripped"
  else
    echo "  grpc_extra_deps.bzl: not fetched yet (will patch on retry)"
  fi

  # (c) BoringSSL: add a __PPC64__ branch to base.h
  local bssl
  bssl=$(find "$CONDA_PREFIX/share/bazel" \
    -path '*boringssl/src/include/openssl/base.h' 2>/dev/null | head -1)
  if [ -n "$bssl" ]; then
    BSSL_BASE="$bssl" python - <<'PY'
import os, pathlib
p = pathlib.Path(os.environ["BSSL_BASE"])
s = p.read_text()
if "PATCHED_PPC64LE" in s:
    print("  boringssl base.h: already patched")
else:
    # The #else and the #error are separated by a comment block, so anchor on
    # the #else immediately preceding BoringSSL's "Note ..." comment.
    anchor = '#else\n// Note BoringSSL only supports standard 32-bit and 64-bit'
    if anchor not in s:
        raise SystemExit("FATAL: base.h fallthrough not found; inspect manually")
    insert = (
        '// PATCHED_PPC64LE: POWER8+ little-endian is a standard 64-bit\n'
        "// two's-complement little-endian target, exactly what BoringSSL\n"
        '// requires below. Define only OPENSSL_64_BIT, not OPENSSL_PPC64LE,\n'
        '// so generic C is used instead of POWER asm this build never links.\n'
        '#elif defined(__PPC64__) || defined(__powerpc64__)\n'
        '#define OPENSSL_64_BIT\n')
    p.write_text(s.replace(anchor, insert + anchor, 1))
    print(f"  boringssl base.h: patched ({p})")
PY
  else
    echo "  boringssl base.h: not fetched yet (will patch on retry)"
  fi
}

#----------------------------------------------------------------------- build
run_build() {
  HOME="$BAZEL_HOME" python build/build.py \
    --bazel_path="$(which bazel)" \
    --enable_cuda=false \
    --use_clang=false \
    --target_cpu_features=default \
    --bazel_options=--jobs="$JOBS" \
    --bazel_options=--disk_cache="$ROOT/bazelcache" \
    --bazel_options=--repo_env=CC="$CC" \
    --bazel_options=--repo_env=CXX="$CXX" \
    --output_path="$DIST"
}

SECONDS=0
built=0
for attempt in 1 2 3; do
  echo "=== build attempt $attempt/3 ==="
  apply_patches
  set +e
  run_build
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then built=1; break; fi
  echo "--- attempt $attempt failed (rc=$rc)."
  echo "    On a fresh env this is expected: external repos are fetched during"
  echo "    the run, so patches (b) and (c) only become applicable afterwards."
  echo "    Re-applying patches and retrying."
done

if [ "$built" -ne 1 ]; then
  echo "FATAL: build failed after 3 attempts. The first compiler error is what"
  echo "       matters -- find it with:" >&2
  echo "       grep -n -m3 -B5 -A30 'error:' $ROOT/build-${SLURM_JOB_ID}.log" >&2
  exit 1
fi

echo "=== build succeeded in $((SECONDS / 60)) min ==="
ls -lh "$DIST"

WHEEL=$(ls -t "$DIST"/jaxlib-*.whl | head -1)
echo "wheel: $WHEEL"

#---------------------------------------------------------- runtime env + test
# Dependencies come from conda, not pip: ml_dtypes has no ppc64le wheel on
# PyPI, so pip would try to build it, which drags in a numpy source build that
# fails against the system gcc 8.4.1 ("NumPy requires GCC >= 9.3"). With the
# deps present, --no-deps stops pip from trying to resolve anything.
echo "--- creating runtime env $RUN_ENV ---"
conda create -y -n "$RUN_ENV" -c conda-forge \
  "python=${PYVER}" numpy scipy ml_dtypes opt_einsum || {
    echo "conda could not supply ml_dtypes; falling back to a compiled install"
    conda create -y -n "$RUN_ENV" -c conda-forge \
      "python=${PYVER}" numpy scipy opt_einsum \
      gcc_linux-ppc64le=12 gxx_linux-ppc64le=12 pybind11
    conda activate "$RUN_ENV"
    pip install --no-build-isolation ml_dtypes
    conda deactivate
  }

conda deactivate
conda activate "$RUN_ENV"
python -V

pip install --no-deps "$WHEEL"
pip install --no-deps "jax==${JAX_PIP_VERSION}"

echo "--- verification ---"
# Compiling on an untested architecture proves the code built, not that XLA's
# POWER9 code generation is correct. These checks establish the latter.
python - <<'PY'
import numpy as np, jax, jax.numpy as jnp

print("jax     :", jax.__version__)
print("jaxlib  :", jax.lib.__version__)
print("devices :", jax.devices())

x = jnp.arange(10.0)
xn = np.arange(10.0)

f = jax.jit(lambda v: jnp.sum(jnp.sin(v) ** 2))
got, want = float(f(x)), float(np.sum(np.sin(xn) ** 2))
print(f"jit     : {got:.12f} vs numpy {want:.12f}")
assert np.allclose(got, want), "JIT NUMERICAL MISMATCH"

g = jax.grad(lambda v: jnp.sum(jnp.sin(v) ** 2))
assert np.allclose(g(x), 2 * np.sin(xn) * np.cos(xn)), "GRAD MISMATCH"
print("grad    : ok")

a = jax.random.normal(jax.random.PRNGKey(0), (256, 256))
assert np.allclose(a @ a, np.asarray(a) @ np.asarray(a), atol=1e-3), "MATMUL MISMATCH"
print("matmul  : ok")

v = jax.vmap(lambda r: jnp.sum(r ** 2))(jnp.ones((8, 4)))
assert np.allclose(v, np.full(8, 4.0)), "VMAP MISMATCH"
print("vmap    : ok")

c = jnp.fft.fft(jnp.arange(8.0))
assert np.allclose(c, np.fft.fft(np.arange(8.0)), atol=1e-5), "FFT MISMATCH"
print("fft     : ok")

print("\nALL CHECKS PASSED")
PY

echo "=============================================================="
echo " done $(date)"
echo
echo " wheel : $WHEEL"
echo " env   : conda activate $RUN_ENV"
echo
echo " to install into another python ${PYVER} env:"
echo "   conda install -c conda-forge ml_dtypes opt_einsum numpy scipy"
echo "   pip install --no-deps $WHEEL"
echo "   pip install --no-deps jax==${JAX_PIP_VERSION}"
echo "=============================================================="
