#!/bin/bash

# What:
#   提供 Slurm 腳本共用的環境初始化、uv/python 解析與診斷輸出工具。
# Why:
#   讓 train / eval / postprocess 腳本只保留實驗差異，避免多份近似腳本長期漂移。

slurm_job_id() {
  printf '%s' "${SLURM_JOB_ID:-manual}"
}

slurm_node_name() {
  if [ -n "${SLURM_NODELIST:-}" ]; then
    printf '%s' "${SLURM_NODELIST}"
  else
    hostname
  fi
}

slurm_find_uv() {
  if [ -n "${UV_BIN:-}" ] && [ -x "${UV_BIN}" ]; then
    :
  elif [ -x "${HOME}/.local/bin/uv" ]; then
    UV_BIN="${HOME}/.local/bin/uv"
  elif command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    echo "uv not found. Set UV_BIN or install uv." >&2
    exit 1
  fi
  export UV_BIN
}

slurm_prepare_project() {
  PROJECT_DIR="$1"
  CREATE_RUNS_DIR="${2:-1}"

  mkdir -p "${PROJECT_DIR}/logs"
  if [ "${CREATE_RUNS_DIR}" = "1" ]; then
    mkdir -p "${PROJECT_DIR}/runs"
  fi
  cd "${PROJECT_DIR}" || exit 1
}

slurm_require_path() {
  local path="$1"
  local label="${2:-Path not found}"

  if [ ! -e "${path}" ]; then
    echo "${label}: ${path}" >&2
    exit 1
  fi
}

slurm_setup_base_env() {
  slurm_find_uv
  export UV_PROJECT_ENVIRONMENT="${PROJECT_DIR}/.venv"
  export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
}

slurm_setup_cuda_env() {
  slurm_setup_base_env

  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
  export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

  local py_ver site_packages
  py_ver="$("${UV_BIN}" run python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")"
  site_packages="${PROJECT_DIR}/.venv/lib/${py_ver}/site-packages"

  if [ -d "${site_packages}" ]; then
    export LD_LIBRARY_PATH="${site_packages}/nvidia/cudnn/lib:${site_packages}/nvidia/cublas/lib:${site_packages}/nvidia/cuda_runtime/lib:${site_packages}/nvidia/cuda_cupti/lib:${site_packages}/nvidia/cufft/lib:${site_packages}/nvidia/cusolver/lib:${site_packages}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
  fi
}

slurm_setup_cpu_env() {
  slurm_setup_base_env

  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
}

slurm_print_header() {
  local title="$1"
  shift

  echo "==========================="
  echo "${title}"
  echo "==========================="
  echo "Job ID   : $(slurm_job_id)"
  echo "Node     : $(slurm_node_name)"

  while [ "$#" -ge 2 ]; do
    printf '%-9s: %s\n' "$1" "$2"
    shift 2
  done

  echo "==========================="
}

slurm_print_python_info() {
  "${UV_BIN}" --version
  "${UV_BIN}" run python -V
  "${UV_BIN}" run python -c "import jax; print('JAX', jax.__version__); print('devices:', jax.devices())"
}

slurm_print_gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  fi
}

slurm_print_footer() {
  local exit_code="$1"
  local start_time="$2"
  local end_time elapsed

  end_time="$(date +%s)"
  elapsed="$((end_time - start_time))"

  echo "Exit code : ${exit_code}"
  echo "Elapsed   : ${elapsed}s"
}
