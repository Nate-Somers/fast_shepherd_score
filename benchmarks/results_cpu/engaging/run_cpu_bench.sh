#!/bin/bash
#SBATCH --job-name=fss_cpu_bench
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/home/nsomers/Software/Github/fast_shepherd_score/benchmarks/results_cpu/engaging/slurm-%j.out

REPO=/home/nsomers/Software/Github/fast_shepherd_score
PY=/home/nsomers/.conda/envs/SimModelEnv/bin/python
JX=/home/nsomers/.conda/envs/fss/bin/python
cd "$REPO" || exit 1

echo "================= NODE INFO ================="
date
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE  nproc=$(nproc)"
lscpu | grep -E "Model name|^CPU\(s\):|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|NUMA node\(s\)"

# physical-core count (unique core,socket pairs); cap the ladder there
PHYS=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
echo "physical cores = $PHYS"
ALL="1 4 8 16 32 48 64 96 128"
LADDER=""
for p in $ALL; do [ "$p" -le "$PHYS" ] && LADDER="$LADDER $p"; done
echo "$LADDER" | grep -qw "$PHYS" || LADDER="$LADDER $PHYS"
echo "core ladder =$LADDER"

echo; echo "########## RUN 1/3: numba THREADS scaling (no JAX) ##########"
date
$PY -m benchmarks.benchmark_cpu --no-original --numba-mode threads \
    --procs $LADDER --tag eng_threads --cap 20 --budget 8 2>&1
echo "RUN1 exit=$?"

echo; echo "########## RUN 2/3: numba POOL scaling (no JAX) ##########"
date
$PY -m benchmarks.benchmark_cpu --no-original --numba-mode pool \
    --procs $LADDER --tag eng_pool --cap 20 --budget 8 2>&1
echo "RUN2 exit=$?"

echo; echo "########## RUN 3/3: original JAX vs fork numba (procs 1 48) ##########"
date
$PY -m benchmarks.benchmark_cpu --orig-python "$JX" --numba-mode threads \
    --procs 1 48 --sizes 10 100 1000 --tag eng_vs_jax --cap 15 --budget 6 2>&1
echo "RUN3 exit=$?"

echo; echo "================= DONE ================="
date
