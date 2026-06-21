#!/usr/bin/env bash
# Cluster recon for the fig5/fig6 runs. Run via:
#   ssh -S ~/.ssh/cm-eng mit-engaging 'bash -s' < paper/_engaging/probe.sh
# Best-effort; never exits non-zero on a missing tool.
set +e
# make `module` available in this non-login shell
for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /etc/profile.d/z00_lmod.sh \
         /usr/share/lmod/lmod/init/bash; do [ -f "$f" ] && source "$f" 2>/dev/null; done

echo "=================== HOST ==================="
hostname; whoami; echo "HOME=$HOME"; echo "PWD=$PWD"
uname -a 2>/dev/null
echo; echo "=================== SCRATCH / STORAGE ==================="
for d in /home/$USER /nobackup1/$USER /pool001/$USER /nfs/*/$USER ~/scratch /orcd/*/$USER; do
  [ -d "$d" ] && echo "DIR $d  ($(df -h "$d" 2>/dev/null | awk 'NR==2{print $4" free"}'))"
done

echo; echo "=================== MODULES (cuda / conda / anaconda) ==================="
command -v module >/dev/null && { module --version 2>&1 | head -1; \
  module -t avail 2>&1 | grep -iE 'cuda|conda|anaconda|gcc|cmake|miniforge' | head -40; } \
  || echo "no module command"

echo; echo "=================== CONDA ==================="
command -v conda >/dev/null && { echo "conda: $(command -v conda)"; conda env list 2>/dev/null; } \
  || echo "conda not on PATH (may need: module load miniforge / anaconda)"
ls -d ~/miniconda3 ~/anaconda3 ~/miniforge3 2>/dev/null

echo; echo "=================== REPO PRESENCE ==================="
find "$HOME" /orcd/*/"$USER" /nobackup1/"$USER" /pool001/"$USER" -maxdepth 5 \
     -type d \( -iname 'fast_shepherd_score' -o -iname 'shepherd_score' -o -iname 'shepherd-score' \) \
     2>/dev/null | head -20
echo "--- electrostatic_scoring.py hits ---"
find "$HOME" -maxdepth 6 -name electrostatic_scoring.py 2>/dev/null | head

echo; echo "=================== SLURM: pi_melkin / node3615 ==================="
command -v sinfo >/dev/null && {
  echo "--- partition pi_melkin ---"; sinfo -p pi_melkin -o '%P %a %l %D %t %N %G' 2>/dev/null
  echo "--- any node matching 3615 ---"; sinfo -N -o '%N %P %t %G' 2>/dev/null | grep -i 3615
  echo "--- my partitions ---"; sinfo -o '%P' 2>/dev/null | sort -u | tr '\n' ' '; echo
  echo "--- scontrol partition pi_melkin ---"; scontrol show partition pi_melkin 2>/dev/null | head -20
} || echo "no slurm here (login node only? sinfo missing)"

echo; echo "=================== BUILD TOOLS ==================="
for t in gcc g++ cmake git nvcc rsync; do
  p=$(command -v $t 2>/dev/null); echo "$t: ${p:-MISSING} $([ -n "$p" ] && $t --version 2>/dev/null | head -1)"
done

echo; echo "=================== NETWORK (github / dudez) ==================="
timeout 12 git ls-remote https://github.com/molecularinformatics/roshambo HEAD >/dev/null 2>&1 \
  && echo "github roshambo: reachable" || echo "github roshambo: NOT reachable (proxy needed?)"
timeout 12 bash -c 'curl -sI https://dudez.docking.org >/dev/null 2>&1' \
  && echo "dudez.docking.org: reachable" || echo "dudez: not reachable / curl missing"
echo; echo "=================== DONE ==================="
