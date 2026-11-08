#!/bin/bash
# Starts the CPU kernel + GPU forward server of a session. Usage: wsp_session.sh RUN BUDGET [MAX_RUNS]
set -u
RUN=$1; BUDGET=$2; MAXR=${3:-300}
export WSP_SESSION=<server>/wsp/sessions/$RUN WSP_BUDGET=$BUDGET WSP_MAX_RUNS=$MAXR
mkdir -p $WSP_SESSION
PY=<server>/wsp/venv-hf/bin/python
$PY -u <server>/wsp/code/wsp_kernel.py > $WSP_SESSION/kernel.log 2>&1 &
K=$!
trap "kill $K 2>/dev/null" EXIT
$PY -u <server>/wsp/code/wsp_fwd.py > $WSP_SESSION/fwd.log 2>&1
