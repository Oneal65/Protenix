#!/bin/bash
# usage: run.sh <gpu> <tag> <protenix pred args...>
gpu=$1; tag=$2; shift 2
log=experiments/logs/$tag.log
echo "[start] $(date '+%F %T') GPU=$gpu cmd: protenix pred $*" > $log
s=$(date +%s)
CUDA_VISIBLE_DEVICES=$gpu protenix pred "$@" >> $log 2>&1
rc=$?
echo "[end] $(date '+%F %T') rc=$rc wall=$(( $(date +%s)-s ))s" >> $log
