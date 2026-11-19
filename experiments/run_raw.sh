#!/bin/bash
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# usage: run_raw.sh <gpu> <tag> <runner/inference.py dotted args...>
gpu=$1; tag=$2; shift 2
log=experiments/logs/$tag.log
echo "[start] $(date '+%F %T') GPU=$gpu cmd: python3 runner/inference.py $*" > $log
s=$(date +%s)
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="${PYTHONPATH}:$(pwd)" python3 runner/inference.py "$@" >> $log 2>&1
rc=$?
echo "[end] $(date '+%F %T') rc=$rc wall=$(( $(date +%s)-s ))s" >> $log
