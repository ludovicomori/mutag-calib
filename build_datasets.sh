#!/usr/bin/env bash
shopt -s nullglob

for cfg in datasets/datasets_definitions*.json; do
  echo "Processing $cfg"
  pocket-coffea build-datasets --cfg "$cfg" -o -rs 'T[123]_CH_\w+'
done

