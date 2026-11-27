#!/usr/bin/env bash
# Metadata-only check of the WMT direct-assessment claim in REPORT.md (no 638 MB download).
curl -s --max-time 20 "https://huggingface.co/api/datasets/RicardoRei/wmt-da-human-evaluation" | python3 -c "import json,sys; d=json.load(sys.stdin); print('head sha', d.get('sha'))"
curl -s --max-time 20 "https://huggingface.co/api/datasets/RicardoRei/wmt-da-human-evaluation/tree/main" | python3 -c "import json,sys; print([(x.get('path'), x.get('size')) for x in json.load(sys.stdin)])"
# First 2 KB of train.csv (header only) to see the columns.
curl -sL --max-time 20 -r 0-2047 "https://huggingface.co/datasets/RicardoRei/wmt-da-human-evaluation/resolve/main/train.csv" | head -1
