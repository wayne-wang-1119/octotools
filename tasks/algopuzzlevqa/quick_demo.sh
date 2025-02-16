#!/bin/bash

############ [1] Single Run ############
PROJECT_DIR="./"

LABEL="quick_demo"

TASK="algopuzzlevqa"
DATA_FILE="$TASK/data/data.json"
LOG_DIR="$TASK/logs/$LABEL"
OUT_DIR="$TASK/results/$LABEL"

LLM="gpt-4o"

ENABLED_TOOLS="Generalist_Solution_Generator_Tool"

i=0

python solve.py \
--index $i \
--task $TASK \
--data_file $DATA_FILE \
--llm_engine_name $LLM \
--output_json_dir $OUT_DIR \
--output_types direct \
--enabled_tools "$ENABLED_TOOLS" \
--max_time 300
