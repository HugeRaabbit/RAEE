#!/bin/bash
set -e

SCRIPTS_DIR=/RAEE/scripts

echo "run build_table_llama.sh"
bash $SCRIPTS_DIR/build_table_llama.sh

echo "run predict_llama.sh"
bash $SCRIPTS_DIR/predict_llama.sh

echo "end"