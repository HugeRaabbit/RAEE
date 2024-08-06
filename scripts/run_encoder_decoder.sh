#!/bin/bash
set -e

SCRIPTS_DIR=/RAEE/scripts

echo "run build_table_t5.sh"
bash $SCRIPTS_DIR/build_table_t5.sh

echo "run predict_t5.sh"
bash $SCRIPTS_DIR/predict_t5.sh

echo "end"