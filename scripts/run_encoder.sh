#!/bin/bash
set -e

SCRIPTS_DIR=/RAEE/scripts

echo "run build_table_roberta.sh"
bash $SCRIPTS_DIR/build_table_roberta.sh

echo "run predict_roberta.sh"
bash $SCRIPTS_DIR/predict_roberta.sh

echo "end"