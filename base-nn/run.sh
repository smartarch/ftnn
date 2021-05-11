#!/bin/bash

cd `dirname "$0"` || exit 1

ITERATION=$1
shift

FILENAME=$1
shift

WIDTHS="$*"
WIDTHS_STR=`echo "$WIDTHS" | tr ' ' '-'`
DATAFILE="./data/$FILENAME"
RESULTFILE="./results/$FILENAME-dense-${WIDTHS_STR}_$ITERATION.csv"

if [ ! -f "$DATAFILE" ]; then
    echo "Input file $DATAFILE not found..."
    exit 0
fi

if [ -f "$RESULTFILE" ]; then
    echo "Result $RESULTFILE already exists..."
    exit 0
fi

DATE=`date`
echo "Job dense-$WIDTHS_STR on $FILENAME iteration $ITERATION started ... $DATE"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
srun -p volta-lp --mem=32G --time=1-00:00:00 --gpus=1 --exclude=volta01 ch-run 'tensorflow.tensorflow:latest-gpu' -c /home/krulis/d3s python3 ./dense.py "$DATAFILE" $WIDTHS > $RESULTFILE 2> "$RESULTFILE.stderr"

DATE=`date`
echo "Job dense-$WIDTHS_STR on $FILENAME iteration $ITERATION concluded ... $DATE"

