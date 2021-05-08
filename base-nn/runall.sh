#!/bin/bash

cd `dirname "$0"` || exit 1

SIZES='64 128 256 512 1024 2048'
FILES=`ls -1 ./data/ | grep '[.]hdf5'`
ITERATIONS=10
STEP=5

for FILE in $FILES; do
for SIZE in $SIZES; do
	for ((i=1; i <= $ITERATIONS; i+=$STEP)); do
		for ((s=0; s < $STEP; ++s)); do
			ITERATION=$(($i+$s))
			./run.sh $ITERATION "$FILE" $SIZE &
		done
		wait
	done
done
done
