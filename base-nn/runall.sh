#!/bin/bash

cd `dirname "$0"` || exit 1

SIZES='128 256 512 1024 2048'
FILES=`ls -1 ./data/ | grep '[.]hdf5'`
ITERATIONS=5
STEP=1

#for ((s=0; s < $STEP; ++s)); do
#	ITERATION=$(($i+$s))

for ((i=1; i <= $ITERATIONS; i+=$STEP)); do
	ITERATION=$i
	for FILE in $FILES; do
		for SIZE in $SIZES; do
			./run.sh $ITERATION "$FILE" $SIZE &
		done
		wait
		for SIZE in $SIZES; do
			./run.sh $ITERATION "$FILE" $SIZE $SIZE &
		done
		wait
	done
done
