#!/bin/bash

cd `dirname "$0"` || exit 1

SIZES='128 256 512 1024 2048'
FILES=`ls -1 ./data/ | grep '[.]hdf5'`
ITERATIONS=5

mkdir -p ./results-aggregated

for FILE in $FILES; do
	DEST="./results-aggregated/$FILE.csv"
	rm -f "$DEST"
	for SIZE in $SIZES; do
		for ((i=1; i <= $ITERATIONS; i++)); do
			RESULTFILE="./results/$FILE-dense-${SIZE}_$i.csv"
			if [ -f "$RESULTFILE" ]; then
				echo -n "$i;" >> "$DEST"
				tail -1 "$RESULTFILE" >> "$DEST"
			else
				echo "File $RESULTFILE does not exist..."
			fi
		done
	done
done

for FILE in $FILES; do
	DEST="./results-aggregated/$FILE.csv"
	for SIZE in $SIZES; do
		for ((i=1; i <= $ITERATIONS; i++)); do
			RESULTFILE="./results/$FILE-dense-${SIZE}-${SIZE}_$i.csv"
			if [ -f "$RESULTFILE" ]; then
				echo -n "$i;" >> "$DEST"
				tail -1 "$RESULTFILE" >> "$DEST"
			else
				echo "File $RESULTFILE does not exist..."
			fi
		done
	done
done
