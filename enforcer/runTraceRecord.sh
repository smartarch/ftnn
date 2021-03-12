#!/bin/bash

#firstTraceFile=`python3 -c 'import sys; sId=int(sys.argv[1]); print("data/ftnn-traces-v1/{:04d}/{:03d}".format(sId//10, sId%10*100))' $1`.jsonl.gz
firstTraceFile=`python3 -c 'import sys; sId=int(sys.argv[1]); print("data/ftnn-traces-v1/{:04d}/000".format(sId))' $1`.jsonl.gz
echo Checking file $firstTraceFile

while [ ! -f $firstTraceFile ]; do
  env JAVA_OPTS="-Xmx20g" sbt "runMain ftnn.TraceRecord ${1} 1000"
done