import csv
import json
import sys
from json import JSONDecodeError
from pathlib import Path

import numpy as np
import pandas as pd

resultsName = sys.argv[1]
outputName = sys.argv[2]

if len(sys.argv) == 4:
    experimentMask = sys.argv[3]
else:
    experimentMask = '*'

resultsDirPath = Path('/mnt/data/results') / resultsName
analyticsDirPath = Path('/mnt/data/analytics')

analyticsDirPath.mkdir(parents=True, exist_ok=True)

excludedPrefixes = ['config.dsConfig.files']

def flatten(data, prefix, output=None):
    if output is None:
        output = {}

    if isinstance(data, dict):
        for key, val in data.items():
            flatten(val, f'{prefix}.{key}', output)
    elif isinstance(data, list):
        for idx, val in enumerate(data):
            flatten(val, f'{prefix}[{idx}]', output)
    else:
        if not any((prefix.startswith(excludedPrefix) for excludedPrefix in excludedPrefixes)):
            output[prefix] = data

    return output

trialRows = []

for experimentDirPath in resultsDirPath.glob(experimentMask):
    if experimentDirPath.is_dir():
        for trialDirPath in experimentDirPath.iterdir():
            if trialDirPath.is_dir():
                print(f'Processing trial in directory {trialDirPath}')
                with open(trialDirPath / 'trial.json') as trialFile:
                    try:
                        params = json.load(trialFile)

                        rowBase = {}
                        hp = flatten(params['config'], 'config')
                        for key, val in hp.items():
                            rowBase[key] = val

                        for resultFilePath in trialDirPath.glob('results-*.json'):
                            print(f'Reading {resultFilePath}')
                            with open(resultFilePath) as resultFile:
                                try:
                                    result = json.load(resultFile)
                                    row = rowBase | result

                                    trialRows.append(row)
                                except (UnicodeDecodeError, JSONDecodeError):
                                    print(f'Cannot process file {resultFile}')

                    except (UnicodeDecodeError, JSONDecodeError):
                        print(f'Cannot process file {trialFile}')



trials = pd.DataFrame(trialRows).set_index(['experimentId', 'trialId', 'repetition']).sort_index()

aggStats = {}
for col, colLabel in [('trainLoss', 'TL'), ('trainAccuracy', 'TA'), ('valLoss', 'VL'), ('valAccuracy', 'VA')]:
    for aggLabel, aggFn in [('min', np.min), ('max', np.max), ('avg', np.average), ('med', np.median), ('stddev', np.std)]:
        aggName = f'{aggLabel}{colLabel}'
        aggStats[aggName] = pd.NamedAgg(column=col, aggfunc=aggFn)

bestCols = []
for col in trials.columns:
    if col.startswith('config'):
        bestCols.append(col)

tgh = trials.groupby(['experimentId', 'trialId'])

def bestEntry(df):
    return df\
        .reset_index(level=['experimentId', 'trialId'], drop=True) \
        .sort_values(by='valAccuracy', ascending=False)\
        [bestCols]\
        .head(1)

aggs = pd.merge(tgh.apply(bestEntry), tgh.agg(**aggStats), left_index=True, right_index=True).sort_values(by='avgVA', ascending=False)
aggs = aggs.reset_index(level='repetition').rename(columns={'repetition': 'bestRepetition'})

with pd.ExcelWriter(analyticsDirPath / f'{outputName}.xlsx') as writer:
    trials.to_excel(writer, sheet_name="All", merge_cells=False)
    aggs.to_excel(writer, sheet_name="Aggs", merge_cells=False)