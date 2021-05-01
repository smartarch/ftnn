import gzip
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from nn import util
from nn.util import getKeysMap

baseDataDir = util.getBaseDataDir()


def preprocessDataset_processPath(taskSpec):
    inPath, transformSimulation, inputKeysMap, outputKeysMap = taskSpec
    samples = []
    with gzip.open(inPath, 'rt') as file:
        for line in file:
            samples.append(json.loads(line))

    return transformSimulation(samples, inputKeysMap, outputKeysMap)


def preprocessJSONSimDataset(dsConfig, transformSimulation, processCount):
    inputKeys = dsConfig['inputKeys']
    outputKeys = dsConfig['outputKeys']
    inputKeysMap = getKeysMap(inputKeys)
    outputKeysMap = getKeysMap(outputKeys)
    outputSamplesPerSimulation = dsConfig['outputSamplesPerSimulation']

    outDir = baseDataDir / 'dataset'
    outDir.mkdir(parents=True, exist_ok=True)

    tracesDir = baseDataDir / 'traces' / dsConfig['tracesName']
    allFiles = sorted(tracesDir.rglob('*/*.jsonl.gz'))

    dsHash = dsConfig.toHash()
    dsName = dsConfig['name']
    datasetDir = baseDataDir / 'dataset'

    with open(datasetDir / f'{dsName}-{dsHash}.json', 'wt') as metadataFile:
        metadata = {
            'dataSetConfig': dsConfig
        }
        json.dump(metadata, metadataFile, indent=4)

    with h5py.File(datasetDir / f'{dsName}-{dsHash}.hdf5', 'w') as hf:
        def createDS(prefix, inPaths):
            inPathsChunked = [inPaths[idx:idx + processCount] for idx in range(0, len(inPaths), processCount)]

            totalSamplesCount = len(inPaths) * outputSamplesPerSimulation

            compression = None
            #compression = 'gzip'

            inputs = hf.create_dataset(f'{prefix}-inputs', (totalSamplesCount, len(inputKeysMap)), dtype=np.float32, compression=compression)
            outputs = hf.create_dataset(f'{prefix}-outputs', (totalSamplesCount, len(outputKeysMap)), dtype=np.float32, compression=compression)
            weights = hf.create_dataset(f'{prefix}-weights', (totalSamplesCount, 1), dtype=np.float32, compression=compression)

            startIdx = 0
            for inPathsChunk in inPathsChunked:
                logging.info(f'Processing {inPathsChunk}')

                if processCount == 1:
                    results = [preprocessDataset_processPath((inPath, transformSimulation, inputKeysMap, outputKeysMap)) for inPath in inPathsChunk]
                else:
                    with Pool(processes=processCount) as pool:
                        results = pool.map(preprocessDataset_processPath, [(inPath, transformSimulation, inputKeysMap, outputKeysMap) for inPath in inPathsChunk], chunksize=1)

                for simInputs, simOutputs, simWeights in results:
                    inputs[startIdx:startIdx+outputSamplesPerSimulation] = simInputs
                    outputs[startIdx:startIdx + outputSamplesPerSimulation] = simOutputs
                    weights[startIdx:startIdx + outputSamplesPerSimulation] = simWeights
                    startIdx += outputSamplesPerSimulation

        createDS('train', allFiles[:dsConfig['trainSimulationCount']])
        createDS('val', allFiles[dsConfig['trainSimulationCount']:dsConfig['trainSimulationCount'] + dsConfig['valSimulationCount']])
