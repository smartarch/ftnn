import json
import logging
import sys

import h5py

from nn import util
from nn.config import OnTheFlyMonteCarloSimDataSetConfig, BalancedSimDataSetConfig, SynthConjSimDataSetConfig
from nn.dataset import loadDS
from nn10 import accessToWorkplaceDsGen

import tensorflow as tf

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.DEBUG)

factor = int(sys.argv[1])

conjCount = 10
deniedCatCount = 2 ** conjCount - 1

dsConfigSynthConj = SynthConjSimDataSetConfig(
    name='ftnn-synth-conj',
    trainGenBatchCount=9,
    valGenBatchCount=1,
    allowedCountInBatch=deniedCatCount * factor,
    deniedCountInBatch=deniedCatCount * factor,
    inputKeys=[f'y{idx}' for idx in range(conjCount)],
    outputKeys=['access'],
)

dsConfig = BalancedSimDataSetConfig(
    name='ftnn-synth-conj',
    trainCount=9 * deniedCatCount * 2 * factor,
    valCount=1 * deniedCatCount * 2 * factor,
    inputKeys=[f'y{idx}' for idx in range(conjCount)],
    outputKeys=['access']
)

if factor < 1000:
    batchSize = 100 * factor
else:
    batchSize = 100000


trainDs, valDs = loadDS(dsConfigSynthConj, batchSize)

baseDataDir = util.getBaseDataDir()
dsHash = dsConfig.toHash()
dsName = dsConfig['name']
datasetDir = baseDataDir / 'dataset'

with open(datasetDir / f'{dsName}-{dsHash}.json', 'wt') as metadataFile:
    metadata = {
        'dataSetConfig': dsConfig
    }
    json.dump(metadata, metadataFile, indent=4)


with h5py.File(datasetDir / f'{dsName}-{dsHash}.hdf5', 'w') as hf:
    def createDS(prefix, ds, totalSamplesCount):
        compression = None
        # compression = 'gzip'

        inputs = hf.create_dataset(f'{prefix}-inputs', (totalSamplesCount, len(dsConfig['inputKeys'])), dtype='float32', compression=compression)
        outputs = hf.create_dataset(f'{prefix}-outputs', (totalSamplesCount, len(dsConfig['outputKeys'])), dtype='float32', compression=compression)

        startIdx = 0
        totalAllowed = 0
        for inputBatch, outputBatch, weightsBatch in ds:
            logging.info(f'Processing {prefix}:{startIdx}')
            totalAllowed += tf.reduce_sum(outputBatch[:, 0]).numpy()
            inputs[startIdx:startIdx + batchSize] = inputBatch
            outputs[startIdx:startIdx + batchSize] = outputBatch
            startIdx += batchSize

        logging.info(f'Finished processing {prefix} count={startIdx} #allowed={totalAllowed} #denied={startIdx-totalAllowed}')

    createDS('train', trainDs, dsConfig['trainCount'])
    createDS('val', valDs, dsConfig['valCount'])
