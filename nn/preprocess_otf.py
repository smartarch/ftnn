import json
import logging
import sys

import h5py

from nn import util
from nn.config import OnTheFlyMonteCarloSimDataSetConfig, BalancedSimDataSetConfig
from nn.dataset import loadDS
from nn10 import accessToWorkplaceDsGen

import tensorflow as tf

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.DEBUG)

factor = int(sys.argv[1])

dsConfigOTF = OnTheFlyMonteCarloSimDataSetConfig(
    name='ftnn-otf',
    trainGenBatchCount=9,
    valGenBatchCount=1,
    allowedCountInBatch=24 * factor,
    deniedCountInBatch=24 * factor,
    inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
    outputKeys=['accessToWorkplace'],
    batchGen = accessToWorkplaceDsGen
)

dsConfig = BalancedSimDataSetConfig(
    name='ftnn-otf',
    trainCount=9 * 48 * factor,
    valCount=1 * 48 * factor,
    inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
    outputKeys=['accessToWorkplace']
)

if factor < 1000:
    batchSize = 24 * factor
else:
    batchSize = 24000


trainDs, valDs = loadDS(dsConfigOTF, batchSize)

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
