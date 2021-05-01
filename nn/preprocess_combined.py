import json
import logging
import sys

import h5py
import tensorflow as tf

from nn import util
from nn.config import JSONSimDataSetConfig, OnTheFlyMonteCarloSimDataSetConfig, BalancedSimDataSetConfig
from nn.dataset import loadDS
from nn.prep_otf import accessToWorkplaceDsGen

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.DEBUG)

factor = int(sys.argv[1])

dsConfigJSONSim = JSONSimDataSetConfig(
    name='ftnn-traces-v1',
    tracesName='ftnn-traces-v1',
    trainSimulationCount=9 * factor,
    valSimulationCount=1 * factor,
    inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
    outputKeys=['accessToWorkplace'],
    outputSamplesPerSimulation=24,
    allowedOnly = True
)

dsConfigOTF = OnTheFlyMonteCarloSimDataSetConfig(
    name='ftnn-otf-1',
    trainGenBatchCount=9,
    valGenBatchCount=1,
    allowedCountInBatch=0,
    deniedCountInBatch=24 * factor,
    inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
    outputKeys=['accessToWorkplace'],
    batchGen = accessToWorkplaceDsGen
)

dsConfig = BalancedSimDataSetConfig(
    name='ftnn-combined',
    trainCount=9 * 48 * factor,
    valCount=1 * 48 * factor,
    inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
    outputKeys=['accessToWorkplace']
)


def interleave(allowed, denied):
    aInputs, aOutputs, aWeights = allowed
    dInputs, dOutputs, dWeights = denied

    tf.assert_equal(tf.reduce_all(aOutputs[:, 0] == 1), tf.constant(True))
    tf.assert_equal(tf.reduce_all(dOutputs[:, 0] == 0), tf.constant(True))

    inputs = tf.concat([aInputs, dInputs], axis=1)
    inputs = tf.reshape(inputs, (-1, 8))
    outputs = tf.concat([aOutputs, dOutputs], axis=1)
    outputs = tf.reshape(outputs, (-1, 1))

    return (inputs, outputs)


if factor < 1000:
    batchSize = 24 * factor
else:
    batchSize = 24000


allowedTrainDs, allowedValDs = loadDS(dsConfigJSONSim, batchSize)
deniedTrainDs, deniedValDs = loadDS(dsConfigOTF, batchSize)

trainDs = tf.data.Dataset.zip((allowedTrainDs, deniedTrainDs)).map(interleave).unbatch()
valDs = tf.data.Dataset.zip((allowedValDs, deniedValDs)).map(interleave).unbatch()


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
        for inputBatch, outputBatch in ds.batch(batchSize):
            logging.info(f'Processing {prefix}:{startIdx}')
            totalAllowed += tf.reduce_sum(outputBatch[:, 0]).numpy()
            inputs[startIdx:startIdx + batchSize] = inputBatch
            outputs[startIdx:startIdx + batchSize] = outputBatch
            startIdx += batchSize

        logging.info(f'Finished processing {prefix} count={startIdx} #allowed={totalAllowed} #denied={startIdx-totalAllowed}')

    createDS('train', trainDs, dsConfig['trainCount'])
    createDS('val', valDs, dsConfig['valCount'])
