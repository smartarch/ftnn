import h5py
import tensorflow as tf

from nn import util

baseDataDir = util.getBaseDataDir()

cachedDS = {}


def loadJSONSimDS(dsConfig, batchSize):
    dsName = dsConfig['name']
    dsHash = dsConfig.toHash()

    if dsHash in cachedDS:
        (trainDS, valDS) = cachedDS[dsHash]
    else:
        with h5py.File(baseDataDir / 'dataset' / f'{dsName}-{dsHash}.hdf5', 'r') as hf:
            trainDS = tf.data.Dataset.from_tensor_slices((hf['train-inputs'], hf['train-outputs'], hf['train-weights'])).cache()
            valDS = tf.data.Dataset.from_tensor_slices((hf['val-inputs'], hf['val-outputs'], hf['val-weights'])).cache()

            cachedDS[dsHash] = (trainDS, valDS)

    trainDS = trainDS.batch(batchSize).prefetch(500).shuffle(buffer_size=500)
    valDS = valDS.batch(batchSize).prefetch(500)

    return (trainDS, valDS)



def loadOnTheFlySimDS(dsConfig, batchSize):
    trainGenBatchCount = dsConfig['trainGenBatchCount']
    valGenBatchCount = dsConfig['valGenBatchCount']
    batchGen = dsConfig['batchGen']

    trainDS = tf.data.Dataset.range(trainGenBatchCount).map(batchGen, num_parallel_calls=8).unbatch().batch(batchSize).cache()
    valDS = tf.data.Dataset.range(valGenBatchCount).map(batchGen, num_parallel_calls=8).unbatch().batch(batchSize)

    return (trainDS, valDS)


def loadDS(dsConfig, batchSize):
    if dsConfig.name == 'JSONSimDataSetConfig':
        return loadJSONSimDS(dsConfig, batchSize)
    elif dsConfig.name == 'OnTheFlySimDataSetConfig':
        return loadOnTheFlySimDS(dsConfig, batchSize)
    else:
        assert False