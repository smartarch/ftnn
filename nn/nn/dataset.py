import h5py
import tensorflow as tf

from nn import util
from nn.prep_synth import synthConjDsGen

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


def loadOnTheFlyMonteCarloSimDS(dsConfig, batchSize):
    trainGenBatchCount = dsConfig['trainGenBatchCount']
    valGenBatchCount = dsConfig['valGenBatchCount']
    batchGen = dsConfig['batchGen']

    allowedCountInBatch = dsConfig['allowedCountInBatch']
    deniedCountInBatch = dsConfig['deniedCountInBatch']
    genBatchSize = allowedCountInBatch + deniedCountInBatch

    outSig = (
        tf.TensorSpec(shape=(genBatchSize, len(dsConfig['inputKeys'])), dtype=tf.float32),
        tf.TensorSpec(shape=(genBatchSize, len(dsConfig['outputKeys'])), dtype=tf.float32),
        tf.TensorSpec(shape=(genBatchSize, 1), dtype=tf.float32)
    )

    trainDS = tf.data.Dataset.from_generator(batchGen(dsConfig, trainGenBatchCount), output_signature=outSig).unbatch().batch(batchSize).cache()
    valDS = tf.data.Dataset.from_generator(batchGen(dsConfig, valGenBatchCount), output_signature=outSig).unbatch().batch(batchSize).cache()

    return (trainDS, valDS)


def loadSynthConjSimDS(dsConfig, batchSize):
    trainGenBatchCount = dsConfig['trainGenBatchCount']
    valGenBatchCount = dsConfig['valGenBatchCount']

    allowedCountInBatch = dsConfig['allowedCountInBatch']
    deniedCountInBatch = dsConfig['deniedCountInBatch']
    genBatchSize = allowedCountInBatch + deniedCountInBatch

    outSig = (
        tf.TensorSpec(shape=(genBatchSize, len(dsConfig['inputKeys'])), dtype=tf.float32),
        tf.TensorSpec(shape=(genBatchSize, len(dsConfig['outputKeys'])), dtype=tf.float32),
        tf.TensorSpec(shape=(genBatchSize, 1), dtype=tf.float32)
    )

    trainDS = tf.data.Dataset.from_generator(synthConjDsGen(dsConfig, trainGenBatchCount), output_signature=outSig).unbatch().batch(batchSize).cache()
    valDS = tf.data.Dataset.from_generator(synthConjDsGen(dsConfig, valGenBatchCount), output_signature=outSig).unbatch().batch(batchSize).cache()

    return (trainDS, valDS)


def addUnitWeights(inputs, outputs):
    return (inputs, outputs, tf.ones_like(outputs))

def loadBalancedSimDS(dsConfig, batchSize):
    dsName = dsConfig['name']
    dsHash = dsConfig.toHash()

    if dsHash in cachedDS:
        (trainDS, valDS) = cachedDS[dsHash]
    else:
        with h5py.File(baseDataDir / 'dataset' / f'{dsName}-{dsHash}.hdf5', 'r') as hf:
            trainDS = tf.data.Dataset.from_tensor_slices((hf['train-inputs'], hf['train-outputs'])).cache()
            valDS = tf.data.Dataset.from_tensor_slices((hf['val-inputs'], hf['val-outputs'])).cache()

            cachedDS[dsHash] = (trainDS, valDS)

    trainDS = trainDS.batch(batchSize).map(addUnitWeights, num_parallel_calls=2).prefetch(500).unbatch().batch(batchSize).shuffle(buffer_size=500)
    valDS = valDS.batch(batchSize).map(addUnitWeights, num_parallel_calls=2).prefetch(500)

    return (trainDS, valDS)


def loadDS(dsConfig, batchSize):
    if dsConfig.name == 'JSONSimDataSetConfig':
        return loadJSONSimDS(dsConfig, batchSize)
    elif dsConfig.name == 'OnTheFlyMonteCarloSimDataSetConfig':
        return loadOnTheFlyMonteCarloSimDS(dsConfig, batchSize)
    elif dsConfig.name == 'SynthConjSimDataSetConfig':
        return loadSynthConjSimDS(dsConfig, batchSize)
    elif dsConfig.name == 'BalancedSimDataSetConfig':
        return loadBalancedSimDS(dsConfig, batchSize)
    else:
        assert False