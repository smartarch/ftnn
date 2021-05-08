import numpy as np
import tensorflow as tf


def synthConjDsGen(dsConfig, batchCount):
    def generator():
        conjCount = len(dsConfig['inputKeys'])
        countInBatch = dsConfig['countInBatch']

        intervals = np.linspace(0, 1, conjCount * 2).reshape((-1, 2))

        insideIntervalLen = intervals[0, 1] - intervals[0, 0]
        outsideIntervalScale = 0.5 / (1 - insideIntervalLen)
        interpOnes = np.ones((conjCount,), dtype=np.float32)
        fp = np.stack([interpOnes * -1, intervals[:, 0], intervals[:, 1], interpOnes * 2], axis=1)
        xp = np.stack([interpOnes * (-outsideIntervalScale), fp[:, 1] * outsideIntervalScale, fp[:, 1] * outsideIntervalScale + 0.5, interpOnes * (1 + outsideIntervalScale)], axis=1)

        rng = np.random.default_rng()

        for batchIdx in range(batchCount):
            inputs = rng.uniform(0, 1, (countInBatch, conjCount))
            for conjIdx in range(conjCount):
                inputs[:, conjIdx] = np.interp(inputs[:, conjIdx], xp[conjIdx, :], fp[conjIdx, :])

            inputsInInterval = np.logical_and(inputs >= intervals[:, 0], inputs <= intervals[:, 1])
            outputs = np.all(inputsInInterval, axis=1).astype(np.float32)
            outputs = np.expand_dims(outputs, axis=1)
            weights = np.ones_like(outputs)

            yield (tf.constant(inputs, dtype=tf.float32), tf.constant(outputs, dtype=tf.float32), tf.constant(weights, dtype=tf.float32))

    return generator


