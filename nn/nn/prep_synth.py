import numpy as np
import tensorflow as tf


def synthConjDsGen(dsConfig, batchCount):
    def generator():
        conjCount = len(dsConfig['inputKeys'])
        allowedCountInBatch = dsConfig['allowedCountInBatch']
        deniedCountInBatch = dsConfig['deniedCountInBatch']

        deniedCatCount = 2 ** conjCount - 1

        assert deniedCountInBatch % deniedCatCount == 0
        countInDeniedCat = deniedCountInBatch // deniedCatCount

        intervals = np.linspace(0, 1, conjCount * 2).reshape((conjCount, 2))

        rng = np.random.default_rng()

        for batchIdx in range(batchCount):
            data = np.zeros(shape=(allowedCountInBatch+deniedCountInBatch, conjCount + 2), dtype=np.float32)
            dataIdx = 0

            def createAllowedData(conjIdx):
                nonlocal dataIdx

                if conjIdx < conjCount:
                    data[dataIdx:dataIdx+allowedCountInBatch, conjIdx] = rng.uniform(intervals[conjIdx][0], intervals[conjIdx][1], (allowedCountInBatch,))
                    createAllowedData(conjIdx + 1)

                if conjIdx == 0:
                    dataIdx += allowedCountInBatch
                    data[dataIdx:dataIdx+allowedCountInBatch, conjCount] = 1
                    data[dataIdx:dataIdx+allowedCountInBatch, conjCount + 1] = 1


            def createDeniedData(catIdx, conjIdx):
                nonlocal dataIdx

                if conjIdx < conjCount:
                    shouldSucceed = bool((catIdx >> conjIdx) & 1)
                    if shouldSucceed:
                        data[dataIdx:dataIdx+countInDeniedCat, conjIdx] = rng.uniform(intervals[conjIdx][0], intervals[conjIdx][1], (countInDeniedCat,))
                    else:
                        intSize = intervals[conjIdx][1] - intervals[conjIdx][0]
                        vals = rng.uniform(0, 1 - intSize, (countInDeniedCat,))
                        data[dataIdx:dataIdx + countInDeniedCat, conjIdx] = vals + (vals > intervals[conjIdx][0]) * intSize

                    createDeniedData(catIdx, conjIdx + 1)

                if conjIdx == 0:
                    dataIdx += countInDeniedCat
                    data[dataIdx:dataIdx+countInDeniedCat, conjCount] = 1
                    data[dataIdx:dataIdx+countInDeniedCat, conjCount + 1] = 1

            createAllowedData(0)

            for catIdx in range(deniedCatCount):
                createDeniedData(catIdx, 0)

            rng.shuffle(data, axis=0)
            inputs = data[:,0:conjCount]
            outputs = data[:, conjCount:conjCount+1]
            weights = data[:, conjCount+1:conjCount+2]

            yield (tf.constant(inputs, dtype=tf.float32), tf.constant(outputs, dtype=tf.float32), tf.constant(weights, dtype=tf.float32))

    return generator


