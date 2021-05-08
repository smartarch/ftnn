import tensorflow as tf
import numpy as np


def accessToWorkplaceDsGen(dsConfig, batchCount):
    def generator():
        for batchIdx in range(batchCount):
            allowedCountInBatch = dsConfig['allowedCountInBatch']
            deniedCountInBatch = dsConfig['deniedCountInBatch']

            allowedCount = 0
            deniedCount = 0

            data = np.zeros((allowedCountInBatch + deniedCountInBatch, 10), dtype=np.float32)
            dataIdx = 0

            resultCategoryCount = 8

            if allowedCountInBatch == 0:
                deniedWeight = 1 / deniedCountInBatch
                allowedWeight = 0
            elif deniedCountInBatch == 0:
                deniedWeight = 0
                allowedWeight = allowedCountInBatch
            else:
                deniedWeight = 0.5 / deniedCountInBatch
                allowedWeight = 0.5 / allowedCountInBatch

            while dataIdx < deniedCountInBatch + allowedCountInBatch:
                genSize = 100000
                gatePosByWpId = np.array([[178.64561, 98.856514], [237.03545, 68.872505], [237.0766, 135.65627]],
                                         dtype=np.float32)

                rng = np.random.default_rng()

                wpIdToOneHot = np.eye(3)
                hgerEventToOneHot = np.eye(2)
                catToOneHot = np.eye(resultCategoryCount)

                time = rng.uniform(0, 36000, genSize)
                posX = rng.uniform(0, 316.43506, genSize)
                posY = rng.uniform(0, 177.88289, genSize)

                wpId = rng.integers(0, 3, genSize)
                wpId_OneHot = wpIdToOneHot[wpId, :]

                gatePos = gatePosByWpId[wpId, :]
                gateDist = np.sqrt(np.square(posX - gatePos[:, 0]) + np.square(posY - gatePos[:, 1]))

                hgerEvents_0 = rng.integers(0, 2, genSize)
                hgerEvents_0_OneHot = hgerEventToOneHot[hgerEvents_0, :]

                isRightTime = np.logical_and(time >= 2400, time <= 33600)
                atGate = gateDist <= 10
                hasHeadGear = hgerEvents_0 == 1

                cat = isRightTime * 1 + atGate * 2 + hasHeadGear * 4
                catCounts = np.sum(catToOneHot[cat, :], axis=0).astype(np.int32)

                mbInputs = np.concatenate([np.stack([time, posX, posY], axis=1), wpId_OneHot, hgerEvents_0_OneHot],
                                          axis=1)

                allowedCountToInclude = np.minimum(catCounts[resultCategoryCount - 1],
                                                   allowedCountInBatch - allowedCount)
                mbInputsAllowed = mbInputs[cat == resultCategoryCount - 1]
                data[dataIdx:dataIdx + allowedCountToInclude, 0:8] = mbInputsAllowed[:allowedCountToInclude, :]
                data[dataIdx:dataIdx + allowedCountToInclude, 8] = 1
                data[dataIdx:dataIdx + allowedCountToInclude, 9] = allowedWeight

                dataIdx += allowedCountToInclude
                allowedCount += allowedCountToInclude

                deniedCountToIncludePerCategory = np.minimum(np.min(catCounts[0:resultCategoryCount - 1]),
                                                             np.ceil(deniedCountInBatch - deniedCount / 7).astype(
                                                                 np.int32))
                for catIdx in range(resultCategoryCount - 1):
                    deniedCountToIncludePerThisCategory = deniedCountToIncludePerCategory
                    if deniedCount + deniedCountToIncludePerThisCategory > deniedCountInBatch:
                        deniedCountToIncludePerThisCategory = deniedCountInBatch - deniedCount

                    mbInputsDeniedInCategory = mbInputs[cat == catIdx]
                    data[dataIdx:dataIdx + deniedCountToIncludePerThisCategory, 0:8] = mbInputsDeniedInCategory[
                                                                                       :deniedCountToIncludePerThisCategory]
                    data[dataIdx:dataIdx + deniedCountToIncludePerThisCategory, 8] = 0
                    data[dataIdx:dataIdx + allowedCountToInclude, 9] = deniedWeight

                    dataIdx += deniedCountToIncludePerThisCategory
                    deniedCount += deniedCountToIncludePerThisCategory

            rng.shuffle(data, axis=0)

            yield (tf.constant(data[:, 0:8], dtype=tf.float32), tf.constant(data[:, 8:9], dtype=tf.float32), tf.constant(data[:, 9:10], dtype=tf.float32))

    return generator


