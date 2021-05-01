import tensorflow as tf

def accessToWorkplaceDsGen(dsConfig, batchCount):
    def generator():
        for batchIdx in range(batchCount):
            allowedCountInBatch = dsConfig['allowedCountInBatch']
            deniedCountInBatch = dsConfig['deniedCountInBatch']
            genBatchSize = allowedCountInBatch + deniedCountInBatch

            allowedCount = 0
            deniedCount = 0

            data = tf.zeros((0, 9), dtype=tf.float32)

            while deniedCount < deniedCountInBatch or allowedCount < allowedCountInBatch:
                genSize = 10000
                gatePosByWpId = tf.constant([[178.64561, 98.856514], [237.03545, 68.872505], [237.0766, 135.65627]], dtype=tf.float32)

                time = tf.random.uniform(shape=(genSize,), minval=0, maxval=36000, dtype=tf.float32)
                posX = tf.random.uniform(shape=(genSize,), minval=0, maxval=316.43506, dtype=tf.float32)
                posY = tf.random.uniform(shape=(genSize,), minval=0, maxval=177.88289, dtype=tf.float32)
                pos = tf.stack([posX, posY], axis=1)

                wpId = tf.random.uniform(shape=(genSize,), minval=0, maxval=3, dtype=tf.int32)
                wpId_OneHot = tf.one_hot(wpId, depth=3, dtype=tf.float32)

                gatePos = tf.gather(gatePosByWpId, wpId, axis=0)
                gateDistSquaredElems = tf.math.square(pos - gatePos)
                gateDist = tf.math.sqrt(gateDistSquaredElems[:,0] + gateDistSquaredElems[:,1])

                hgerEvents_0 = tf.random.uniform(shape=(genSize,), minval=0, maxval=2, dtype=tf.int32)
                hgerEvents_0_OneHot = tf.one_hot(hgerEvents_0, depth=2, dtype=tf.float32)

                atGate = gateDist <= 10
                hasHeadGear = hgerEvents_0 == 1
                isRightTime = tf.logical_and(time >= 2400, time <= 33600)

                mbOutputs = tf.reduce_all([atGate, hasHeadGear, isRightTime], axis=0)
                mbDeniedAccesses = mbOutputs == False
                mbAllowedAccesses = mbOutputs == True

                mbData = tf.concat([tf.expand_dims(time, axis=1), pos, wpId_OneHot, hgerEvents_0_OneHot, tf.expand_dims(tf.cast(mbOutputs, dtype=tf.float32), axis=1)], axis=1)

                mbDeniedData = mbData[mbDeniedAccesses]
                mbAllowedData = mbData[mbAllowedAccesses]

                deniedCountToInclude = tf.minimum(mbDeniedData.shape[0], deniedCountInBatch-deniedCount)
                allowedCountToInclude = tf.minimum(mbAllowedData.shape[0], allowedCountInBatch - allowedCount)

                data = tf.concat([
                    data,
                    mbDeniedData[0:deniedCountToInclude],
                    mbAllowedData[0:allowedCountToInclude]
                ], axis=0)

                deniedCount += deniedCountToInclude
                allowedCount += allowedCountToInclude

            data = tf.random.shuffle(data)

            tf.assert_equal(data.shape[0], genBatchSize)
            tf.assert_equal(tf.reduce_sum(tf.cast(data[:, 8], dtype=tf.int32)), allowedCountInBatch)

            if allowedCountInBatch == 0:
                deniedWeight = tf.cast(1 / deniedCountInBatch, dtype=tf.float32)
                allowedWeight = tf.constant(0, dtype=tf.float32)
            elif deniedCountInBatch == 0:
                deniedWeight = tf.constant(0, dtype=tf.float32)
                allowedWeight = tf.cast(1 / allowedCountInBatch, dtype=tf.float32)
            else:
                deniedWeight = 0.5 * tf.cast(1 / deniedCountInBatch, dtype=tf.float32)
                allowedWeight = 0.5 * tf.cast(1 / allowedCountInBatch, dtype=tf.float32)

            inputs = data[:, 0:8]
            outputs = data[:, 8:9]

            weightByOutput = tf.stack([deniedWeight, allowedWeight], axis=0)
            weights = tf.expand_dims(tf.gather(weightByOutput, tf.cast(data[:, 8], dtype=tf.int32), axis=0), axis=1)

            print(f'inputs={inputs.shape} outputs={outputs.shape} weights={weights.shape}')

            yield (inputs, outputs, weights)

    return generator

