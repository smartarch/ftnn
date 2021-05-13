import logging
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

from nn10 import TrainableNN10

config = {
    "nnArch": {
        "name": "nn10",
    },
    "batchSize": 128,
    "learningRate": 0.01,
    "maxEpochs": 50
}

net = TrainableNN10(config=TrainableNN10.createConfig(config))

net.buildFromConfig()
net.summary()

net.train(50)
#net.checkResults()

