import logging
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

from nn10 import TrainableNN10

config = {
    "nnArch": {
        "name": "nn10",
    },
    "batchSize": 100,
    "learningRate": 0.001,
    "maxEpochs": 30
}

net = TrainableNN10(config=TrainableNN10.createConfig(config))

net.buildFromConfig()
net.summary()

net.train(30)
#net.checkResults()

