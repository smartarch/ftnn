import logging
import math
import sys
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum

import numpy as np

from nn.config import JSONSimDataSetConfig, OnTheFlySimDataSetConfig
from nn.dataset import loadDS
from nn.prep import preprocessJSONSimDataset
from nn10 import timeOTFDsGen

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.DEBUG)

dsConfigJSONSim_900000_100000 = JSONSimDataSetConfig(
    name='ftnn-traces-v1',
    tracesName='ftnn-traces-v1',
    trainSimulationCount=900000,
    valSimulationCount=100000,
    inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
    outputKeys=['accessToWorkplace'],
    outputSamplesPerSimulation=24,
    allowedOnly = True
)

dsConfigOTF_900000_100000 = OnTheFlySimDataSetConfig(
    name='ftnn-otf-1',
    trainGenBatchCount=900,
    valGenBatchCount=100,
    genBatchSize=10000,
    inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
    outputKeys=['accessToWorkplace'],
    deniedOnly=True
)
dsConfigOTF_900000_100000['batchGen'] = lambda genBatchIdx: timeOTFDsGen(dsConfigOTF_900000_100000)


allowedTrain, allowedVal = loadDS(dsConfigJSONSim_900000_100000)
deniedTrain, deniedVal = loadDS(dsConfigOTF_900000_100000)
