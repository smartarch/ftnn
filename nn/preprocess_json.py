import logging
import math
import sys
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum

import numpy as np

from nn.config import JSONSimDataSetConfig
from nn.prep import preprocessJSONSimDataset

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.DEBUG)


def parseTS(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


class EventType(Enum):
    TAKE_HGEAR = 0
    RET_HGEAR = 1


class Position():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, pos):
        return math.sqrt((self.x - pos.x) ** 2 + (self.y - pos.y) ** 2)

    def __repr__(self):
        return f'Position({self.x},{self.y})'


gateAccessPosDistanceThreshold = 5
gateAccessTsDistanceThreshold = 300

wpIdToOneHot = {
    'A': np.array([1, 0, 0], dtype=np.float32),
    'B': np.array([0, 1, 0], dtype=np.float32),
    'C': np.array([0, 0, 1], dtype=np.float32)
}


gatePositions = {
    'A': Position(178.64561, 98.856514),
    'B': Position(237.03545, 68.872505),
    'C': Position(237.0766, 135.65627)
}

startTime = parseTS('2018-12-03T08:00:00Z')

allowedOnly = True
outputSamplesPerSimulation = 24 if allowedOnly else 10800

def transformSimulation(samples, inputKeysMap, outputKeysMap):
    ikm = inputKeysMap
    okm = outputKeysMap

    hasHeadGearByWorkerId = defaultdict(bool)
    eventsByWorkerId = defaultdict(deque)
    lastGateAccessTsByWorkerId = defaultdict(float)

    inputs = np.zeros((outputSamplesPerSimulation, len(inputKeysMap)), dtype=np.float32)
    outputs = np.zeros((outputSamplesPerSimulation, len(outputKeysMap)), dtype=np.float32)
    weights = np.zeros((outputSamplesPerSimulation, 1), dtype=np.float32)

    idx = 0
    for sample in samples:
        ts = (parseTS(sample['time']) - startTime).total_seconds()

        for worker in sample['workers']:
            wId = worker['id']
            wpId = worker['wpId']
            wPos = Position(**worker['position'])
            events = eventsByWorkerId[wId]
            hasHeadGear = worker['hasHeadGear']

            if not hasHeadGearByWorkerId[wId] and hasHeadGear:
                hasHeadGearByWorkerId[wId] = True
                events.appendleft(EventType.TAKE_HGEAR)
            elif hasHeadGearByWorkerId[wId] and not hasHeadGear:
                hasHeadGearByWorkerId[wId] = False
                events.appendleft(EventType.RET_HGEAR)

            hgerEvents = [ev for ev in events if ev == EventType.TAKE_HGEAR or ev == EventType.RET_HGEAR]

            isAccessRequested = gatePositions[wpId].distance(wPos) <= gateAccessPosDistanceThreshold and ts - \
                                lastGateAccessTsByWorkerId[wId] >= gateAccessTsDistanceThreshold
            if isAccessRequested:
                lastGateAccessTsByWorkerId[wId] = ts

            isAllowed = isAccessRequested and (
                    gatePositions[wpId].distance(wPos) < 10 and  # atGate
                    len(hgerEvents) > 0 and hgerEvents[0] == EventType.TAKE_HGEAR and  # hasHeadGear
                    ts >= 40 * 60 and ts <= 9 * 3600 + 20 * 60  # now between shift start and end
            )

            if isAllowed or not allowedOnly:
                inputs[idx, ikm['time']] = ts
                inputs[idx, ikm['posX']] = wPos.x
                inputs[idx, ikm['posY']] = wPos.y

                inputs[idx, ikm['wpId:A']:ikm['wpId:C']+1] = wpIdToOneHot[wpId]

                for evIdx, ev in enumerate(hgerEvents[0:1]):
                    for evType in EventType:
                        inputs[idx, ikm[f'hgerEvents[{evIdx}]:{evType.name}']] = evType == ev

                outputs[idx, okm['accessToWorkplace']] = isAllowed

                idx += 1

    assert idx == outputSamplesPerSimulation

    deniedAccesses = outputs[:,okm['accessToWorkplace']] == 0
    allowedAccesses = outputs[:,okm['accessToWorkplace']] == 1

    deniedAccessCount = np.sum(deniedAccesses)
    allowedAccessCount = np.sum(allowedAccesses)

    weights[deniedAccesses, 0] = 1 / deniedAccessCount if deniedAccessCount > 0 else 0
    weights[allowedAccesses, 0] = 1 / allowedAccessCount if allowedAccessCount > 0 else 0

    assert allowedAccessCount == 24

    return inputs, outputs, weights


if __name__ == '__main__':
    trainSimulationCount = int(sys.argv[1])
    valSimulationCount = int(sys.argv[2])
    processCount = int(sys.argv[3])

    dsConfig = JSONSimDataSetConfig(
        name='ftnn-traces-v1',
        tracesName='ftnn-traces-v1',
        trainSimulationCount=trainSimulationCount,
        valSimulationCount=valSimulationCount,
        inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
        outputKeys=['accessToWorkplace'],
        outputSamplesPerSimulation=outputSamplesPerSimulation,
        allowedOnly=allowedOnly
    )

    preprocessJSONSimDataset(dsConfig=dsConfig, transformSimulation=transformSimulation, processCount=processCount)

