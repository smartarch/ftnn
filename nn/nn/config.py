from pathlib import Path
import hashlib
import json

class ConfigLayer(dict):
    def __init__(self, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            self[key] = value

    def __setitem__(self, key, value):
        cls = type(self)
        if key in cls.fields:
            super().__setitem__(key, value)
        else:
            raise Exception(f'Field "{key}" not permitted for {cls.name}')

    def toJson(self):
        return json.dumps(self, sort_keys=True, indent=2)

    def toHash(self):
        return hashlib.md5(self.toJson().encode('utf-8')).hexdigest()


class JSONSimDataSetConfig(ConfigLayer):
    name = 'JSONSimDataSetConfig'
    fields = {'name', 'tracesName', 'trainSimulationCount', 'valSimulationCount', 'inputKeys', 'outputKeys', 'outputSamplesPerSimulation', 'allowedOnly'}

class OnTheFlyMonteCarloSimDataSetConfig(ConfigLayer):
    name = 'OnTheFlyMonteCarloSimDataSetConfig'
    fields = {'name', 'trainGenBatchCount', 'valGenBatchCount', 'allowedCountInBatch', 'deniedCountInBatch', 'inputKeys', 'outputKeys', 'batchGen'}

class SynthConjSimDataSetConfig(ConfigLayer):
    name = 'SynthConjSimDataSetConfig'
    fields = {'name', 'trainGenBatchCount', 'valGenBatchCount', 'countInBatch', 'inputKeys', 'outputKeys'}

class BalancedSimDataSetConfig(ConfigLayer):
    name = 'BalancedSimDataSetConfig'
    fields = {'name', 'trainCount', 'valCount', 'inputKeys', 'outputKeys'}

class TrainingConfig(ConfigLayer):
    name = 'TrainingConfig'
    fields = {'dsConfig', 'batchSize', 'learningRate', 'nnArch'}
