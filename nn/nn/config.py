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

class OnTheFlySimDataSetConfig(ConfigLayer):
    name = 'OnTheFlySimDataSetConfig'
    fields = {'name', 'trainGenBatchCount', 'valGenBatchCount', 'genBatchSize', 'inputKeys', 'outputKeys', 'batchGen', 'deniedOnly'}

class TrainingConfig(ConfigLayer):
    name = 'TrainingConfig'
    fields = {'dsConfig', 'batchSize', 'learningRate', 'nnArch'}
