from pathlib import Path


def getKeysMap(keys):
    return {
        key: idx for idx, key in enumerate(keys)
    }


def getBaseDataDir():
    baseDataDir = Path('/mnt/data')

    if not baseDataDir.is_dir():
        baseDataDir = Path(__file__).absolute().parent.parent / 'data'

    if not baseDataDir.is_dir():
        baseDataDir = Path(__file__).absolute().parent.parent / 'data-test'

    return baseDataDir
