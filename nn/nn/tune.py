import json
import logging
import signal
import ssl
import sys
import time
from pathlib import Path

import numpy as np
import pika
import traceback

resultsBaseDir = Path('/mnt/data/results')

def initPika():
    certsPath = Path(__file__).parent.parent.parent / 'certs'
    context = ssl.create_default_context(cafile=certsPath / 'ca_certificate.pem')
    context.load_cert_chain(certsPath / 'client_certificate.pem', certsPath / 'client_key.pem')
    sslOptions = pika.SSLOptions(context, 'localhost')
    connParams = pika.ConnectionParameters(host='XX.XX.XX.XX', port=5671, ssl_options=sslOptions)

    mqConn = pika.BlockingConnection(connParams)
    mqCh = mqConn.channel()
    mqCh.queue_declare('workerAssignments')
    mqCh.queue_declare('workerResponses')

    return mqConn, mqCh


class ExplorerMasterBase():
    def __init__(self, resultsSubDir, workerSourceFile, workerCountPerInstance):
        self.resultsSubDir = resultsSubDir
        self.workerSourceFile = workerSourceFile
        self.workerCountPerInstance = workerCountPerInstance
        self.trialsPending = set()
        self.trialsInProcessing = set()
        self.randomGen = np.random.default_rng()

    def suggestConfig(self):
        raise NotImplementedError()

    def randInt(self, low, high):
        return self.randomGen.integers(low, high, endpoint=True).item()

    def randFloat(self, low, high):
        return self.randomGen.random().item() * high + low

    def randChoice(self, choices):
        return self.randomGen.choice(np.array(choices, dtype=object))

    def randBool(self):
        return self.randChoice([False, True])

    def run(self, experimentId, trialsCount, trialRepetitionsCount):
        mqConn, mqCh = initPika()
        mqCh.exchange_declare(exchange='mlAgentCommands', exchange_type='fanout')
        mqCh.queue_purge(queue='workerAssignments')
        mqCh.queue_purge(queue='workerResponses')

        def sendMLAgentCommand(cmd, data={}):
            msg = dict(data)
            msg['command'] = cmd
            msg['targetInstanceType'] = None

            mqCh.basic_publish(
                exchange='mlAgentCommands',
                routing_key='',
                body=json.dumps(msg),
                properties=pika.BasicProperties(
                    content_type='application/json'
                ),
            )

        def sendShutdown():
            sendMLAgentCommand('stopService')

        def sendStart():
            sendMLAgentCommand('startService', {
                'count': self.workerCountPerInstance,
                'args': ['/usr/bin/python3', self.workerSourceFile]
            })

        def handleResponse(ch, method, properties, body):
            msg = json.loads(body)
            experimentId = msg['experimentId']
            trialId = msg['trialId']
            repetition = msg['repetition']
            status = msg['status']
            results = msg['results']

            trialKey = (experimentId, trialId, repetition)

            if status == 'accepted':
                try:
                    self.trialsPending.remove(trialKey)
                    self.trialsInProcessing.add(trialKey)
                except KeyError:
                    logging.warning(f'Dropping {status} response for {experimentId}:{trialId}:{repetition}')

            elif status == 'completed' or status == 'error':
                try:
                    self.trialsInProcessing.remove(trialKey)
                    if status == 'completed':
                        logging.info(f'Trial {experimentId}:{trialId}:{repetition} finished with status {status}. Results: train loss = {results["trainLoss"]}, train acc = {results["trainAccuracy"] * 100}, val loss = {results["valLoss"]}, val acc = {results["valAccuracy"] * 100}')
                    else:
                        logging.info(f'Trial {experimentId}:{trialId}:{repetition} finished with status {status}.')

                except KeyError:
                    logging.warning(f'Dropping {status} response for {experimentId}:{trialId}:{repetition}')

        def sigIntHandler(sig, frame):
            logging.warning('SIGNINT detected. Sending shutdown to all workers and exiting...')
            sendShutdown()
            time.sleep(3)
            mqCh.queue_purge(queue='workerAssignments')
            mqConn.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, sigIntHandler)

        logging.info('Shutting down dangling workers...')
        sendShutdown()
        time.sleep(3)
        logging.info('Starting workers ...')
        sendStart()

        mqCh.basic_consume(queue='workerResponses', auto_ack=True, on_message_callback=handleResponse)

        for trialId in range(trialsCount):
            logging.info(f'Starting suggestConfig')
            trialConfig = self.suggestConfig()
            logging.info(f'Received trialConfig from suggestConfig')

            logging.info(f'Starting with trial {experimentId}:{trialId}')
            logging.debug(trialConfig)

            resultsDir = resultsBaseDir / self.resultsSubDir / str(experimentId) / str(trialId)
            resultsDir.mkdir(parents=True, exist_ok=True)

            trialTemplate = {
                'experimentId': experimentId,
                'trialId': trialId,
                'config': trialConfig
            }
            with open(resultsDir / f'trial.json', 'wt') as trialFile:
                json.dump(trialTemplate, trialFile, indent=4)

            for repetition in range(trialRepetitionsCount):
                while len(self.trialsPending) > 0:
                    logging.debug('Ensuring that services are running')
                    sendStart()

                    mqConn.process_data_events(time_limit=10)

                logging.debug(f'Submitting trial {experimentId}:{trialId}:{repetition}')

                trial = dict(trialTemplate)
                trial['repetition'] = repetition

                self.trialsPending.add((experimentId, trialId, repetition))

                mqCh.basic_publish(
                    exchange='',
                    routing_key='workerAssignments',
                    body=json.dumps(trial),
                    properties=pika.BasicProperties(
                        content_type='application/json',
                        delivery_mode=2,  # make message persistent
                    ),
                )

                mqConn.process_data_events()

        while len(self.trialsPending) > 0 or len(self.trialsInProcessing) > 0:
            mqConn.process_data_events()

        sendShutdown()
        mqConn.close()




class ExplorerWorker():
    def __init__(self, resultsSubDir, TrainableNNClass):
        self.resultsSubDir = resultsSubDir
        self.TrainableNNClass = TrainableNNClass

    def run(self):
        mqConn, mqCh = initPika()

        mqCh.basic_qos(prefetch_count=1)

        def publishResponse(experimentId, trialId, repetition, status, results=None):
            response = {
                'experimentId': experimentId,
                'trialId': trialId,
                'repetition': repetition,
                'status': status,
                'results': results
            }

            mqCh.basic_publish(
                exchange='',
                routing_key='workerResponses',
                body=json.dumps(response),
                properties=pika.BasicProperties(
                    content_type='application/json'
                ),
            )

        def yieldFn():
            mqConn.process_data_events()

        for method, properties, body in mqCh.consume('workerAssignments'):
            msg = json.loads(body)
            experimentId = msg['experimentId']
            trialId = msg['trialId']
            repetition = msg['repetition']
            config = msg['config']

            publishResponse(experimentId, trialId, repetition, 'accepted')
            logging.info(f'Received task {experimentId}:{trialId}:{repetition}')

            try:
                resultsDir = resultsBaseDir / self.resultsSubDir / str(experimentId) / str(trialId)
                resultsDir.mkdir(parents=True, exist_ok=True)

                nnConf = self.TrainableNNClass.createConfig(config)
                nn = self.TrainableNNClass(config=nnConf)
                results = nn.train(config['maxEpochs'], yieldFn=yieldFn)

                nn.save_weights(str(resultsDir / f'weights-{repetition}.hdf5'), save_format='h5')
                with open(resultsDir / f'results-{repetition}.json', 'wt') as resultsFile:
                    data = dict(results)
                    data['experimentId'] = experimentId
                    data['trialId'] = trialId
                    data['repetition'] = repetition
                    json.dump(data, resultsFile, indent=4)

                logging.info(f'Task {experimentId}:{trialId}:{repetition} completed')
                status = 'completed'
            except Exception:
                logging.error(f'Task {experimentId}:{trialId}:{repetition} encountered an error')
                logging.error(traceback.format_exc())
                status = 'error'
                results = None

            publishResponse(experimentId, trialId, repetition, status, results)
            mqCh.basic_ack(delivery_tag=method.delivery_tag)




