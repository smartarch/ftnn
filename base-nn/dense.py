import sys
import h5py
import re
import tensorflow as tf

verbose = 0

def loadData(fileName, batchSize):
    with h5py.File(fileName, 'r') as hf:
        trainDS = tf.data.Dataset.from_tensor_slices((hf['train-inputs'], hf['train-outputs'])).cache()
        valDS = tf.data.Dataset.from_tensor_slices((hf['val-inputs'], hf['val-outputs'])).cache()

    trainDS = trainDS.shuffle(buffer_size=len(trainDS)).batch(batchSize).prefetch(1000)
    valDS = valDS.batch(batchSize).prefetch(1000)

    return (trainDS, valDS)

if len(sys.argv) < 3:
    print("At least 2 args expected (file name and one layer witdth)")
    sys.exit(0)

tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

fileName = sys.argv[1]
trainDS, valDS = loadData(fileName, 100)

model = tf.keras.models.Sequential()
for width in sys.argv[2:]:
    model.add(tf.keras.layers.Dense(int(width), activation='relu'))
#dropout - zacit na 0.5 a snizovat
#model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


learning_rate = tf.keras.experimental.CosineDecay(0.001, 100 * len(trainDS))
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
    metrics=['accuracy']
)
# binary crossentropy labal smoothing = 0.1 - dulezite proti overfittingu

accuracies = []
for epochIdx in range(0, 100):
    model.fit(trainDS, epochs=1, verbose=verbose)
    evalRes = model.evaluate(valDS, return_dict=True, verbose=verbose)
    accuracies.append(evalRes['accuracy'])

bareFileName = re.sub(r"^.*/", "", fileName)
print("{};dense-{};{}".format(bareFileName, "-".join(sys.argv[2:]), ";".join(map(str, accuracies))))
