import tensorflow_datasets as tfds

x_train, y_train = tfds.load('caltech_birds2011', split='train',
                             batch_size=-1, as_supervised=True,)

