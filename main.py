import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from data import make_batches
from utils import positional_encoding

if __name__ == '__main__':

    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    for pt_examples, en_examples in train_examples.batch(3).take(1):
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))
        print()
        for en in en_examples.numpy():
            print(en.decode('utf-8'))

    model_name = "ted_hrlr_translate_pt_en_converter"
    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir='.', cache_subdir='', extract=True
    )

    tokenizers = tf.saved_model.load(model_name)
    for en in en_examples.numpy():
        print(en.decode('utf-8'))

    encoded = tokenizers.en.tokenize(en_examples)

    for row in encoded.to_list():
        print(row)

    round_trip = tokenizers.en.detokenize(encoded)
    for line in round_trip.numpy():
        print(line.decode('utf-8'))

    tokens = tokenizers.en.lookup(encoded)

    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    n, d = 2048, 512
    pos_encoding = positional_encoding(n, d)
    print(pos_encoding.shape)
    pos_encoding = pos_encoding[0]

    # Juggle the dimensions for the plot
    pos_encoding = tf.reshape(pos_encoding, (n, d // 2, 2))
    pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
    pos_encoding = tf.reshape(pos_encoding, (d, n))

    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()
