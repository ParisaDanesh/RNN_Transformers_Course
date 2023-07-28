import tensorflow as tf
import tensorflow_datasets as tfds


def net():
    """
    :return: compiled mlp architecture
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model


def normalize_img(img, label):
    """
    :param img: image in format 'uint8'
    :param label: the class of the number in the image, [0, 9]
    :return: image with format 'float32' and its label
    """
    return tf.cast(img, tf.float32) / 255., label


def main():
    # preparing dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # train data
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    )

    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # test data
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = net()
    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test
    )

    results = model.evaluate(ds_test)

    res_txt = "Loss: {:.2f}\nAccuracy: {:.2f}".format(results[0], results[1])
    with open("results_mlp.txt", "w") as f:
        f.write(res_txt)


if __name__ == '__main__':
    main()
