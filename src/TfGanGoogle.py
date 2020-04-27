import os
import time
from random import Random
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display
from tensorflow.keras import layers
from tqdm import tqdm
from PIL import Image
import numpy as np
import sys
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import requests
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

# Configuration
DATA_PATH = 'Data'
GRADIENT_PATH = "/artifacts"
GENERATE_RES = 4  # Generation resolution factor (1=32, 2=64, 3=96, 4=128, etc.)
GENERATE_SQUARE = int(32 * GENERATE_RES)  # rows/cols (should be square) 380
BUFFER_SIZE = 60000
BATCH_SIZE = 42  # 512  # 256
EPOCHS = 5000
noise_dim = 100
num_examples_to_generate = 16


def load_data(training_binary_path=None):
    if training_binary_path is None:
        training_binary_path = os.path.join(DATA_PATH, f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')
    print(f"Looking for file: {training_binary_path}")
    if not os.path.isfile(training_binary_path):
        print("Loading training images...")
        training_data = []
        faces_path = os.path.join(DATA_PATH, 'Cat')
        for filename in tqdm(os.listdir(faces_path)):
            path = os.path.join(faces_path, filename)
            try:
                image = Image.open(path).convert('L').resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
                training_data.append(np.asarray(image))
            except Exception as e:
                print(str(e))
        training_data = np.reshape(training_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, 1))
        training_data = training_data.astype(np.float32)
        training_data = (training_data - 127.5) / 127.5
        for _ in range(2):
            plt.imshow(
                training_data[Random().randint(0, len(training_data) - 1)].reshape(GENERATE_SQUARE, GENERATE_SQUARE),
                cmap=plt.get_cmap('gray'))
            plt.show()

        print("Saving training image binary...")
        np.save(training_binary_path, training_data)
        return training_data
    else:
        print("Loading previous training pickle...")
        return np.load(training_binary_path)





def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(int(((GENERATE_SQUARE / 2) / 2) * int((GENERATE_SQUARE / 2) / 2) * 256), use_bias=False,
                           input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape(((int(GENERATE_SQUARE // 2) // 2), int((GENERATE_SQUARE // 2) // 2), 256)))
    assert model.output_shape == (
        None, int((GENERATE_SQUARE / 2) / 2), int((GENERATE_SQUARE / 2) / 2), 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int((GENERATE_SQUARE / 2) / 2), int((GENERATE_SQUARE / 2) / 2), 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int((GENERATE_SQUARE / 2) / 2), int((GENERATE_SQUARE / 2) / 2), 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int((GENERATE_SQUARE / 2) / 2), int((GENERATE_SQUARE / 2) / 2), 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(GENERATE_SQUARE / 2), int(GENERATE_SQUARE / 2), 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, GENERATE_SQUARE, GENERATE_SQUARE, 1)

    return model


generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')


def make_discriminator_model():
    model = tf.keras.Sequential()
    # model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[GENERATE_SQUARE, GENERATE_SQUARE, 1]))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    # model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    # model.add(layers.Flatten())
    # model.add(layers.Dense(1))
    model.add(
        layers.Conv2D(32, kernel_size=3, strides=2, input_shape=[GENERATE_SQUARE, GENERATE_SQUARE, 1], padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))  # .assert_consumed()

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(dataset, epochs, path):
    print(
        f"Configuration:\nResolution: {GENERATE_SQUARE}*{GENERATE_SQUARE}\nBuffer_SIZE: {BUFFER_SIZE}\nBATCH_SIZE: {BATCH_SIZE}\nEPOCHS: {EPOCHS}\nNOISE_DIM: {noise_dim}\num_examples_to_generate: {num_examples_to_generate}\nDATA_PATH: {DATA_PATH}")
    for epoch in range(epochs):
        start = time.time()
        gen_loss_list = []
        disc_loss_list = []
        for image_batch in dataset:
            t = train_step(image_batch)

            gen_loss_list.append(t[0])
            disc_loss_list.append(t[1])

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed, path)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        g_loss = sum(gen_loss_list) / len(gen_loss_list)
        d_loss = sum(disc_loss_list) / len(disc_loss_list)
        print(
            f'Epoch {epoch + 1}, gen loss={g_loss},disc loss={d_loss}, Time for epoch {epoch + 1} is {time.time() - start} sec')

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed, path)


def generate_and_save_images(model, epoch, test_input, path):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'{path}/image_at_epoch_{epoch:04d}.png')
    plt.close(fig)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    BATCH_SIZE, RunOnGradient, LoadFromCheckPoint = sys.argv[1:]
    BATCH_SIZE = int(BATCH_SIZE)
    RunOnGradient = int(RunOnGradient)
    LoadFromCheckPoint = int(LoadFromCheckPoint)
    path = "output"
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(load_data()).shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE)
    if bool(RunOnGradient):
        checkpoint_dir = '/artifacts/training_checkpoints'
        path = "/artifacts/output"
        ID = str(sys.argv[4])
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists('/artifacts/Data'):
            os.mkdir('artifacts/Data')
        download_file_from_google_drive(ID, 'artifacts/Data/data.np')
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(load_data('artifacts/Data/data.np')).shuffle(
            BUFFER_SIZE).batch(BATCH_SIZE)
    if bool(LoadFromCheckPoint):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    train(train_dataset, EPOCHS, path)
