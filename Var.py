import numpy as np np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()
x_train = np.expand_dims(x_train.astype('float32') / 255.0, -1)
x_test = np.expand_dims(x_test.astype('float32') / 255.0, -1)

input_shape = (28,28,1)
latent_dim = 2  # 2D latent space for visualization


encoder_inputs = layers.Input(shape=input_shape)
x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

# Sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(7*7*64, activation='relu')(latent_inputs)
x = layers.Reshape((7,7,64))(x)
x = layers.Conv2DTranspose(64,3,strides=2,padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(32,3,strides=2,padding='same', activation='relu')(x)
decoder_outputs = layers.Conv2DTranspose(1,3,activation='sigmoid',padding='same')(x)

decoder = Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstructed = self.decoder(z)
        # Reconstruction loss
        recon_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.reshape(x, [tf.shape(x)[0], -1]),
                                                tf.reshape(reconstructed, [tf.shape(reconstructed)[0], -1]))
        )
        recon_loss *= 28*28
        # KL divergence
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        self.add_loss(tf.reduce_mean(recon_loss + kl_loss))
        return reconstructed

vae = VAE(encoder, decoder)
vae.compile(optimizer='adam')


history = vae.fit(x_train, x_train,
                  epochs=30,
                  batch_size=128,
                  validation_data=(x_test, x_test))

def visualize_vae_keras(vae, encoder, decoder, x_test, n_images=10, latent_dim=2):
    idx = np.random.randint(0, x_test.shape[0], n_images)
    x_sample = x_test[idx]
    x_recon = vae.predict(x_sample, verbose=0)
    
    plt.figure(figsize=(15, 6))
    
    # Original images
    for i in range(n_images):
        plt.subplot(3, n_images, i+1)
        plt.imshow(x_sample[i].reshape(28,28), cmap='gray')
        plt.axis('off')
        if i == n_images//2:
            plt.title("Original")
    
    # Reconstructed images
    for i in range(n_images):
        plt.subplot(3, n_images, i+1+n_images)
        plt.imshow(x_recon[i].reshape(28,28), cmap='gray')
        plt.axis('off')
        if i == n_images//2:
            plt.title("Reconstructed")
    
    # Grid of generated digits from 2D latent space
    if latent_dim == 2:
        grid_size = n_images
        grid_x = np.linspace(-3, 3, grid_size)
        grid_y = np.linspace(-3, 3, grid_size)
        figure = np.zeros((28*grid_size, 28*grid_size))
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample, verbose=0)
                figure[i*28:(i+1)*28, j*28:(j+1)*28] = x_decoded[0].reshape(28,28)
        plt.subplot(3,1,3)
        plt.imshow(figure, cmap='gray')
        plt.axis('off')
        plt.title("2D Latent Space Grid")
    
    plt.tight_layout()
    plt.show()


visualize_vae_keras(vae, encoder, decoder, x_test, n_images=10, latent_dim=2)


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VAE Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
