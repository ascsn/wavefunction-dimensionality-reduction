import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense
from tensorflow.keras.models import Model

class ConvAutoencoder:
    '''
    This is a class that will perform dimensionality reduction with a convolutional autoencoder
    
    Methods
    '''
    
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        # Encoder
        inputs = Input(shape=self.input_shape)
        x = Conv1D(16, 3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(8, 3, activation='relu', padding='same')(x)
        encoded = MaxPooling1D(2, padding='same')(x)

        # Decoder
        x = Conv1D(8, 3, activation='relu', padding='same')(encoded)
        x = UpSampling1D(2)(x)
        x = Conv1D(16, 3, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def train(self, data, epochs=10, batch_size=32, validation_split=0.2):
        self.autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def encode(self, data):
        encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer('max_pooling1d_1').output)
        encoded_data = encoder.predict(data)
        return encoded_data

    def decode(self, encoded_data):
        decoded_data = self.autoencoder.predict(encoded_data)
        return decoded_data
