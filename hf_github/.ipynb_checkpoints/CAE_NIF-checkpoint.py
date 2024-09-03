from tensorflow.keras import layers, models, initializers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

def custom_loss(y_true, y_pred, encoded):
    reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    periodicity_loss = tf.reduce_mean(tf.square(tf.sin(encoded)))  # Encourage values to follow a sinusoidal pattern
    return reconstruction_loss + periodicity_loss

class CAE(models.Model):
    def __init__(self, num, size):
        super(CAE, self).__init__()
        self.encoded = None
        self.decoded = None
        self.num = num
        self.size = size
        
        initializer = initializers.GlorotUniform(seed=42)

        # Encoder
        self.encoder = models.Sequential([
            layers.Conv1D(filters=4, kernel_size=10, padding='same', strides=1),
            layers.Activation('relu'),
            layers.MaxPooling1D(2, padding='same'),
            
            #layers.Conv1D(filters=8, kernel_size=5, padding='same', strides=1),
            #layers.Activation('relu'),
            #layers.MaxPooling1D(2, padding='same'),
            
            #layers.Conv1D(filters=16, kernel_size=5, padding='same', strides=1),
            #layers.Activation('relu'),
            #layers.MaxPooling1D(2, padding='same'),
            
            #layers.Conv1D(filters=32, kernel_size=3, padding='same', strides=1),
            #layers.Activation('relu'),
            #layers.MaxPooling1D(2, padding='valid'),
            
            layers.Flatten(),
            #layers.Dense(64, use_bias=True),
            #layers.Activation('relu'),
            
            #layers.Dense(32, use_bias=True),
            #layers.Activation('relu'),
            
            layers.Dense(32, use_bias=True),
            layers.Activation('relu'),
            
            layers.Dense(5, use_bias=True),
            layers.Activation('relu'),
            
            layers.Dense(self.num, use_bias=True),
        ])
        # Decoder
        self.decoder = models.Sequential([
            layers.Dense(5, use_bias=True),
            layers.Activation('relu'),
            
            layers.Dense(32, use_bias=True),
            layers.Activation('relu'),
            
            layers.Reshape((32, 1)),  # Reshape to match the convolutional input
            
            layers.Conv1DTranspose(filters=4, kernel_size=10, strides=2, padding='same'),
            layers.Activation('relu'),
            
            #layers.Conv1DTranspose(filters=8, kernel_size=3, strides=2, padding='same'),
            #layers.Activation('relu'),
            
            #layers.Conv1DTranspose(filters=4, kernel_size=3, strides=2, padding='same'),
            #layers.Activation('relu'),
            
            #layers.Conv1DTranspose(filters=2, kernel_size=3, strides=2, padding='same'),
            #layers.Activation('relu'),
            
            layers.Flatten(),  # Flatten before final dense layer if needed
            layers.Dense(size, activation='sigmoid'),  # Dense layer to ensure output size matches input size
            layers.Reshape((size, 1)),  # Reshape to the desired output shape
        ])
        
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        self.encoded = encoded
        self.decoded = decoded
        return decoded
    
    def getEncoded(self):
        return self.encoder
    
    def encode(self, inputs):
        return self.encoder(inputs)
    
    def decode(self, inputs):
        return self.decoder(inputs)
    