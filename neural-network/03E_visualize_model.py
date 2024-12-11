from pathlib import Path

from keras.layers import SpatialDropout2D, BatchNormalization, LeakyReLU, Multiply, GlobalAveragePooling1D
from tensorflow.keras import layers, models, regularizers, Input
from tensorflow.keras.layers import Add, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model


def bottleneck_block(x, filters, stride=1):
    shortcut = layers.Conv2D(filters * 4, (1, 1), strides=stride, kernel_regularizer=regularizers.l2(0.001))(x)
    shortcut = BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, (1, 1), strides=stride, kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(filters * 4, (1, 1), kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.1)(x)
    x = add_se_block(x)
    return x


def add_se_block(x, ratio=16):
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    return Multiply()([x, se])


def dense_block(x, growth_rate, _layers):
    for _ in range(_layers):
        cb = _layers.BatchNormalization()(x)
        cb = _layers.ReLU()(cb)
        cb = _layers.Conv2D(growth_rate, (3, 3), padding='same')(cb)
        x = _layers.Concatenate()([x, cb])
    return x


def create_cnn():
    inputs = Input(shape=(8, 8, 3), dtype="float32")

    # Initial Convolutional Block
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = SpatialDropout2D(0.2)(x)  # Spatial dropout
    x = add_se_block(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)  # Replace pooling with strided conv

    # First Bottleneck Block
    x = bottleneck_block(x, 64)
    x = SpatialDropout2D(0.3)(x)

    # Second Bottleneck Block with increased filters
    x = bottleneck_block(x, 128)
    x = SpatialDropout2D(0.3)(x)

    # Multi-Scale Feature Extraction Block
    x1 = layers.Conv2D(256, (1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x2 = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x3 = layers.Conv2D(256, (5, 5), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Concatenate()([x1, x2, x3])  # Combine multi-scale features
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = add_se_block(x)
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)  # Replace pooling with strided conv

    # Third Bottleneck Block with more filters
    x = bottleneck_block(x, 256)
    x = SpatialDropout2D(0.3)(x)

    # Global pooling to reduce dimensionality before dense layers
    x = GlobalAveragePooling2D()(x)

    # Dense layers with dropout for regularization
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)

    return models.Model(inputs, x)


def create_rnn(cnn_model, sequence_length=150):
    input_seq = Input(shape=(sequence_length, 8, 8, 3), dtype="float32")
    x = layers.Masking(mask_value=0.0)(input_seq)
    x = layers.TimeDistributed(cnn_model)(x)

    # Apply bidirectional LSTM with MultiHead Attention
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attention_output])  # Residual connection for attention output
    x = GlobalAveragePooling1D()(x)

    # Dense layers with dropout
    x = layers.Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_white = layers.Dense(1, name='white_output')(x)
    output_black = layers.Dense(1, name='black_output')(x)

    return models.Model(inputs=input_seq, outputs=[output_white, output_black])


# Create the CNN model
model = create_rnn(create_cnn())

pickls_directory = Path('Data/Pickls')
cnn_model_file = r"cnn_model.png"

# Save the model architecture to an image file
plot_model(model, to_file=pickls_directory / cnn_model_file, show_shapes=True, show_layer_names=True)
