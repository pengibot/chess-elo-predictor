import pickle
from pathlib import Path
import tensorflow as tf
from keras.layers import GlobalAveragePooling1D, SpatialDropout2D, BatchNormalization, LeakyReLU, Multiply
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers, Input
from tensorflow.keras.layers import Add, Dense, Dropout, GlobalAveragePooling2D


class CombinedMAECheckpoint(Callback):
    def __init__(self, filepath):
        super(CombinedMAECheckpoint, self).__init__()
        self.filepath = filepath
        self.best_combined_mae = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        combined_mae = logs['val_white_output_mae'] + logs['val_black_output_mae']
        if combined_mae < self.best_combined_mae:
            self.best_combined_mae = combined_mae
            self.model.save(self.filepath)
            print(f"\nEpoch {epoch + 1}: Combined MAE improved to {combined_mae:.4f}, saving model to {self.filepath}")
        else:
            print(f"\nEpoch {epoch + 1}: Combined MAE did not improve from {self.best_combined_mae:.4f}")


def extract_move_number(filename):
    move_number = re.search(r'chessboard_move_(\d+)_', filename)
    return int(move_number.group(1)) if move_number else None


def extract_ratings(filename):
    parts = filename.split('_')
    white_rating = int(parts[3][1:])
    black_rating = int(parts[4][1:])
    return white_rating, black_rating


def process_files(directory):
    """Checks if the filename matches the chessboard move format and is a PNG file."""
    filenames = os.listdir(directory)
    pattern = r'chessboard_move_(\d+)_'  # Regex for move number
    for filename in filenames:
        move_number_match = re.search(pattern, filename)

        if move_number_match and filename.endswith(".png"):
            continue
        else:
            full_path = os.path.join(directory, filename)
            os.remove(full_path)
            print(f"Deleted '{filename}'.")


def load_images_and_labels_from_game(game_path):
    process_files(game_path)
    filenames = os.listdir(game_path)
    print(game_path)
    filenames = sorted(filenames, key=extract_move_number)

    images = []
    white_ratings = []
    black_ratings = []

    for filename in filenames:
        img = load_img(os.path.join(game_path, filename), target_size=(8, 8))
        img = img_to_array(img) / 255.0
        white_rating, black_rating = extract_ratings(filename)
        images.append(img)
        white_ratings.append(white_rating)
        black_ratings.append(black_rating)

    return np.array(images), np.array(white_ratings), np.array(black_ratings)


def load_images_and_labels(directory):
    all_images = []
    all_white_ratings = []
    all_black_ratings = []
    max_moves = 0

    for game_folder in os.listdir(directory):
        game_path = os.path.join(directory, game_folder)
        if os.path.isdir(game_path):
            images, white_ratings, black_ratings = load_images_and_labels_from_game(game_path)

            all_images.append(images)
            all_white_ratings.append(white_ratings[0])
            all_black_ratings.append(black_ratings[0])
            max_moves = max(len(images), max_moves)

    all_images_padded = pad_sequences(all_images, maxlen=max_moves, padding='post', dtype='float32')

    return np.array(all_images_padded), np.array(all_white_ratings), np.array(all_black_ratings), max_moves


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

    model = models.Model(inputs=input_seq, outputs=[output_white, output_black])
    return model


def main():

    print("Started Creating Model")

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    models_directory = Path('Data/Models')
    models_directory.mkdir(parents=True, exist_ok=True)

    pickls_directory = Path("Data/Pickls")
    pickls_directory.mkdir(parents=True, exist_ok=True)

    games_directory = Path('Data/Games')
    games_directory.mkdir(parents=True, exist_ok=True)

    best_model_filename = 'best_model_combined_mae.keras'

    combined_mae_checkpoint = CombinedMAECheckpoint(filepath=models_directory / best_model_filename)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available and TensorFlow can use it.")
        for gpu in gpus:
            print(f"GPU found: {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. TensorFlow is using the CPU.")

    images, white_ratings, black_ratings, max_moves = load_images_and_labels(games_directory)

    # Train-Validation-Test split
    X_train_val, X_test, y_train_val_white, y_test_white, y_train_val_black, y_test_black = train_test_split(
        images, white_ratings, black_ratings, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train_white, y_val_white, y_train_black, y_val_black = train_test_split(
        X_train_val, y_train_val_white, y_train_val_black, test_size=0.125, random_state=42
    )

    cnn_model = create_cnn()
    model = create_rnn(cnn_model, max_moves)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics={
        'white_output': ['mae'],
        'black_output': ['mae']
    })

    model.summary()

    # Train the model
    history = model.fit(
        X_train, [y_train_white, y_train_black],
        validation_data=(X_val, [y_val_white, y_val_black]),  # Use validation set here
        epochs=100,
        batch_size=8,
        callbacks=[combined_mae_checkpoint, early_stopping]
    )

    # Save the history
    with open(pickls_directory / r'training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    # Test evaluation
    test_loss, test_white_loss, test_black_loss, test_white_mae, test_black_mae = model.evaluate(
        X_test, [y_test_white, y_test_black]
    )
    print(f"Test loss: {test_loss}")
    print(f"White MAE on test set: {test_white_mae}")
    print(f"Black MAE on test set: {test_black_mae}")

    # Save the model
    model.save(models_directory / 'model.keras')

    print("\nFinished Creating Model")


if __name__ == "__main__":
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    main()
