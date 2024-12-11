import pickle
from pathlib import Path

from matplotlib import pyplot as plt


def main():

    print("Started Analysing History")

    pickls_directory = Path("Data/Pickls")
    training_history_file = r'training_history.pkl'

    # Load History File
    with open(pickls_directory / training_history_file, 'rb') as file:
        history = pickle.load(file)

        print(history)

        # Plot the training and validation loss
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    print("Finished Analysing History")


if __name__ == "__main__":
    main()
