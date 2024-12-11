from pathlib import Path
from pickle import load
from arff import dump


def main():
    print("Started Exporting Data Frame to ARFF...")

    pickls_directory = Path("Data/Pickls")
    input_filename = r"training_data_df"
    output_filename = r"training_data_df"

    # Load the combined dataframe
    with open(pickls_directory / '{}.pkl'.format(input_filename), 'rb') as file:
        df = load(file)

    print(df.tail())

    df = df.drop(columns=['evals'])  # Drop the Stockfish column for the features

    # Define the attributes (column names and their types)
    attributes = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # If it's a string, it is nominal. We define the possible values.
            unique_values = df[col].unique()
            attributes.append((col, unique_values))
        elif df[col].dtype == 'bool':
            # If the column is boolean, treat it as a nominal attribute with 'True' and 'False'
            attributes.append((col, ['True', 'False']))
        else:
            # Otherwise, it's a numeric attribute
            attributes.append((col, 'NUMERIC'))

    # Convert DataFrame to ARFF
    arff_dict = {
        'description': u'',
        'relation': 'my_relation',
        'attributes': attributes,
        'data': df.values.tolist(),
    }

    # Save the ARFF file
    with open(pickls_directory / '{}.arff'.format(output_filename), 'w') as f:
        dump(arff_dict, f)

    print("Finished Exporting Data Frame to ARFF...")


if __name__ == "__main__":
    main()
