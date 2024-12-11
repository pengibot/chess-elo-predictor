# Chess Elo Analyser

This is my Masters Project code to analyse chess game data stored in PGN format 
and predict a chess player's Elo rating. It uses regression and neural networks, 
separated into 2 separate folders. 

## Installing Software

Install Anaconda from https://docs.anaconda.com/anaconda/install/ 

Create a Virtual Environment using Python 3.10
```
conda create -n py310 python=3.10
conda activate py310
```

Upgrade PIP, SetupTools and Wheel
```
python -m pip install --upgrade pip setuptools wheel
```

## Installing Requirements

Install CUDA Toolkit and cuDNN (needed for Tensorflow to use on NVIDIA GPUs)
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Install Requirements
```
pip install -r requirements.txt
```


Set TF_GPU_ALLOCATOR. This ensures that TensorFlow uses the cuda_malloc_async allocator, 
which is often more efficient for managing memory on GPUs

```
set TF_GPU_ALLOCATOR=cuda_malloc_async
```

## Preparing the Data

There is test data you can use in **filter-games/Lichess** folder that you can use to see the code 
running (contains 110 games after filtering). To build real models you will need a much larger dataset, 
that you can get from Lichess.org. 

Navigate to https://database.lichess.org/ and download a PGN file to process.
Each of these files are ~30GB compressed and ~200GB uncompressed on disk. 

These will then need to be split up into smaller parts (~1GB each file) using **PgnSplit.exe**, 
which is available from https://github.com/cyanfish/pgnsplit/releases/download/v1.1/PgnSplit.exe.

Once you have some data ready to process, you need to place it in the **Lichess** folder

## Running the Program

The games need to be filtered by running the following command

```
cd filter-games
python main.py
```

Go back to the root directory if needed
```
cd ..
```

To run the regression code use the following command
```
cd regression
python main.py
```

Go back to the root directory if needed
```
cd ..
```

Run the neural network code using the following command
```
cd neural-network
python main.py
```
