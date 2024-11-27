# CSE 842 Project: Semantic Relatedness

Thad Greiner

Yufeng Li

### Setup:

Here is our project repo. First, clone this repo wherever you want it stored locally. Then, get the data
provided by cloning this repo: https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024.git.
I recommend doing so inside the CSE-842-Project folder that way the filepaths don't need changed.

After gathering all the data, I recommend setting up a virtual environment in the workspace. You will need the following packages:
1. pandas
2. numpy
3. scikit-learn
4. matplotlib
5. Levenshtein
6. scipy
7. torch
8. keras
9. tansformers
10. torchsort

I recommend installing torch and torchsort with GPU support and transformers[pytorch]. Depending on your python version, you
may need to workaround distilutils being deprecated. After getting steup, the files of note are the Jupyter notebooks.

Here are the full install comands used:

pip3 install numpy pandas scipy matplotlib scikit-learn
pip3 install python-Levenshtein
pip3 install matplotlib
pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cuda/12.0/torch_stable.html
pip3 install torchsort
pip3 install keras
pip3 install 'tensorflow[and-cuda]'
pip3 install transformers[torch]

You may need to install wheel for torch / torchsort installs:
pip3 install wheel

If using ipykernel for local development, you may need these as well:
pip3 install --upgrade ipykernel pyzmq tornado traitlets
pip3 install --upgrade jupyter ipywidgets

If you want gpu, make sure you have the nvcc command working (NVIDIA CUDA Toolkit):
sudo apt install nvidia-cuda-toolkit
You may also need this command:
pip3 install --force-reinstall --no-cache-dir --no-deps torchsort

Make sure your nvcc and torch cuda versions match, I have opted for cuda 12.

### Baseline model:

The SemEval group gave us a baseline model of simple dice score. It takes both sentences, does very basic preprocessing, then
outputs the dice score. This can be seen for English in project_baseline_eng.ipynb. Our models will need to try and beat this.

### Initial Exeriments:

Our initial exeriments will be kept in project_initial_eng.ipynb. To start, we will stick to fairly basic approaches, so English,
basic preprocessing / feature extraction, and simple models. Here are the models we will test:
1. Linear regression
2. SVR
3. Random Forrest Regressor
4. RNN

It apprears that the basic linear regression overfits rather hard, so we'll likely need to reduce this going forward. Edit: We discovered
that we were using the limited dataset (250 samples), full training data is being used now (5500 samples).

### Final Model

Fpr our final model, we seperated the process into two files: preprocessing and model training/testing. To preprocess the data, run 
project_final_preprocessing.ipynb. This will generate our processed features for each sample, then save them to file (~5mins with cpu, gpu is seconds).
If you already have the saved files, you can skip to project_final_eng.ipynb for training and evaluating the models using that data. It is highly recommended
to use GPU support for this.

Our model uses a two BiLSTM system fallowed by dropout and a Linear output layer. This step uses a customized spearman correlation loss function.
The idea details were borrowed from the following link, as we found an existing method of calculating this with graidents using a relatively new method
(https://forum.numer.ai/t/differentiable-spearman-in-pytorch-optimize-for-corr-directly/2287/26). We adjusted the spearman function to a loss variant,
but the orginal spearman implementation was found here.

After training to minimize spearman loss, or sonversely maximize spearman correlation, we then need to train a tranformation model. This is because while
the corrlated predictions are good, they need shifted and scaled to match the proper scoring values. This involves a simple training of two variables,
shift and scale, such that we use MSE loss on the forward step: x = x * scale + shift. This is intended to be fit as tightly as possible, so it uses an
extensive number of epochs to guarantee overfitting, which is the desired goal. It works just as intended, preserving correlation while minimizing the MSE
loss of the actual values.

Finally, we test our model on the training, testing, and full datasets.