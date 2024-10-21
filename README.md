# CSE 842 Project: Semantic Relatedness

Thad Greiner

Yufeng Li

### Setup:

Here is our project repo. First, clone this repo whereever you want it stored locally. Then, get the data
provided by cloning this repo: https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024.git.
I recommend doing so inside the CSE-842-Project folder that way the filepaths don't need changed.

After gathering all the data, I recommend setting up a virtual environment in the workspace. You will need the following packages:
1. pandas
2. numpy
3. scikit-learn
4. matplotlib
5. Levenshtein

After getting steup, the files of not are the Jupyter notebooks.

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

It apprears that the basic linear regression overfits rather hard, so we'll likely need to reduce this going forward.