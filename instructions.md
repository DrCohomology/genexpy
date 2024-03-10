# Project Template repository

This repository is intended as a template on how to setup a good data science project. It contains a basic python package, some notebooks, instructions how to import your source code to a notebook and how to install python, python environments and structure your project.

## 1. Installing Python and Python Environments


## 1.1. Python For Windows

By installing Anaconda you basically get everything you would need in a Python installation. Find it here: 
[https://www.anaconda.com/download#downloads](https://www.anaconda.com/download#downloads)

## 1.2 Python for Linux

For linux users I recommend to install and use [https://github.com/pyenv/pyenv](https://github.com/pyenv/pyenv)

Pyenv also works in conjunction with Pipenv (see below) which automatically installs the desired python versions when pyenv is available.

## 1.3 Managing Python Environments (Windows and Linux)

To manage and install python environments I recommend to use Pipenv.

Install via pip:
```
pip install pipenv
```

Basic usage examples:
```
pipenv --python 3.10                # Create a new environment
pipenv shell                        # Activate local environment, Pipfile needs to be present in folder
exit                                # Deactivate the local environment
pipenv install <package_name>       # Install a new package to the environment, adds entry to Pipfile
pipenv uninstall <_package_name>    # Uninstall a package, removes entry from Pipfile
pipenv update                       # Update all packages to their newest version
pipenv install                      # Install all packages listed in the Pipfile
```

## 2. How to use this project

After installing python and pipenv you need to setup your python environment. Use the pipenv install command to install the template. Then make your python environment aware of your own package. For this, first you need to activate the shell and then execute the setup script. The develop keyword will cause your python environment to link to your sourcefiles instead of copying and actually packaging your package. As it states in the name, this is the intended usage for developing a python package.

```
pipenv install
pipenv shell
python setup.py develop
```

Now open up a notebook and run it. That should be it.

## 3. Project Structure

    │
    ├── build              <- Automatically generated folder used by Pipenv. Do not manually edit!
    │
    ├── data               <- The original, immutable data dump. Do not modify manually, instead perform
    │                         any modifications with python scripts. Only save intermediate results if 
    │                         computation is excessive (longer than a minute).
    │
    ├── docs               <- A folder for all your documentation. Usually as markdown files in the 
    │                         form of <topic>.md
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   │                      the creator's initials, and a short `-` delimited description, e.g.
    │   │                     `1.0-jqp-initial-data-exploration`.
    │   ├── simple.ipynb   <- An example notebook with at least one import from your sourcefolder src
    │   └── simple-neo4j-connection.ipynb   <- An example how to query neo4j database
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in your thesis
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── settings.py    <- A settings file to save your settings in. For example your user and 
    │   │                     password for a database like neo4j.
    │   ├── module.py      <- A hello world module as example. Will be imported in your notebook.
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── plot          <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── Pipfile            <- The pipfile for reproducing your analysis environment
    ├── Pipfile.lock       <- Automatically generated Pipfile with fixed versions. Do not manually edit!
    ├── README.md          <- The top-level README for developers using this project.
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported

--------



## 4. Write your own readme

Any good project also has good documentation. You will read anything you write far more often than you write it. Avoid using unnecessary abbreviations. Anything written is always better if it is easy to read rather than saving time writing.

Include the following topics in your readme:
- Introduction, write a bit what this project contains
- Installation, tell your user how to install any dependencies and run your code
- Usage, have you defined any steps in which your program should be run? List them here. Also include how it should be configured.
- References, used any external resources or projects in your own? Link here what might be interesting for your reader to follow up on. 

## 5. Start your project

This should be it. Now happy coding and for any further questions email me at daniel.betsche@kit.edu.