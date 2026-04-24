# SafetyCage Tutorials and Examples

Welcome to the safetycage tutorials explaining how to use each safetycage method!

We hope this helps you get started with using safetycage, helps explain how to use safetycage and how it can benefit your ML classification projects.

Check out the [safetycage repo](https://github.com/safety-cage/safetycage) and [PyPI](https://pypi.org/project/safetycage/)!

## Explaining this Repo

There are currently 3 examples in this repository, listed in order of simplicity, they are:

- The Iris Dataset with a Random Forest Classifier.
- The MNIST Dataset with a MLP (Multilayer Perceptron).
- The CIFAR-10 Dataset with a CNN (Convolutional Neural Network).

Each example is available as its own folder.

In each folder, there are the following folders and files:
- modules/
   - These store the data module and model module, a key element of using the safetycage package.
   - These modules organize your data and model in a way so that even when you change your safetycage method, you don't have to change the data module or model module!
- train_model/
   - This contains the Python scripts to run and train your model.
- [method]_tutorial.ipynb.
   - The tutorials demonstrate how to use a specific safetycage method.
   - These are step by step Jupyter Notebooks that explain how to use safetycage and explores what safetycage can do!
- [method].py
   - Most methods are provided in the safetycage library. However, MNIST and CIFAR-10 include [method].py files to demonstrate how to organize custom SafetyCage methods.

While running the tutorials, you may also generate the following folders:
- metadata/
   - For saving information about your data and model.
- results/ and results/plots/
   - For saving information and plots of what safetycage has revealed.
- saved_cage
   - To save and load data about your safetycage.

Each folder has a multitude of safetycage's methods, see below for where you can find which method:

| Method | Iris | MNIST | CIFAR-10 |
|--------|------|-------|----------|
| MSP | ✓ | ✓ | |
| DOCTOR | | ✓ | |
| MAHALANOBIS | | | ✓ |
| SPARDACUS | | ✓ | ✓ |

## Getting Started

Begin by git cloning this repo. You will need [Git](https://git-scm.com/install/) installed to do this.

```
git clone https://github.com/safety-cage/safetycage-tutorials.git
cd safetycage-tutorials
```

Then, to run the Python scripts and Jupyter Notebooks, you will need [Python](https://www.python.org/downloads/). Safetycage requires Python==3.11.7.

uv users can make use of the `pyproject.toml` file and run the following commands.

```
uv venv --python 3.11.7
uv sync
```

Other users can make use of the `requirements.txt` file to install dependencies.

> WARNING: Sometimes there are issues related to using Tensorflow on a Linux/Mac vs Windows operating system. We are working to help you avoid these issues!

## Running the Code
We recommend going through the examples in order of simplicity. 

1. First, you must train your model before you run it. Run train_model/main.py.
2. Go through modules/data_module.py and modules/model_module.py to get an understanding and how to define these and what they are doing.
3. Run the Juptyer Notebook.
   - Get an understanding of how to use safetycage and the data and model modules!
   - Edit the code, play with parameters, and check out the source code!

> TIP: Restarting your kernel can help fix odd errors!


### Good luck and have fun!