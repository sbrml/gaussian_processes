# Welcome to the Gaussian Processes Section!

Here we demonstrate tasks for which GPs are suitable, and examine their 
advantages and disadvantages.

## Current Implementations

 - Gaussian Process Latent Variable Models for 
 Visualisation of High Dimensional Data [[paper](https://dl.acm.org/citation.cfm?id=2981387)]

 - Variational Learning of Inducing Variables in Sparse Gaussian
 Processes [[paper](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.330.2631)]
 
## Installation

**Note:** when code blocks are prefixed with `>`, it means the command should 
be preformed in the terminal.

1. Clone the repository:
```
> git clone https://github.com/sbrml/gaussian_processes.git
```

2. Create a virtual environment in the repo's folder, and activate it:
```
> cd gaussian_processes
> python3 -m venv gp_venv
> source gp_venv/bin/activate
```

3. Install the requirements:
```
(gp_venv)> pip install --upgrade pip 
(gp_venv)> pip install -r requirements.txt 
```

4. Run a Jupyter Notebook:
```
(gp_venv)> jupyter notebook
```

The notebooks can be found in the `code` folder.
