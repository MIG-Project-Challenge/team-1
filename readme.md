<div align="center">
    <a href="https://www.michiganinvestmentgroup.com/"><img src="./media/logo.jpeg"></a>
</div>

# Project Challenge 1

- [Competition Spec](https://docs.google.com/document/d/1aBTDQckRDnt5PEU_qzmjxsmGURQZOdF2RPsreJY_voY/edit?usp=sharing)
- [Piazza](https://piazza.com/umich/fall2023/mig101)
- [Algorithm Report turn in link](https://forms.gle/Ac4UJxsYPdErdtdC7)

You can download the train data here [Google Drive Link](https://drive.google.com/file/d/1ruCDFeUqLEkPDfSLBqDMf8swyU1ECEwq/view?usp=sharing)

## Sections
[About This Repo](#about)

[Getting Started](#getting-started)

[Fundamental Prep](#fundamental-prep)

[Developer Environments](#developer-environments)


## <a name="about"></a>About This Repo
The repo will give you everything you need to get started building Algorithms for the MIG Algo Competition.

## <a name="getting-started"></a>Getting Started

1. [Create an Anaconda python environment](#developer-environments) for the examples and future algorithm development
2. [Git clone](#cloning-this-repo) this repo onto your computer and go through the examples and make sure you understand
3. Brush up on your [python/git skills](#fundamental-prep)

## <a name="developer-environments"></a>Developer Environments

We will be using anaconda python environments to have easy and portable python environments that are consistent across contestant's machines.

First you will need to go to [anaconda.com](https://www.anaconda.com/download) and download and install it anaconda (if you have not already).

(the below may differ for windows machines please refer to anacondas docs for more info)

Now we can create a new python environment (where <env_name> is the name of your environment like "migenv")
```
$ conda create --name <env_name> python=3
```

to see a list conda environments that are on our system:
<br>(you should be able to see the <env_name> that you made in the previous step)
```
$ conda info --envs
```

activate conda environment:
```
$ conda activate <env_name>
```

you should now see the name of the environment in your terminal like so:
```
(<env_name>) user@computer % ...
```

next we need to install the python packages that will be used when developing our algorithms:
```
$ pip install -r requirements.txt
```

(Optional - extra info)
If you ever want to switch which environment you are using, first deactivate your env, then activate your desired env:
```
conda deactivate
conda activate <name_of_other_env>
```

### VSCode Jupyter Notebook Extension
In the example I have included some jupyter notebooks as they are more interactive. You can view and edit them with vs code and the jupyter notebook extension. You can alternatively use [jupyter notebooks natively](https://jupyter.org/), but I prefer and recommend to use VSCode and the extension.

## <a name="fundamental-prep"></a>Fundamental Prep

### Python
Most of the code we will be writing will be in python. We will have an education session that covers python, but we have provided some supplemental guides and study material below:
- [Python Learn by example](https://python-by-examples.readthedocs.io/en/latest/)
- [Python Tutorials](https://www.learnpython.org/en/Hello%2C_World%21)

There are many more resources online and if you're ever confused make sure to use resources like ChatGPT, stackoverflow, and others to your advantage!

### <a name="git"></a>Git
We will also be using git with github as a way to develop our code. 

Git is a version control system for software. It allows developers to work on code simultaneously and then merge there code together. There are many other things you can do with git as well. If you want more info you can go here to see some git [tutorials](https://www.w3schools.com/git/git_intro.asp?remote=github)

### <a name="cloning-this-repo"></a>Cloning this Repo

Open up a terminal of your choice and cd into a directory where you want this project to exist. Then:
```
git clone <repo link>
```

the repo link can be found in the top right by the big green button that says code. It should look something like "https://github.com/MIG-Project-Challenge/Challenge-1-base.git" but may differ slightly for different groups. 

---

Now you can develop your algorithm! Happy coding!