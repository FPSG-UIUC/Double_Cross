# Double Cross: Subverting Active Learning Systems
## Abstract
Active learning is widely used in data labeling services to support real-world
machine learning applications. By selecting and labeling the samples that have
the highest impact on model retraining, active learning can reduce labeling
efforts, and thus reduce cost.

In [Double
Cross](https://www.usenix.org/conference/usenixsecurity21/presentation/vicarte),
we present a novel attack called Double Cross, which aims to manipulate data
labeling and model training in active learning settings. To perform a
double-cross attack, the adversary crafts inputs with a special trigger pattern
and sends the triggered inputs to the victim model retraining pipeline. The
goals of the triggered inputs are (1) to get selected for labeling and
retraining by the victim; (2) to subsequently mislead human annotators into
assigning an adversary-selected label; and (3) to change the victim model's
behavior after retraining occurs. After retraining, the attack causes the
victim to mislabel any samples with this trigger pattern to the
adversary-chosen label. At the same time, labeling other samples, without the
trigger pattern, is not affected. We develop a trigger generation method that
simultaneously achieves these three goals. We evaluate the attack on multiple
existing image classifiers and demonstrate that both gray-box and black-box
attacks are successful. Furthermore, we perform experiments on a real-world
machine learning platform (Amazon SageMaker) to evaluate the attack with human
annotators in the loop, to confirm the practicality of the attack. Finally, we
discuss the implications of the results and the open research questions moving
forward.

## Setting up the Environment
The necessary packages are included in a conda environment file. This file
can be used to create an environment by running:
```zsh
$ conda create --name double_cross --file double_cross_env.txt
```

## Attack Components
![Figure 2](full_attack.pdf)
Figure 2 describes the attack process which is emulated by this code.
At a high level, that process consists of:
1. Training a Trigger Generator ([acgan.py](acgan.py)).
2. Evaluating trigger selectability against a victim model, across various
   selection criterion ([threshold_perf.py](threshold_perf.py)).
3. Evaluating trigger effectiveness at attacking a victim model
   ([train_on_gen.py](train_on_gen.py)).

## Quick and Dirty Usage
As described in the paper, there are many possible configurations for trigger
generation (before an attack) and usage (during an attack).
[sweep.sh](sweep.sh) simplifies exploration by performing the three
stages described above across a multitude of configurations.
This script is highly configurable.
It is described at a high level below, as a starting point for usage.
Each component can be used individually as well.
Components generate sample triggers and triggered images as well.

Configuration Options:
- `-s` Evaluate another dataset. *The script defaults to Imagenet*. Other
    available datasets are:
   - `cifar10` with ResNet18
   - `cifar10_r20` with ResNet20
   - `cifar10_r32` with ResNet32
   - `cifar10_r44` with ResNet44
   - `cifar10_r56` with ResNet56
   - `cifar10_pgd` uses PGD training for the victim model
   - `svhn`
   - `gtsrb`
- `-x` Perform black box trigger generation. *Default is gray box*.
- `-a` Perform baseline trigger generation with a GAN.
- `-e` Perform baseline trigger generation with random noise.
- `-q` Sweep number of queries. *Default behavior is to use 400 queries*. This configuration will train five different generators. Using this option with Imagenet constrains training scales to a single value (2.5).
- `-t` Target label.
- `-d` Path to dataset. *Default is `./data`*.
- `-u` Number of CPUs to use during training. Default behavior is to infer using `lscpu`.
- `-g` Train the generator(s) and evaluate their selectability without training the victim model.
- `-f` Cutoff to use for generator training. *Default is dataset specific*.
- `-h` Range to use for generator training. *Default is dataset specific*.

# Processing Results Files
The final stage of any attack is [train_on_gen.py](train_on_gen.py).
This script performs a Double Cross attack on a pre-trained victim, using a trained generator.
The results of this attack are both dumped to STDOUT and logged to csv files.
These log files contain a vast amount of information.
Some processing scripts are included in this repository for parsing and visualizing some relevant metrics.
Notably, use of these files to interpret results is entirely optional.
The raw data available after running [train_on_gen.py](train_on_gen.py) can be used directly.

A warning: these scripts are very picky about filenames and their formatting.
If you experience problems loading/processing files, it's most likely because of that.
These scripts attempt to "intelligently" parse different attack configurations for easy comparison.
However, that means they (a) ask for a lot of information and (b) are brittle in the way they get that information.

The first, [shorten.py](shorten.py), extracts some relevant metrics from each result file and merges them into a single file (per experiment).
The second, [visualize.py](visualize.py), processes _shortened_ files.
This script compares the different attack configurations and highlights the most successful ones.

# Advanced Configuration
As described in the paper, "Double Cross" attacks have a large design space.
The most efficient way to understand each script's configurability will be by running them with `--help`.

# Sources for Models
- Pretrained Cifar models: https://github.com/chenyaofo/pytorch-cifar-models
- GTSRB models: https://github.com/poojahira/gtsrb-pytorch
- MNIST models: https://github.com/aaron-xichen/pytorch-playground
- Generator model: https://github.com/clvrai/ACGAN-PyTorch
