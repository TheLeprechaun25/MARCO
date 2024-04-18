# MARCO: Memory-Augmented Reinforcement framework for Combinatorial Optimization

## Overview
MARCO offers an innovative approach to neural combinatorial optimization (NCO). It integrates a memory module that prevents redundant exploration and promotes the discovery of diverse, high-quality solutions across various problem domains. We include implementations for the maximum cut (MC), maximum independent set (MIS), and the traveling salesman problem (TSP).

![marco](marco.png)

## Requirements

* Python 3.8
* PyTorch

## Usage

To train the model, run the following command:

```bash
python train.py
```

To test the model, run:

```bash
python eval.py
```

## NCOLib
We are currently developing a PyTorch-based NCO library: designed to simplify the application of neural network models and deep learning algorithms to solve combinatorial optimization problems. Check it out
[here](https://github.com/TheLeprechaun25/NCOLib).
