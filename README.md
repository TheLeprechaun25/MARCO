# MARCO: Memory-Augmented Reinforcement framework for Combinatorial Optimization

## Overview
MARCO offers an innovative approach to neural combinatorial optimization (NCO). It integrates a memory module that prevents redundant exploration and promotes the discovery of diverse, high-quality solutions across various problem domains. We include implementations of improvement methods for the maximum cut (MC) and maximum independent set (MIS), and constructive method for the traveling salesman problem (TSP).
<div align="center">
  <img src="marco_general_framework.png" alt="marco">
</div>

## Supplementary material
Access the supplementary material [here](Supplementary_Material_MARCO.pdf)

## Requirements
* PyTorch

## Usage

To train a model, run the following command inside a problem folder:

```bash
python train.py
```

To test a model, run:

```bash
python eval.py
```
### Configuration

To adjust training and evaluation settings, modify the parameters in:

-    *options/train_options.py*
-    *options/eval_options.py*

## NCOLib
Explore our new PyTorch-based library, NCOLib, designed to simplify the application of neural network models and deep learning algorithms to solve combinatorial optimization problems. Learn more [here](https://github.com/TheLeprechaun25/NCOLib).

## Contributing
We welcome contributions! If you'd like to improve MARCO or report an issue, please open an issue on our repository.

## Citation
If you find MARCO useful in your research or projects, we kindly request that you cite our article:
