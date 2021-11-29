# Polar Transformer Networks #

Convolutional neural networks (CNNs) are inherently equivariant to translation.
Efforts to embed other forms of equivariance have concentrated solely on rotation.
We expand the notion of equivariance in CNNs through the  Polar Transformer Network (PTN).
PTN combines ideas from the Spatial Transformer Network (STN) and canonical coordinate representations.
The result is a network invariant to translation and equivariant to both rotation and scale.
PTN is trained end-to-end and composed of three distinct stages: a polar origin predictor, the newly introduced polar transformer module and a classifier.
PTN achieves state-of-the-art on rotated MNIST and the newly introduced SIM2MNIST dataset, an MNIST variation obtained by adding clutter and perturbing digits with translation, rotation and scaling.
The ideas of PTN are extensible to 3D which we demonstrate through the Cylindrical Transformer Network.


## Demo ##

We provide a demo code for the paper, where we train and test the PTN-B+ and PTN-B++ variations on the rotated MNIST 12k dataset.

Check requirements in requirements.txt. Our codebase has been tested on TensorFlow 1.15 but the dependency is commented out to silence GitHub's security warnings.

The following code should

  * Create a virtualenv and install the requirements
  * Download the dataset to /tmp
  * Train and test the PTN-B+

```sh
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=1 python3 -u train.py --run_id=ptn-bplus
```

Sample output:

```
...
Training Step: 60  | total loss: 0.65867 | time: 5.775s
| Adam | epoch: 001 | loss: 0.65867 - acc: 0.7749 | val_loss: 1.00748 - val_acc: 0.6736 -- iter: 12000/12000
--
...
Training Step: 30000  | total loss: 0.13798 | time: 4.350s
| Adam | epoch: 500 | loss: 0.13798 - acc: 0.9844 | val_loss: 0.03460 - val_acc: 0.9976 -- iter: 12000/12000
--
...
ptn-bplus. train augmentation. # of params: 131959. Training time: 2344.89 s.
Test accuracy (no test time augmentation): 0.9893
Test accuracy (with test time augmentation): 0.9909
```

## References ##

Esteves, C., Allen-Blanchette, C., Zhou, X. and Daniilidis, K, "Polar Transformer Networks", International Conference on Learning Representations, ICLR 2018, https://openreview.net/pdf?id=HktRlUlAZ.

```bibtex
@article{esteves2018polar,
title={Polar Transformer Networks},
author={Carlos Esteves, Christine Allen-Blanchette, Xiaowei Zhou, Kostas Daniilidis},
journal={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=HktRlUlAZ},
note={accepted as poster},
}
```

## Authors

[Carlos Esteves](http://machc.github.io), [Christine Allen-Blanchette](http://www.seas.upenn.edu/~allec/), [Xiaowei Zhou](https://fling.seas.upenn.edu/~xiaowz), [Kostas Daniilidis](http://www.cis.upenn.edu/~kostas/)

[GRASP Laboratory](http://grasp.upenn.edu), [University of Pennsylvania](http://www.upenn.edu)
