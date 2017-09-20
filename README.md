# Polar Transformer Networks #

  Convolutional neural networks (CNNs) are equivariant with respect to translation; a translation in the input causes a translation in the output. Attempts to generalize equivariance have concentrated on rotations.
  In this paper, we combine the idea of the spatial transformer, and the canonical coordinate representations of groups (polar transform) to realize a network that is invariant to translation, and equivariant to rotation and scale.
  A conventional CNN is used to predict the origin of a polar transform.
  The polar transform is performed in a differentiable way, similar to the Spatial Transformer Networks, and the resulting polar representation is fed into a second CNN.
  The model is trained end-to-end with a classification loss.
  We apply the method on variations of MNIST, obtained by perturbing it with clutter, translation, rotation, and scaling.
  We achieve state of the art performance in the rotated MNIST, with fewer parameters and faster training time than previous methods, and we outperform all tested methods in the SIM2MNIST dataset, which we introduce.

## Demo ##

We provide a demo code for the paper, where we train and test the PTN-B+ and PTN-B++ variations on the rotated MNIST 12k dataset.

Check requirements in requirements.txt

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

Esteves, C., Allen-Blanchette, C., Zhou, X. and Daniilidis, K., 2017. Polar Transformer Networks. arXiv preprint arXiv:1709.01889. http://arxiv.org/abs/1709.01889v1

```bibtex
@article{esteves2017polar,
  title={Polar Transformer Networks},
  author={Esteves, Carlos and Allen-Blanchette, Christine and Zhou, Xiaowei and Daniilidis, Kostas},
  journal={arXiv preprint arXiv:1709.01889},
  url = {http://arxiv.org/abs/1709.01889v1},
  year={2017}
}
```

## Authors

[Carlos Esteves](http://machc.github.io), [Christine Allen-Blanchette](http://www.seas.upenn.edu/~allec/), [Xiaowei Zhou](https://fling.seas.upenn.edu/~xiaowz), [Kostas Daniilidis](http://www.cis.upenn.edu/~kostas/)

[GRASP Laboratory](http://grasp.upenn.edu), [University of Pennsylvania](http://www.upenn.edu)
