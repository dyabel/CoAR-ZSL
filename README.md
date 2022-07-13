# CoAR-ZSL
This is the codes for *Boosting Zero-shot Learning via Contrastive Optimization of Attribute Representations* [arxiv](https://arxiv.org/abs/2207.03824). 
This repo is built on [GEM-ZSL](https://github.com/osierboy/GEM-ZSL) [Class-Norm](https://github.com/universome/class-norm)
# Prepare Data
  Dataset: please download the dataset, i.e., [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [AWA2](https://cvml.ist.ac.at/AwA2/), [SUN](https://groups.csail.mit.edu/vision/SUN/hierarchy.html), and put it under datasets/
# Requirement
- apex
- PyTorch==1.7
- timm

# How to run
see [train_cub.sh](train_cub.sh)
