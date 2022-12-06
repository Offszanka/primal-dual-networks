# primal-dual-networks
Improving the Chambolle-Pock algorithm with neural networks

Trying out different approaches to apply machine learning to the Chambolle-Pock algorithm which is given in the paper

    *A. Chambolle and T. Pock*, [**A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging**](https://doi.org/10.1007/s10851-010-0251-1)

The improvements are oriented on the papers

    *S. Wang, S. Fidler and R.Urtasun*, [**Proximal Deep Structured Models**](https://proceedings.neurips.cc/paper/2016/file/f4be00279ee2e0a53eafdaa94a151e2c-Paper.pdf)

    *C. Brauer and D. Lorenz*, [**Primal-dual residual networks**](https://doi.org/10.48550/arXiv.1806.05823)

### Examples:
<img src = "ex_images/castle clean.png"
     style = "float: center; margin-right: 10px;"
     width="200"/>
<img src = "ex_images/castle noisy.png"
     style = "float: center; margin-right: 10px;"
     width="200"/>
<img src = "ex_images/castle PDResNet_.png"
     style = "float: center; margin-right: 10px;"
     width="200"/>
<img src = "ex_images/castle EAPDN.png"
     style = "float: center; margin-right: 10px;"
     width="200"/>

### Requirements
See the requirements.yml file.

To train the models a training dataset will be needed.