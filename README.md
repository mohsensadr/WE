[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2308.02607.svg)](https://doi.org/10.48550/arXiv.2308.02607)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Moment-Constraint and Entropic Optimal Transport

In this repository, we provide a 3D implementation of Wasserstein-Entropic approximation to optimal transport problem. The code is based on the paper:

```
@article{sadr2024wasserstein,
  title={Wasserstein-penalized Entropy closure: A use case for stochastic particle methods},
  author={Sadr, Mohsen and Hadjiconstantinou, Nicolas G and Gorji, M Hossein},
  journal={Journal of Computational Physics},
  volume={511},
  pages={113066},
  year={2024},
  publisher={Elsevier}
}
```

Preprint is available at https://doi.org/10.48550/arXiv.2308.02607.

## Usage:

To use the code, first, import the content of the ```src/WE.py``` via

```
import sys
import os
src_path = os.path.abspath(os.path.join(os.getcwd(), '[path/to/src]'))
```

Then, instantiate the class by passing list of powers for each dimensions, and the call ```forward(.)```. For example, for second-moment matching, use

```
id0 = [1,0,0, 2,0,0,1,1,0]
id1 = [0,1,0, 0,2,0,1,0,1]
id2 = [0,0,1, 0,0,2,0,1,1]
we = WE(id0,id1,id2)
X, Y, wdist = we.forward(X, Y, id0, id1, id2, Nt=50, dt=1.e-6)
```

For more details, see the notebooks in the directory ```examples/```.
