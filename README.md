DLSuR: Dynamic light scattering microrheology in Python
======================================

DLSuR is a data analysis tool for analyzing the scattering intensity from a dynamic light scattering instrument and deriving the microrheology spectrum in the Python programming language.

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

To use DLSuR, you need to:
* have data from a dynamic light scattering instrument,
* save the data in the specific format that is listed in this paper[cite], and
* be sure to collect data following the methods listed in this paper[cite]

# The DLSuR environment

## Easy Implementation

The DLSuR method is simple to implement, utilizing just the scattering autocorrelation of embedded particles in a given soft material sample. The methods are split into different ways to analyze and visualize one's data.

By using only the scattering autocorrelation, the methodology of analyzing the mean-squared displacement of embedded particles to derive the frequency-dependent complex modulus becomes much simpler than other microrheology techniques such as video particle tracking.

## Large Range of Rheological Behavior

DLSuR has the capability of measuring up to six decades in rheological behavior without using time-temperature superposition. This is a major advantage over state-of-the-art rheological techniques such as oscillatory rheometers. 

Each algorithm comes packaged with a frontend and backend. The frontend takes care of
interfacing with the user. The backend defines functions necessary for
computation of the scattering transform.

Currently, there are six available frontend–backend pairs, NumPy (CPU), scikit-learn (CPU), pure PyTorch (CPU and GPU), PyTorch+scikit-cuda (GPU), TensorFlow (CPU and GPU), and Keras (CPU and GPU).

## How to cite

If you use this package, please cite the following paper:

Andreux M., Angles T., Exarchakis G., Leonarduzzi R., Rochette G., Thiry L., Zarka J., Mallat S., Andén J., Belilovsky E., Bruna J., Lostanlen V., Hirn M. J., Oyallon E., Zhang S., Cella C., Eickenberg M. (2019). Kymatio: Scattering Transforms in Python. arXiv preprint arXiv:1812.11214. [(paper)](https://arxiv.org/abs/1812.11214)

# Installation


## Dependencies

DLSuR requires:

* Python (>= 3.5)
* SciPy (>= 0.13)


### Standard installation (on CPU hardware)
We strongly recommend running DLSuR in an Anaconda environment, because this simplifies the installation of other
dependencies. You may install the latest version of DLSuR using the package manager `pip`, which will automatically download
DLSuR from the Python Package Index (PyPI):

```
pip install DLSuR
```

Linux and macOS are the two officially supported operating systems.


# Frontend

## NumPy

To explicitly call the `numpy` frontend, run:

```
from kymatio.numpy import Scattering2D
scattering = Scattering2D(J=2, shape=(32, 32))
```

## Scikit-learn

After installing the latest version of scikit-learn, you can call `Scattering2D` as a `Transformer` using:

```
from kymatio.sklearn import Scattering2D

scattering_transformer = Scattering2D(2, (32, 32))
```

## PyTorch

After installing the latest version of PyTorch, you can call `Scattering2D` as a `torch.nn.Module` using:

```
from kymatio.torch import Scattering2D

scattering = Scattering2D(J=2, shape=(32, 32))
```

## TensorFlow

After installing the latest version of TensorFlow, you can call `Scattering2D` as a `tf.Module` using:

```
from kymatio.tensorflow import Scattering2D

scattering = Scattering2D(J=2, shape=(32, 32))
```

## Keras

Alternatively, with TensorFlow installed, you can call `Scattering2D` as a Keras `Layer` using:

```
from tensorflow.keras.layers import Input
from kymatio.keras import Scattering2D

inputs = Input(shape=(32, 32))
scattering = Scattering2D(J=2)(inputs)
```

# Installation from source

Assuming the Kymatio source has been downloaded, you may install it by running

```
pip install -r requirements.txt
python setup.py install
```

Developers can also install Kymatio via:

```
pip install -r requirements.txt
python setup.py develop
```


## GPU acceleration

Certain frontends, `numpy` and `sklearn`, only allow processing on the CPU and are therefore slower. The `torch`, `tensorflow`, and `keras` frontends, however, also support GPU processing, which can significantly accelerate computations. Additionally, the `torch` backend supports an optimized `skcuda` backend which currently provides the fastest performance in computing scattering transforms. In 2D, it may be instantiated using:

```
from kymatio.torch import Scattering2D

scattering = Scattering2D(J=2, shape=(32, 32), backend='torch_skcuda')
```

This is particularly useful when working with large images, such as those in ImageNet, which are of size 224×224.

## PyTorch and scikit-cuda

To run Kymatio on a graphics processing unit (GPU), you can either use the PyTorch-style `cuda()` method to move your
object to GPU. Kymatio is designed to operate on a variety of backends for tensor operations. For extra speed, install
the CUDA library and the `skcuda` dependency by running the following pip command:

```
pip install scikit-cuda cupy
```

The user may control the choice of backend at runtime via for instance:

```
from kymatio.torch import Scattering2D
scattering = Scattering2D(J=2, shape=(32, 32)), backend='torch_skcuda')
```

# Documentation

The documentation of Kymatio is officially hosted on the [kymat.io](https://www.kymat.io/) website.


## Online resources

* [GitHub repository](https://github.com/PamCai/DLSuR)
* [GitHub issue tracker](https://github.com/PamCai/DLSuR/issues)
* [BSD-3-Clause license](https://github.com/PamCai/DLSuR/blob/master/LICENSE.md)


## Building the documentation from source
The documentation can also be found in the `doc/` subfolder of the GitHub repository.
To build the documentation locally, please clone this repository and run

```
pip install -r requirements_optional.txt
cd doc; make clean; make html
```

## Support

We wish to thank Stanford University, National Science Foundation, Stanford Bio-X Initiative for their financial support.