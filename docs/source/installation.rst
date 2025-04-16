Installation
==================

Base Package
----------------
The repository is available on PyPI and can be installed using your
preferred package manager.

.. code-block:: bash

   pip install deinterlacing

GPU-Support
----------------

If you want to use gpu-parallelization, you will need to install `CuPy <https://github.com/cupy/cupy>`_
by following these `instructions <https://docs.cupy.dev/en/stable/install.html>`_.
As of the time of writing, on windows this would be:

.. centered:: Install CUDA 12X

Download & install the latest version of the CUDA toolkit from
`Nvidia CUDA Toolkit 12.X <https://developer.nvidia.com/cuda-downloads>`_

Ensure proper installation by running the following command in your terminal:

.. code-block:: bash

    nvidia-smi

.. centered:: Install CuPy

Install CuPy using pip. Make sure to replace `12x` with `11x` if you are using CUDA 11.X.
.. code-block:: bash

   pip install cupy-cuda12x

Ensure proper installation by running the following command in your python environment:
.. code-block:: python

    import cupy as cp

    # Check if GPU is available
    assert cp.cuda.runtime.getDeviceCount() > 0

.. centered:: Install deinterlacing

Install the deinterlacing package.
.. code-block:: bash

   pip install deinterlacing

Verify the installation by running the following command in your python environment:

.. code-block:: python

    import deinterlacing
    import numpy as np

    parameters = deinterlacing.DeinterlacingParameters(use_gpu=True)

    # Create a random 3D array
    input_array = np.random.rand(100, 100, 3).astype(np.float32)

    # Deinterlace the array
    deinterlacing.deinterlace(input_array, parameters)
