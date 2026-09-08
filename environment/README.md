# Environment

The GUAVA-Hand experiments were executed in an Apptainer environment based on
the PyTorch 24.02 container.

The core runtime versions observed inside the experiment container are recorded
in `versions.txt`.

Several GUAVA dependencies are provided through the repository or through
project-specific Python paths rather than as independently installed top-level
Python packages. These include the Gaussian rasterization modules and other
vendored components inherited from GUAVA.

The experiment shell helper configures temporary-directory and Python module
paths; it does not itself create or activate the Python environment.

Machine-specific filesystem paths, container images, datasets, checkpoints,
tracking outputs, and rendered results are not redistributed in this repository.
