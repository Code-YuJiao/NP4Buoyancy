Citation: Yu Jiao, Han Zhang, Tingkai Xue, Boyang Chen, Juntao Yang, Te Ba, Simon See, Chang Wei Kang, Claire E. Heaney, Christopher C. Pain, Hongying Li, NP4Buoyancy: An End-to-End Differentiable Neural-Physics Framework for Buoyancy-Driven Flows, Journal of Computational Physics (2026), doi: https://doi.org/10.1016/j.jcp.2026.115179

# NP4Buoyancy

**A differentiable PyTorch solver for buoyancy-driven flows.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.3-ee4c2c.svg)](https://pytorch.org/)

NP4Buoyancy is a PyTorch-based solver for Boussinesq buoyancy-driven flow
problems, including Rayleigh-Benard convection and side-heated cavity cases.
The code supports forward simulations in 2D and 3D, and an inverse workflow
for reconstructing initial states from sparse observations.

The implementation uses fixed convolution stencils for spatial operators, which
keeps the solver compatible with PyTorch automatic differentiation. Forward
runs can use CPU, single-GPU, or distributed GPU execution depending on the
available hardware and configuration.

## Repository Structure

```text
NP4Buoyancy/
|-- run.py                  # command-line entry point
|-- configuration.yaml      # example configuration
|-- simulation_manager.py   # forward and inverse driver routines
|-- numerics_setup.py       # numerical operators and solver classes
|-- physics_setup.py        # boundary conditions, initial conditions, restart
|-- utils.py                # runtime, distributed setup, config, tensor helpers
|-- monitor.py              # diagnostics
|-- postprocess.py          # output saving and plots
|-- inverse_solver.py       # inverse rollout and loss utilities
|-- inverse_plot.py         # inverse-analysis plotting utilities
|-- installation/
|   |-- environment.yml     # conda environment
|   `-- requirements.txt    # pip requirements
|-- CITATION.cff
|-- README.md
`-- LICENSE
```

## Installation

Conda:

```bash
conda env create -f installation/environment.yml
conda activate np4buoyancy
```

Pip:

```bash
pip install -r installation/requirements.txt
```

For a specific CUDA build of PyTorch, install PyTorch from the official PyTorch
index first, then install the remaining requirements.

## Quick Start

Edit `configuration.yaml`, then run:

```bash
python run.py --config configuration.yaml
```

For distributed forward simulations, launch with `torchrun`, for example:

```bash
torchrun --nproc_per_node=4 run.py --config configuration.yaml
```

The solver mode is selected in `configuration.yaml`:

| Mode | Description |
|---|---|
| `2D` | 2D forward simulation |
| `3D` | 3D forward simulation |
| `INVERSE` | Inverse state reconstruction |

Outputs are written to the configured output directory. Large generated files
such as field snapshots, plots, and inverse results are excluded from version
control by `.gitignore`.

## Citation

If you use this software, please cite:

> Yu Jiao, Han Zhang, Tingkai Xue, Boyang Chen, Juntao Yang, Te Ba,
> Simon See, Chang Wei Kang, Claire E. Heaney, Christopher C. Pain,
> Hongying Li, **NP4Buoyancy: An End-to-End Differentiable Neural-Physics
> Framework for Buoyancy-Driven Flows**, *Journal of Computational Physics*
> (2026), doi: <https://doi.org/10.1016/j.jcp.2026.115179>.

## License

Released under the [MIT License](./LICENSE).
