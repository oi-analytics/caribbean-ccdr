# East Caribbean CCDR

Scripts supporting infrastructure climate risk analysis for the East Caribbean.

## Setup

The processing and analysis scripts are written in Python, with several package
dependencies. We recommend using
[micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
to install Python and these libraries into a fresh environment.

```bash
# Clone or download this repository and navigate to the project directory
git clone https://github.com/oi-analytics/caribbean-ccdr.git  # or git@github.com:oi-analytics/caribbean-ccdr.git
cd caribbean-ccdr

# Run this once to create the environment
micromamba env create -f environment.yml

# Run this to activate the environment each time you want to use it
micromamba activate caribbean-ccdr
```

## Acknowledgments

This project was developed by Raghav Pant, Alison Peard and Tom Russell at
Oxford Infrastructure Analytics as part of a World Bank project in 2022-2023.

The code is made available under an MIT license, see [`LICENSE`](./LICENSE) for
details.
