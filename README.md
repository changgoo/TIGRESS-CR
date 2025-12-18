This repo depends on [`pyathena`](https://github.com/jeonggyukim/pyathena). Tested and worked for commit ID: 87af62bcc25b7822d31b8e199e88d243d31f11b5

# pyathena installation

Below is an example of how you can set up pyathena. It assumes that you have already installed [miniconda](https://docs.conda.io/en/latest/miniconda.html) or anaconda on your system.

1. Clone the pyathena repo
   ```sh
   git clone https://github.com/jeonggyukim/pyathena.git
   ```
2. Create an environment from the env.yml file
   ```sh
   conda update conda
   conda env create -f path_to_pyathena/env.yml
   ```
3. Activate the pyathena environment
   ```sh
   conda activate pyathena
   ```
4. Install pyathena
   (optional) to absolutely make sure to use the version that works
   ```sh
   git checkout 87af62bcc25b7822d31b8e199e88d243d31f11b5
   ```
   ```sh
   pip install .
   ```

# running figure drawing scripts
```sh
cd python
python paperI.py
```

