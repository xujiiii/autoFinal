# How to rerun the pipeline

## Environment build
Conda is strongly recommended to be used to build the environment.

If you haven't use it before, please see the official link https://www.anaconda.com/download and donwload miniconda or anaconda.

After download， you can use the example command to build the environment.
```bash
conda env create -f environment.yml -n autoFinal
```

Token is needed to use library called modeller. Please see the https://salilab.org/modeller/ and get the token.

After receiving your token "Example_token", please add it to the 
`/miniforge3/envs/autoFinal/lib/modeller-10.8/modlib/modeller/config.py` or the corresponding configuration file in anaconda
(replace XXXX with your Modeller license key).

Finally use the following command to activate the environment.
```bash
conda activate autoFinal
```

## How to use the commands
Please run the commands in the order they appear in the commands.txt, executing each within its corresponding folder.

For example,
```bash
cd Pipe/P1/
```
And run
```bash
python download_pdbs.py --structsCSV ../../input_data/f1_in/structure-matching-IPR011009_updated.tsv --out ../../input_data/PDBs
```
After executing all command in `Pipe/P1`, go to the next folder to run all the corresponding commands.

## Outputs
All outputs are stored under `output_data/` and seperated according to the code created them.
