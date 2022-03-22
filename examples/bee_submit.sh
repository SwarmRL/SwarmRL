#! /bin/bash

# Input script for mdsuite analysis run.

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J SwarmRl_tutorial
#SBATCH -n 8

### ----------------------------------------- ###
### Input parameters for logging and run time ###
### ----------------------------------------- ###

#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=# Your email here

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

module load gcc/8.4.0     	 # gcc compiler
module load openmpi/4.0.5  	 # openmpi binaries
module load python		 # load python
module load py-espresso		 # load espresso
module load py-tqdm		 # load tqdm
module load py-h5py		 # load h5py
module load py-numpy		 # load numpy
module load py-pint		 # load pint
module load py-torch		 # load torch
module load py-yamlreader	 # load yaml reader

# Avoid install the library with pip
export PYTHONPATH="/beegfs/work/stovey/Software/SwarmRL":${PYTHONPATH}

# Custom install as spack install is not new enough.
pypresso=/beegfs/work/stovey/Software/espresso/build/pypresso

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory
${pypresso} find_center.py
