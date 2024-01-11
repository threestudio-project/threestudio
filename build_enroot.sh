#!/bin/bash

#SBATCH --job-name=threestudio_enroot_build
#SBATCH --account=stability
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

docker build -t threestudio-ssc -f docker/Dockerfile.enroot .
enroot import -o threestudio-ssc-0.0.1.sqsh '[dockerd://threestudio-ssc](dockerd://threestudio-ssc)'