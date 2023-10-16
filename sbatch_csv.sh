#!/bin/bash

#SBATCH --account=mod3d
#SBATCH --partition=g40

#SBATCH --job-name=adam_sbatch0         # Job name
#SBATCH --array=0-1  # Create tasks in the job array, with indices for each. Range in our case corresponds number of nodes (and to number of GPUs).
#SBATCH --ntasks=1   # Each array task uses 1 task (typically, one core)
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --nodes=1    # Each task runs on one node
#SBATCH --output=job_output_A%A_a%a_j%j.txt    # Output file name (%j expands to jobID)
#SBATCH --error=job_error_A%A_a%a_j%j.txt      # Error file name (%j expands to jobID)
#SBATCH --time=04:00:00               # Time limit (hh:mm:ss)
#SBATCH --mem-per-cpu=10G              # Memory per processor

# Check if the batchname is provided
if [ -z "$1" ]; then
    echo "Error: No batchname provided."
    echo "Usage: sbatch your_slurm_script.sh <batchname> <input_filename> <threestudio_root_directory> [optional_phase_numbers]"
    exit 1
fi

# Check if the batchname contains a period
case $1 in
    *.*)
    echo "Error: The batchname should not contain a period. (Did you forget the batchname, and accidentally supply the input CSV path here?)"
    exit 1
    ;;
esac
BATCHNAME=$1

# Check if the filename is provided
if [ -z "$2" ]; then
    echo "Error: No input file provided."
    echo "Usage: sbatch your_slurm_script.sh <batchname> <input_filename> <threestudio_root_directory> [optional_phase_numbers]"
    exit 1
fi

# Check if the provided file actually exists
if [ ! -f "$2" ]; then
    echo "Error: File '$2' not found."
    exit 1
fi

# Check if the second parameter (directory) is provided
if [ -z "$3" ]; then
    echo "Error: No directory provided."
    echo "Usage: sbatch your_slurm_script.sh <batchname> <input_filename> <threestudio_root_directory> [optional_phase_numbers]"
    exit 1
fi

# Check if the provided directory exists
if [ ! -d "$3" ]; then
    echo "Error: Directory '$3' not found."
    exit 1
fi
BASE_DIR=$3

# If a phase number is provided, split by comma into an array. Otherwise, use a default array with all phases.
IFS=',' read -ra PHASE_NUMS <<< "${4:-1,2,3}"  # Adjust the default value '1,2,3' to include all your phase numbers

# Check each phase number for validity
for PHASE_NUM in "${PHASE_NUMS[@]}"; do
    if [[ ! "$PHASE_NUM" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: Phase number '$PHASE_NUM' is not a valid positive integer."
        exit 1
    fi
done

echo "Processing for Phases: ${PHASE_NUMS[@]}"

# Count the number of lines in the file
TOTAL_ROWS=$(( $(wc -l < "$2") + 1 ))

# If there are more tasks than rows
if [ $TOTAL_ROWS -le ${SLURM_ARRAY_TASK_MAX} ]; then
    # Only tasks with ID less than TOTAL_ROWS should run, and each task processes one row
    if [ $SLURM_ARRAY_TASK_ID -lt $TOTAL_ROWS ]; then
        START_ROW=$(( $SLURM_ARRAY_TASK_ID + 1 ))
        END_ROW=$START_ROW
    else
        echo "Nothing to process for task ${SLURM_ARRAY_TASK_ID}"
        exit 0
    fi
else
    # Distribute the rows among tasks when there are more rows than tasks
    # Use the following formula to round up during division
    ROWS_PER_TASK=$(( ( TOTAL_ROWS + SLURM_ARRAY_TASK_MAX ) / ( ${SLURM_ARRAY_TASK_MAX} + 1 ) ))

    START_ROW=$(( $SLURM_ARRAY_TASK_ID * ROWS_PER_TASK + 1 ))

    if [ $SLURM_ARRAY_TASK_ID -eq $SLURM_ARRAY_TASK_MAX ]; then
        END_ROW=$TOTAL_ROWS
    else
        END_ROW=$(( ($SLURM_ARRAY_TASK_ID + 1) * ROWS_PER_TASK ))
    fi
fi

echo "Hello from ${SLURM_ARRAY_TASK_ID}"

INPUT_FILE=$2  # Assign the provided filename to INPUT_FILE variable

# Set common base arguments for all potential phases
WANDB_ARGS="system.loggers.wandb.enable=False system.loggers.wandb.project=dummy system.loggers.wandb.name=dummy"
declare -A BASE_ARGS
BASE_ARGS[1]="--config=/fsx/proj-mod3d/adam/threestudio-mesh/configs/zero123_sai_multinoise_amb.yaml --train tag=Phase1 use_timestamp=false ${WANDB_ARGS}"
BASE_ARGS[2]="--config=/fsx/proj-mod3d/adam/threestudio-mesh/configs/zero123_magic123refine.yaml --train tag=Phase2_magic use_timestamp=false ${WANDB_ARGS}"
BASE_ARGS[3]="--export tag=Phase3 system.exporter_type=mesh-exporter use_timestamp=false ${WANDB_ARGS}"

# Load any necessary modules or environment variables
source ~/venv/threestudio/bin/activate

# Process each row within the task's range
for ((i=$START_ROW; i<=$END_ROW; i++)); do
    # Debug: Print the current row being processed
    echo "Processing row: $i"

    read ITEMNAME DEG REFIMAGE PROMPT <<< $(awk -v row="$i" 'BEGIN { FPAT = "([^,]+)|(\"[^\"]+\")" } NR == row { gsub(/^"|"$/, "", $1); gsub(/""/, "\"", $1); gsub(/^"|"$/, "", $2); gsub(/""/, "\"", $2); gsub(/^"|"$/, "", $3); gsub(/""/, "\"", $3); gsub(/^"|"$/, "", $4); gsub(/""/, "\"", $4); print $1, $2, $3, $4 }' "$INPUT_FILE")

    # If all the variables are empty, it means there's no more data
    if [ -z "$ITEMNAME" ] && [ -z "$DEG" ] && [ -z "$REFIMAGE" ] && [ -z "$PROMPT" ]; then
        echo "ITEMNAME: ${ITEMNAME}"
        echo "DEG: ${DEG}"
        echo "REFIMAGE: ${REFIMAGE}"
        echo "PROMPT: ${PROMPT}"
        echo "Warning: Empty data for row $i. Exiting loop."
        break
    fi

    NAME="$BATCHNAME/$ITEMNAME"
    echo "NAME: ${NAME}"

    # If all the variables are empty, it means there's no more data
    if [ -z "$NAME" ] && [ -z "$DEG" ] && [ -z "$REFIMAGE" ] && [ -z "$PROMPT" ]; then
        echo "Warning: Empty data for row $i. Exiting loop."
        break
    fi

    for PHASE_NUM in "${PHASE_NUMS[@]}"; do
        ARGS=${BASE_ARGS[$PHASE_NUM]}

        # For Phase 2, we need INPUT_CKPT
        if [ "$PHASE_NUM" -eq 2 ]; then
            INPUT_CKPT="${BASE_DIR}/outputs/${NAME}/Phase1/ckpts/last.ckpt"
            echo "INPUT_CKPT: ${INPUT_CKPT}"
            ARGS="$ARGS system.geometry_convert_from=\"$INPUT_CKPT\""
        fi

        # For Phase 3, we need to take the config per frame
        if [ "$PHASE_NUM" -eq 3 ]; then
            PHASE3_CONFIG_PATH="${BASE_DIR}/outputs/${NAME}/Phase2_magic/configs/parsed.yaml"
            RESUME_CKPT="${BASE_DIR}/outputs/${NAME}/Phase2_magic/ckpts/last.ckpt"
            echo "PHASE3_CONFIG_PATH: ${PHASE3_CONFIG_PATH}"
            echo "RESUME_CKPT: ${RESUME_CKPT}"
            ARGS="$ARGS --config=${PHASE3_CONFIG_PATH} resume=${RESUME_CKPT}"
            ALL_MESHES_DIR="${BASE_DIR}/outputs/${BATCHNAME}/all_meshes"
        fi

        # Debug: Print the current variables
        echo "BASE_DIR: ${BASE_DIR}"
        echo "INPUT_FILE: ${INPUT_FILE}"
        echo "THE PROMPT: ${PROMPT}"
        echo "NAME: ${NAME}"
        echo "DEG: ${DEG}"
        echo "REFIMAGE: ${REFIMAGE}"
        echo "ARGS: ${ARGS}"

        python /fsx/proj-mod3d/adam/threestudio-mesh/launch.py $ARGS name="$NAME" data.default_elevation_deg="$DEG" data.image_path="$REFIMAGE" system.prompt_processor.prompt="$PROMPT"

        if [ "$PHASE_NUM" -eq 3 ]; then
            # Get the directory "it${N}-export" where N is numerically greatest
            GREATEST_EXPORT_DIR=$(ls -d "${BASE_DIR}/outputs/${NAME}/Phase3/save/it"*"-export" 2>/dev/null | sort -n -t 't' -k 2 | tail -1)

            # Check if the directory exists
            if [ -z "$GREATEST_EXPORT_DIR" ]; then
                echo "Warning: No mesh 'it*-export' directories found for ${NAME}. Skipping this iteration."
                continue
            fi
            
            echo "GREATEST_EXPORT_DIR: ${GREATEST_EXPORT_DIR}"

            # Path to the output .mtl file
            MTL_FILE="${GREATEST_EXPORT_DIR}/model.mtl"
            RENAMED_MTL_FILE="${GREATEST_EXPORT_DIR}/${BATCHNAME}_${ITEMNAME}.mtl"
            TEX_FILE="${GREATEST_EXPORT_DIR}/texture_kd.jpg"
            RENAMED_TEX_FILE="${GREATEST_EXPORT_DIR}/${BATCHNAME}_${ITEMNAME}_tex.jpg"
            OBJ_FILE="${GREATEST_EXPORT_DIR}/model.obj"
            RENAMED_OBJ_FILE="${GREATEST_EXPORT_DIR}/${BATCHNAME}_${ITEMNAME}.obj"

            # Use sed to use unique material and texture file name per object in material output
            sed -i "s/newmtl .*/newmtl ${BATCHNAME}_${ITEMNAME}/" "$MTL_FILE"
            sed -i "s/map_Kd .*/map_Kd ${BATCHNAME}_${ITEMNAME}_tex.jpg/" "$MTL_FILE"

            # E.g.
            # mtllib cactus.mtl
            # usemtl cactus
            # Use sed to use unique material lib and material name in obj output
            sed -i "s/mtllib .*/mtllib ${BATCHNAME}_${ITEMNAME}\.mtl/" "$OBJ_FILE"
            sed -i "s/usemtl .*/usemtl ${BATCHNAME}_${ITEMNAME}/" "$OBJ_FILE"

            if mv "$MTL_FILE" "$RENAMED_MTL_FILE"; then
                echo "File renamed to $RENAMED_MTL_FILE successfully."
            else
                echo "Error: Failed to rename $MTL_FILE to $RENAMED_MTL_FILE."
            fi
            if mv "$TEX_FILE" "$RENAMED_TEX_FILE"; then
                echo "File renamed to $RENAMED_TEX_FILE successfully."
            else
                echo "Error: Failed to rename $TEX_FILE to $RENAMED_TEX_FILE."
            fi
            if mv "$OBJ_FILE" "$RENAMED_OBJ_FILE"; then
                echo "File renamed to $RENAMED_OBJ_FILE successfully."
            else
                echo "Error: Failed to rename $OBJ_FILE to $RENAMED_OBJ_FILE."
            fi

            # Create the ALL_MESHES_DIR directory and any necessary parent directories
            mkdir -p "${ALL_MESHES_DIR}"

            # Copy all files from GREATEST_EXPORT_DIR to ALL_MESHES_DIR
            cp -r "${GREATEST_EXPORT_DIR}"/* "${ALL_MESHES_DIR}/"
        fi
    done
done
