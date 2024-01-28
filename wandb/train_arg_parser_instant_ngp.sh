#!/usr/bin/env bash

# Initialize empty strings for the two groups of arguments
args1=""
args2=""

# Iterate over all passed arguments
for arg in "$@"; do
    # Check if the argument starts with '--pipeline.'
    if [[ $arg == --scale-factor* ]]; then
        # Add to group2
        args2="$args2 $arg"
    else
        # Add to group1
        args1="$args1 $arg"
    fi
done

# Call the ns-train script with the grouped arguments
echo "Parsed arguments, running command"
ns-train rfqa-instant-ngp --vis=wandb --project_name=radiance-field-qa \
    --experiment_name=lego-instant-ngp --viewer.quit_on_train_completion=True --pipeline.model.disable_scene_contraction=True \
    $args1 rfqa-blender-data $args2 --scale-factor=1.5 --data ~/Datasets/NeRF/blender/lego
