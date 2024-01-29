#!/usr/bin/env bash

# Initialize empty strings for the two groups of arguments
args1=""
args2=""

# Iterate over all passed arguments
for arg in "$@"; do
    # Check if the argument starts with '--pipeline.'
    if [[ $arg == --pipeline.data_manager* ]]; then
        # Add to group2
        args2="$args2 $arg"
    else
        # Add to group1
        args1="$args1 $arg"
    fi
done

# Call the ns-train script with the grouped arguments
echo "Parsed arguments, running command"
ns-train rfqa-nerfacto --vis=wandb --project_name=radiance-field-qa \
    --experiment_name=lego-grid --viewer.quit_on_train_completion=True --pipeline.model.disable_scene_contraction=True \
    --pipeline.model.use_gradient_scaling=True \
    $args1 rfqa-blender-data $args2  --scale-factor=1.5 --data ~/Datasets/NeRF/blender/lego
