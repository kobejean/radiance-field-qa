import sys
import os

def main():
    # Initialize empty strings for the two groups of arguments
    args1 = ""
    args2 = ""

    # Iterate over all passed arguments (excluding the script name)
    for arg in sys.argv[1:]:
        # Check if the argument starts with '--pipeline.data_manager'
        if arg.startswith('--pipeline.data_manager'):
            # Add to group2
            args2 += f" {arg}"
        else:
            # Add to group1
            args1 += f" {arg}"

    # Prepare the command for ns-train script with the grouped arguments
    command = f"ns-train rfqa-nerfacto --vis=wandb --project_name=radiance-field-qa " \
              f"--experiment_name=lego-grid --viewer.quit_on_train_completion=True " \
              f"--pipeline.model.disable_scene_contraction=True --pipeline.model.use_gradient_scaling=True " \
              f"{args1} rfqa-blender-data {args2} --scale-factor=1.5 --data ~/Datasets/NeRF/blender/lego"

    # Output the parsed arguments and the command (simulating the echo in bash)
    print("Parsed arguments, running command")
    print(command)

    # If needed, execute the command here
    os.system(command)

if __name__ == "__main__":
    main()
