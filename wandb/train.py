import sys
import os
import math

def parse_arguments(argv):
    args1 = {}
    args2 = {}

    for arg in argv:
        if arg.startswith('--'):
            key, _, value = arg.lstrip('-').partition('=')
            if key.startswith('pipeline.data_manager'):
                args2[key] = value
            else:
                args1[key] = value
    F_log = int(math.log2(float(args1["pipeline.model.features_per_level"])))
    L_log = int(math.log2(float(args1["pipeline.model.num_levels"])))
    T = str(int(args1["pipeline.model.log2_hashmap_total_size"]) - F_log - L_log)
    args1["pipeline.model.log2_hashmap_size"] = T
    del args1["pipeline.model.log2_hashmap_total_size"]
    return args1, args2

def build_command(args1, args2):
    # Convert dictionaries back to command line arguments
    args1_str = ' '.join([f"--{key}={value}" for key, value in args1.items()])
    args2_str = ' '.join([f"--{key}={value}" for key, value in args2.items()])

    command = f"ns-train rfqa-nerfacto --vis=wandb --project_name=radiance-field-qa " \
              f"--experiment_name=lego-grid --viewer.quit_on_train_completion=True " \
              f"--pipeline.model.disable_scene_contraction=True --pipeline.model.use_gradient_scaling=True " \
              f"{args1_str} rfqa-blender-data {args2_str} --scale-factor=1.5 --data ~/Datasets/NeRF/blender/lego"
    return command

def main():
    args1, args2 = parse_arguments(sys.argv[1:])

    command = build_command(args1, args2)

    # Output the parsed arguments and the command
    print("Parsed arguments, running command")
    print(command)

    # Execute the command
    os.system(command)

if __name__ == "__main__":
    main()
