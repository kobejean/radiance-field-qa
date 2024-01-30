import sys
import os
import math

def parse_arguments(argv):
    args1 = {}
    args2 = {}

    for arg in argv:
        if arg.startswith('--'):
            key, _, value = arg.lstrip('-').partition('=')
            if key.startswith('pipeline.data_manager') or key.startswith('data'):
                args2[key] = value
            else:
                args1[key] = value
    scene = args1["scene"]
    del args1["scene"]
    F_log = int(math.log2(float(args1["pipeline.model.features_per_level"])))
    L_log = int(math.log2(float(args1["pipeline.model.num_levels"])))
    Z = int(args1["pipeline.model.log2_hashmap_total_size"])
    T = Z - F_log - L_log
    args1["pipeline.model.log2_hashmap_size"] = str(T)
    del args1["pipeline.model.log2_hashmap_total_size"]

    args1["pipeline.model.proposal_net_args_list.0.features_per_level"] = args1["pipeline.model.features_per_level"]
    args1["pipeline.model.proposal_net_args_list.0.num_levels"] = str(int(math.pow(2, L_log-2) + 1))
    args1["pipeline.model.proposal_net_args_list.0.log2_hashmap_size"] = str(int(T - 2))
    args1["pipeline.model.proposal_net_args_list.1.features_per_level"] = args1["pipeline.model.features_per_level"]
    args1["pipeline.model.proposal_net_args_list.1.num_levels"] = str(int(math.pow(2, L_log-2) + 1))
    args1["pipeline.model.proposal_net_args_list.1.log2_hashmap_size"] = str(int(T - 2))
    return scene, args1, args2

def build_command(scene, args1, args2):
    # Convert dictionaries back to command line arguments
    args1_str = ' '.join([f"--{key}={value}" for key, value in args1.items()])
    args2_str = ' '.join([f"--{key}={value}" for key, value in args2.items()])

    command = f"ns-train rfqa-nerfacto --vis=wandb --project_name=radiance-field-qa " \
              f"--experiment_name=lego-grid --viewer.quit_on_train_completion=True " \
              f"--pipeline.model.disable_scene_contraction=True --pipeline.model.use_gradient_scaling=True " \
              f"{args1_str} rfqa-blender-data {args2_str} --scale-factor=1.5 --data ~/Datasets/NeRF/blender/{scene}"
    return command

def main():
    scene, args1, args2 = parse_arguments(sys.argv[1:])

    command = build_command(scene, args1, args2)

    # Output the parsed arguments and the command
    print("Parsed arguments, running command")
    print(command)

    # Execute the command
    os.system(command)

if __name__ == "__main__":
    main()
