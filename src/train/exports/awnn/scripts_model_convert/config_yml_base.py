#!/usr/bin/env python3
import os
import sys
import numpy as np
from acuitylib.vsi_nn import VSInn


class YmlParam:
    def __init__(self):
        self.dataset = []
        self.dataset_type = []
        self.mean = [0, 0, 0]
        self.scale = [1.0, 1.0, 1.0]
        self.reverse_channel = False
        self.add_preproc_node = True
        self.preproc_type = []
        self.add_postproc_node = True

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_dataset_type(self, dataset_type):
        self.dataset_type = dataset_type

    def set_mean(self, mean):
        self.mean = mean

    def set_scale(self, scale):
        self.scale = scale

    def set_reverse_channel(self, reverse_channel):
        self.reverse_channel = reverse_channel

    def set_add_preproc_node(self, add_preproc_node):
        self.add_preproc_node = add_preproc_node

    def set_preproc_type(self, preproc_type):
        self.preproc_type = preproc_type

    def set_add_postproc_node(self, add_postproc_node):
        self.add_postproc_node = add_postproc_node


def configure_model_yml(yml_param, model_name):

    nn = VSInn()
    net = nn.create_net()

    model_filename = model_name
    model_path = f"{model_filename}.json"
    inputmeta_path = f"{model_filename}_inputmeta.yml"
    postprocess_path = f"{model_filename}_postprocess_file.yml"

    if not os.path.exists(model_path):
        print(f"{model_path} file does not exist.")
        sys.exit(1)

    if not os.path.exists(inputmeta_path):
        print(f"{inputmeta_path} file does not exist.")
        sys.exit(1)

    nn.load_model(net, model_path)
    nn.load_model_inputmeta(net, inputmeta_path)

    print()
    inputmeta_data = net.get_input_meta()

    for i, database in enumerate(inputmeta_data.databases):
        port = database.ports[0]
        print(f"Setting input {port.lid} YML parameters:")

        database.path = yml_param.dataset[i]
        database.type = yml_param.dataset_type[i]
        print(f"Dataset path: {database.path}")
        print(f"Dataset type: {database.type}")

        if len(port.shape) == 4:
            channel = port.shape[1] if port.layout == 'nchw' else port.shape[-1]

            if channel in (1, 3, 4):
                port.preprocess['mean'] = yml_param.mean[:channel]
                print(f"Mean: {yml_param.mean[:channel]}")

                scale_val = port.preprocess['scale']
                if isinstance(scale_val, (int, float, np.generic)):
                    # scalar, no len()
                    port.preprocess['scale'] = yml_param.scale[0]
                    print(f"Scale: {yml_param.scale[0]}")
                elif len(scale_val) == channel:
                    port.preprocess['scale'] = yml_param.scale[:channel]
                    print(f"Scale: {yml_param.scale[:channel]}")
                elif len(scale_val) == 1:
                    port.preprocess['scale'] = yml_param.scale[0]
                    print(f"Scale: {yml_param.scale[0]}")


        port.preprocess['reverse_channel'] = yml_param.reverse_channel
        print(f"Reverse channel: {yml_param.reverse_channel}")

        preproc_node_params = port.preprocess['preproc_node_params']

        preproc_node_params['add_preproc_node'] = yml_param.add_preproc_node
        print(f"Add preprocessing node: {yml_param.add_preproc_node}")
        preproc_node_params['preproc_type'] = yml_param.preproc_type[i]
        print(f"Preprocessing type: {yml_param.preproc_type[i]}")
        print()


    net.update_input_meta(inputmeta_data)
    nn.save_model_inputmeta(net, f"{model_filename}_inputmeta.yml")
    print()

    if yml_param.add_postproc_node:
        if os.path.exists(postprocess_path):
            with open(postprocess_path, "r") as f:
                data = f.read()

            data = data.replace("add_postproc_node: false", "add_postproc_node: true")

            with open(postprocess_path, "w") as f:
                f.write(data)

            print("Updated add_postproc_node: false -> true")
        else:
            print(f"Warning: {postprocess_path} does not exist, skipping postproc update")

        print()






