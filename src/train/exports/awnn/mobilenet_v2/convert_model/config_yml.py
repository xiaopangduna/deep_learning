#!/usr/bin/env python3
import os
import sys
from acuitylib.vsi_nn import VSInn
import numpy as np


# "database" allowed types: "TEXT, NPY, H5FS, SQLITE, LMDB, GENERATOR, ZIP"
DATASET = '../../../quant_data/imagenet_10/dataset.txt'
DATASET_TYPE = "TEXT"

# mean, scale
MEAN    = [123.675, 116.28, 103.53]
SCALE   = [0.0171247, 0.0175070, 0.0174291]

# reverse_channel: True bgr, False rgb
REVERSE_CHANNEL = False

# add_preproc_node, True or False
ADD_PREPROC_NODE = True
# "preproc_type" allowed types:"IMAGE_RGB, IMAGE_RGB888_PLANAR, IMAGE_RGB888_PLANAR_SEP, IMAGE_I420, 
# IMAGE_NV12,IMAGE_NV21, IMAGE_YUV444, IMAGE_YUYV422, IMAGE_UYVY422, IMAGE_GRAY, IMAGE_BGRA, TENSOR"
PREPROC_TYPE = "IMAGE_RGB"

# add_postproc_node, quant output -> float32 output
ADD_POSTPROC_NODE = True


if __name__ == "__main__":

    nn = VSInn()
    net = nn.create_net()

    model_filename = sys.argv[1]
    model = model_filename + ".json"
    inputmeta = model_filename + "_inputmeta.yml"
    postprocess = model_filename + "_postprocess_file.yml"


    if os.path.exists(model) is True:
        nn.load_model(net, model)
    else:
        print("{} file does not exists.".format(model))
        sys.exit(1)

    if os.path.exists(inputmeta) is True:
        nn.load_model_inputmeta(net, inputmeta)
    else:
        print("{} file does not exists.".format(inputmeta))
        sys.exit(1)

    print()
    inputmeta_data = net.get_input_meta()
    port = inputmeta_data.databases[0].ports[0]
    if len(port.shape) == 4:
        if port.layout == 'nchw':
            channel = port.shape[1]
        else:
            channel = port.shape[-1]
        if channel == 3 or channel == 1 or channel == 4:

            port.preprocess['mean'] = MEAN[:channel]
            print("set preprocess param mean " + str(MEAN[:channel]))

            if isinstance(port.preprocess['scale'], (int, float, np.generic)):
                # scalar, no len()
                port.preprocess['scale'] = SCALE[0]
                print("set preprocess param scale " + str(SCALE[0]))
            elif len(port.preprocess['scale']) == channel:
                port.preprocess['scale'] = SCALE[:channel]
                print("set preprocess param scale " + str(SCALE[:channel]))
            elif len(port.preprocess['scale']) == 1:
                port.preprocess['scale'] = SCALE[0]
                print("set preprocess param scale " + str(SCALE[0]))


    nn.set_database(net, dataset_files=DATASET, dataset_type=DATASET_TYPE)
    print("set dataset_files path: " + str(DATASET))
    print("set dataset       type: " + str(DATASET_TYPE))

    port.preprocess['reverse_channel'] = REVERSE_CHANNEL
    print("set reverse_channel " + str(REVERSE_CHANNEL))

    preproc_node_params = port.preprocess['preproc_node_params']

    preproc_node_params['add_preproc_node'] = ADD_PREPROC_NODE
    print("set add_preproc_node " + str(ADD_PREPROC_NODE))
    preproc_node_params['preproc_type'] = PREPROC_TYPE
    print("set preproc_type " + str(PREPROC_TYPE))


    net.update_input_meta(inputmeta_data)
    nn.save_model_inputmeta(net, model_filename + '_inputmeta.yml')


    if (ADD_POSTPROC_NODE == True) :
        with open(postprocess, "r") as f:
            data = f.read()
        data = data.replace("add_postproc_node: false", "add_postproc_node: true")
        with open(postprocess, "w") as f:
            f.write(data)

        print("add_postproc_node: false -> true")



