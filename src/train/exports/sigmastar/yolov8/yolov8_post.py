import os

from calibrator_custom import sgs_chalk
import numpy as np

NUM_CLASSES = int(os.environ.get("YOLO_NUM_CLASSES", "80"))
IN_WIDTH = int(os.environ.get("YOLO_IN_WIDTH", "640"))
IN_HEIGHT = int(os.environ.get("YOLO_IN_HEIGHT", "640"))
LAYER_STRIDES=[8,16,32]
LAYER_GRIDS=[ int((IN_WIDTH / stride) * (IN_HEIGHT / stride)) for stride in LAYER_STRIDES]
ALL_LAYER_GRIDS=0
for r_grid in LAYER_GRIDS:
    ALL_LAYER_GRIDS += r_grid
CONF_THRESHOLD=0.3

def buildGraph(model_config):
    inputs = sgs_chalk.Input(shape=model_config['shape'], name=model_config['input'][0])
    [bboxes, score] = sgs_chalk.Split_V(inputs, [4, model_config['num_classes']], axis=1, name=['box', 'confidence'])

    [cx, cy, w, h] = sgs_chalk.Split_V(bboxes, [1, 1, 1, 1], axis=1, name=['cx', 'cy', 'w', 'h'])
    w_ = sgs_chalk.Mul(w, 0.5)
    h_ = sgs_chalk.Mul(h, 0.5)

    x1 = sgs_chalk.Sub(cx, w_)
    y1 = sgs_chalk.Sub(cy, h_)
    x2 = sgs_chalk.Add(cx, w_)
    y2 = sgs_chalk.Add(cy, h_)

    bboxes_new = sgs_chalk.Concatenation([x1,y1,x2,y2],axis=1, name='boxes')

    score = sgs_chalk.Transpose(score,(0,2,1),name='transpose')
    [max_score, max_index] = sgs_chalk.TopK(score,k=1, name=['max_score', 'classes'])

    reshape_score = sgs_chalk.Reshape(max_score, [1, max_score.shape[1]], name='scores')

    score_index, box_num = sgs_chalk.Score_Filter(reshape_score,threshold=CONF_THRESHOLD, input_element_count=ALL_LAYER_GRIDS,name=['score_index','box_number'])

    outfilename = model_config["name"] + "_postprocess.sim"
    model = sgs_chalk.Model([inputs], [bboxes_new,max_index,reshape_score,score_index,box_num],model_config["name"])
    model.save(outfilename)

    print("\nWell Done! " + outfilename  + " generated!\n")
    return outfilename

def get_postprocess():
    model_config = {
        "name":"yolov8n_ipu_decode",
        "input" : ['output0'],
        'shape': [1, 4 + NUM_CLASSES, ALL_LAYER_GRIDS],
        'num_classes': NUM_CLASSES,
    }

    outfilename = buildGraph(model_config)

    return outfilename

def model_postprocess():
    return get_postprocess()

# if __name__ == '__main__':
#     main()