import os
import sys
import torch
import torchvision
from PIL import Image
import cv2
import numpy as np
import time


IN_HEIGHT = int(os.environ.get("YOLO_IN_HEIGHT", "640"))
IN_WIDTH = int(os.environ.get("YOLO_IN_WIDTH", "640"))
IN_CHANNELS = int(os.environ.get("YOLO_IN_CHANNELS", "3"))


COCO_CLASS_NAMES = [
    'BG',
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def letterbox(im, new_shape=None, color=None, auto=False, scaleFill=False, scaleup=True, stride=32):
    if new_shape is None:
        new_shape = (IN_HEIGHT, IN_WIDTH)
    if color is None:
        color = 114 if IN_CHANNELS == 1 else (114, 114, 114)
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def _read_image(image_file):
    if IN_CHANNELS == 1:
        im = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    else:
        im = cv2.imread(image_file)
    if im is None:
        raise FileNotFoundError(image_file)
    return im


def torch_preprocess(image_file):
    im = _read_image(image_file)

    im, ratio, (dw, dh) = letterbox(im)

    if IN_CHANNELS == 1:
        im = np.expand_dims(im, 0)
    else:
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.expand_dims(im, 0)
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im)
    return im.float() / 255

def image_preprocess(image_file, norm=True):
    im = _read_image(image_file)

    im, ratio, (dw, dh) = letterbox(im)

    if IN_CHANNELS == 1:
        im = np.expand_dims(im, -1)
    else:
        im = im[:, :, ::-1]  # BGR to RGB
    im = np.expand_dims(im, 0)
    im = np.ascontiguousarray(im)
    # im = torch.from_numpy(im)
    if norm:
        return im.astype(np.float32) / 255
    else:
        return im


def torch_postprocess(model_outputs, **kwargs):
    if 'conf_thres' in kwargs and kwargs['conf_thres'] is not None:
        conf_thres = kwargs['conf_thres']
    else:
        conf_thres = 0.25
    if 'iou_thres' in kwargs and kwargs['iou_thres'] is not None:
        iou_thres = kwargs['iou_thres']
    else:
        iou_thres = 0.7

    processed_outputs = non_max_suppression(
        model_outputs,
        conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False)
    return processed_outputs[0]

def sim_postprocess(model_outputs, **kwargs):
    if 'conf_thres' in kwargs and kwargs['conf_thres'] is not None:
        conf_thres = kwargs['conf_thres']
    else:
        conf_thres = 0.25
    if 'iou_thres' in kwargs and kwargs['iou_thres'] is not None:
        iou_thres = kwargs['iou_thres']
    else:
        iou_thres = 0.7
    # model_outputs = model_outputs.cpu().numpy()

    processed_outputs = non_max_suppression(
        torch.from_numpy(model_outputs),
        conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False)
    return processed_outputs[0]


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

def torch_visualize(image_file, processed_outputs):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    detections = processed_outputs.detach().cpu().numpy()
    cats = np.round(detections[:, 5]).astype(int)
    probs = detections[:, 4]

    img_ori = Image.open(image_file).convert('RGB')
    im_width, im_height = img_ori.size
    input_size = 640
    larger_size = max(im_width, im_height)
    scale = larger_size / input_size
    # boxes = np.zeros((detections.shape[0], 4))
    boxes = detections * scale

    plt.figure(figsize=(12.80, 7.20))
    plt.imshow(img_ori)
    for i, box in enumerate(boxes):
        box_x = box[0] - (larger_size - im_width) / 2
        box_y = box[1] - (larger_size - im_height) / 2
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        ax = plt.gca()
        rect = patches.Rectangle(
            (box_x, box_y), box_width, box_height,
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        class_name = COCO_CLASS_NAMES[cats[i] + 1]
        plt.text(box_x, box_y, class_name + '({:.2f})'.format(probs[i]))
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    det_file_name = os.path.basename(image_file).replace(
        '.' + image_file.split('.')[-1],
        '_det.png'
    )
    plt.savefig(os.path.join(output_dir, det_file_name))
    plt.close()