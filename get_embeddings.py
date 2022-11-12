import argparse
import os
import sys
from pathlib import Path
from itertools import chain 

from tqdm import tqdm
import torch
import chroma_client

from utils.metrics import box_iou

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from models.yolo import Detect
from utils.dataloaders import create_imageloader, create_dataloader
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_yaml,
                           non_max_suppression, print_args, scale_boxes, xywh2xyxy)
from utils.torch_utils import select_device, smart_inference_mode

# Associate detections to labels
def associate_preds_labels(preds, targets, im, shapes, iou_thres=0.5):
    all_associations = []
    for i, pred in enumerate(preds):
        labels = targets[targets[:,0]==i][:,1:]
        
        if len(labels) == 0:
            all_associations.append([None]*len(pred))
            continue
        
        if len(pred) == 0:
            all_associations.append([])
            continue

        scale_boxes(im[i].shape[1:], pred[:, :4], shapes[i][0], shapes[i][1])
        tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
        scale_boxes(im[i].shape[1:], tbox, shapes[i][0], shapes[i][1]) 
        labels = torch.cat((labels[:, 0:1], tbox), 1)
        
        # Get the predictions with the greatest IOU for each label
        ious = box_iou(labels[:, 1:], pred[:, :4])
        best_preds = ious.argmax(1)
        best_ious = ious.max(1)[0]

        # If two or more labels have the same best prediction, choose the one with the greatest IOU
        for best_pred in best_preds.unique():
            equal = best_preds == best_pred 
            if equal.sum() > 1:
                best_preds[equal] = -1
                (best_iou, best_idx) = best_ious[equal].max(dim=0)
                best_ious[equal] = -1
                best_ious[equal][best_idx] = best_iou
                best_preds[equal][best_idx] = best_pred

        # Filter out the predictions with IOU < iou_thres
        best_preds[best_ious < iou_thres] = -1
        best_preds = best_preds.tolist()
        best_preds_set = set(best_preds)

        # If a prediction is the best match for a given label, associate the label to the prediction, otherwise associate None
        all_associations.append([labels[best_preds.index(i), 0] if i in best_preds_set else None for i in range(pred.shape[0])])

    return all_associations

def process_batch(im, targets, shapes, paths, cuda, device, half, model, augment, conf_thres, iou_thres, classes, agnostic_nms, max_det, names, chroma, dataset_name, dt):
    # Image Transform
    with dt[0]:
        if cuda:
            im = im.to(device, non_blocking=True)
            if targets is not None:
                targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width


    # Inference
    with dt[1]:
        preds = model(im, augment=augment)

    # NMS
    with dt[2]:
        preds, embeddings = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, with_embeddings=True, max_det=max_det)        

    # Log embeddings to Chroma
    with dt[3]:
        # Get the inference class names
        inference_class_names = [names[int(p[5])] for p in torch.cat(preds,0)]    

        # Get associated label class names
        label_class_names = None
        if targets is not None:        
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
            associations = associate_preds_labels(preds, targets, im, shapes)                 
            label_class_names = [names[int(a)] if a is not None else None for a in chain.from_iterable(associations)]

        # Set the path to be the same for detections from the same image 
        paths = [p for i, path in enumerate(paths) for p in [path]*len(preds[i])]

        chroma.log(embedding_data=torch.cat(embeddings,0).tolist(),input_uri=paths,inference_category_name=inference_class_names,label_category_name=label_class_names,dataset=dataset_name)

    # Print profiling times
    LOGGER.info(f'Image Transform time: {dt[0].dt:.3f}s, Inference time: {dt[1].dt:.3f}s, NMS time: {dt[2].dt:.3f}s, Chroma time: {dt[3].dt:.3f}s')

@smart_inference_mode()
def run(
            data,
            dataset='val',  # train, val, or test
            dataset_name="",
            weights=None,  # model.pt path(s)
            batch_size=32,  # batch size
            imgsz=640,  # inference size (pixels)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IoU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            workers=8,  # max dataloader workers (per RANK in DDP mode)
            augment=False,  # augmented inference
            half=True,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            with_labels=False # Associate label class 
):
    # Initialize/load model and set device
    device = select_device(device, batch_size=batch_size)

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half = model.fp16  # FP16 supported on limited backends with CUDA
    if engine:
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

    data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'

    # Set with embedding output 
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.with_embeddings = True


    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    dataloader = create_dataloader(data[dataset], 
                                    imgsz, 
                                    batch_size, 
                                    stride, 
                                    workers=workers)[0] if with_labels else create_imageloader(data[dataset],
                                    imgsz,
                                    batch_size,
                                    stride,
                                    workers=workers)[0]

    # Chroma client
    chroma = chroma_client.Chroma()

    # Class names
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    
    dt = Profile(), Profile(), Profile(), Profile()  # profiling times
    
    # Infer with labels and associate
    if with_labels:
        for im, targets, paths, shapes in tqdm(dataloader):
            process_batch(im, targets, shapes, paths, cuda, device, half, model, augment, conf_thres, iou_thres, classes, agnostic_nms, max_det, names, chroma, dataset_name, dt)
    
    # Don't do label association
    else:
        for paths, im in tqdm(dataloader):
            process_batch(im, None, None, paths, cuda, device, half, model, augment, conf_thres, iou_thres, classes, agnostic_nms, max_det, names, chroma, dataset_name, dt)

    return


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--dataset-name', type=str, default="", help='dataset name for chroma storage')
    parser.add_argument('--dataset', default='val', help='train, val, or test')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--with-labels', action='store_true', help='Associate labels to get label class')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
