from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np

import matplotlib.pyplot
from visualize_soft_segments import visualize_soft_segments



class Detector():
    
    def __init__(self) -> None:
        self.cfg = get_cfg()
        
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        
        self.cfg.MODEL.ROI_HEADS.CORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"
        
        self.predictor = DefaultPredictor(self.cfg)
    
    def on_image(self, image_path):
        image = cv2.imread(image_path)
        predictions = self.predictor(image)
        
        viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        
        cv2.imshow("result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)


if __name__ == '__main__':
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.DEVICE = "cpu"
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    image = cv2.imread('images/sss_03.png')
    sss = np.load('npy/sss_03.npy')
    
    outputs = predictor(image)
    classes = outputs['instances'].pred_classes
    masks = outputs['instances'].pred_masks
    
    person_args = np.argwhere(classes.cpu() == 0)[0]
    person_masks = masks[person_args].cpu().numpy()
     
    sss = np.where(sss < 0.5, 0., 1.)
    n_classes = sss.shape[-1]
    
    tmp_mask = person_masks[0] * 1.0
    tmp_mask_3c = np.tile(np.expand_dims(tmp_mask, -1), [1, 1, n_classes])
    iou_mask =  np.logical_and(tmp_mask_3c, sss)

    iou_points = np.sum(iou_mask.reshape(-1, n_classes), axis=0)
    print(iou_points)
    iou_max = np.argmax(iou_points)

    tmp_mask_er = cv2.erode(tmp_mask, np.ones((3, 3)))
    cv2.imshow('Person Mask', tmp_mask)
    cv2.waitKey()
    
    cv2.imshow('Person Mask Erode', tmp_mask_er)
    cv2.waitKey()
    
    soft_mask = sss[:, :, iou_max]
    soft_mask = np.logical_or(soft_mask, tmp_mask_er)
    # soft_mask = tmp_mask
    soft_mask = np.stack((soft_mask, soft_mask, soft_mask), axis=-1)
    BG_COLOR = (0, 255, 255)
    res = np.zeros(soft_mask.shape, dtype='uint8')
    for i in range(3):
        tmp = np.where(soft_mask[:, :, i]==1, image[:, :, i], BG_COLOR[i])
        res[:, :, i] = tmp
    cv2.imshow('', res)
    cv2.waitKey()
    
    # cv2.imwrite('results/final/sss_04.png', res)    