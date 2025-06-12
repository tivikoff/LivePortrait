import cv2
import numpy as np
from PIL import Image
from typing import List

from .crop import crop_image, crop_image_by_bbox, parse_bbox_from_landmark, average_bbox_lst
from .animal_landmark_runner import XPoseRunner


_DEF_MAP = {'animal_face_9': 'animal_face', 'animal_face_68': 'face'}


def _get_landmarks(runner: XPoseRunner, img: np.ndarray, face_type: str):
    return runner.run(Image.fromarray(img), 'face', _DEF_MAP[face_type], box_threshold=0.0, IoU_threshold=0.0)


def crop_source_image(img: np.ndarray, cfg, runner: XPoseRunner):
    lmk = _get_landmarks(runner, img, cfg.animal_face_type)
    ret = crop_image(
        img,
        lmk,
        dsize=cfg.dsize,
        scale=cfg.scale,
        vx_ratio=cfg.vx_ratio,
        vy_ratio=cfg.vy_ratio,
        flag_do_rot=cfg.flag_do_rot,
    )
    ret['img_crop_256x256'] = cv2.resize(ret['img_crop'], (256, 256), interpolation=cv2.INTER_AREA)
    ret['lmk_crop'] = lmk
    ret['lmk_crop_256x256'] = ret['pt_crop'] * 256 / cfg.dsize
    return ret


def crop_source_video(frames: List[np.ndarray], cfg, runner: XPoseRunner):
    frame_crop_lst, lmk_crop_lst, M_c2o_lst = [], [], []
    for fr in frames:
        ret = crop_source_image(fr, cfg, runner)
        frame_crop_lst.append(ret['img_crop_256x256'])
        lmk_crop_lst.append(ret['lmk_crop_256x256'])
        M_c2o_lst.append(ret['M_c2o'])
    return {'frame_crop_lst': frame_crop_lst, 'lmk_crop_lst': lmk_crop_lst, 'M_c2o_lst': M_c2o_lst}


def crop_driving_video(frames: List[np.ndarray], cfg, runner: XPoseRunner):
    lmk_lst, bbox_lst = [], []
    for fr in frames:
        lmk = _get_landmarks(runner, fr, cfg.animal_face_type)
        lmk_lst.append(lmk)
        bbox = parse_bbox_from_landmark(
            lmk,
            scale=cfg.scale_crop_driving_video,
            vx_ratio_crop_driving_video=cfg.vx_ratio_crop_driving_video,
            vy_ratio=cfg.vy_ratio_crop_driving_video,
        )["bbox"]
        bbox_lst.append([bbox[0,0], bbox[0,1], bbox[2,0], bbox[2,1]])
    global_bbox = average_bbox_lst(bbox_lst)
    frame_crop_lst, lmk_crop_lst = [], []
    for fr, lmk in zip(frames, lmk_lst):
        ret = crop_image_by_bbox(fr, global_bbox, lmk=lmk, dsize=cfg.dsize, flag_rot=False, borderValue=(0,0,0))
        frame_crop_lst.append(ret['img_crop'])
        lmk_crop_lst.append(ret['lmk_crop'])
    return {'frame_crop_lst': frame_crop_lst, 'lmk_crop_lst': lmk_crop_lst}


def calc_lmk_from_cropped_image(img: np.ndarray, runner: XPoseRunner, face_type: str):
    return runner.run(Image.fromarray(img), 'face', _DEF_MAP[face_type], box_threshold=0.0, IoU_threshold=0.0)


def calc_lmks_from_cropped_video(frames: List[np.ndarray], runner: XPoseRunner, face_type: str):
    return [runner.run(Image.fromarray(fr), 'face', _DEF_MAP[face_type], box_threshold=0.0, IoU_threshold=0.0) for fr in frames]
