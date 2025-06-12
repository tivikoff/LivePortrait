import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image

from .io import contiguous
from .crop import crop_image, crop_image_by_bbox, parse_bbox_from_landmark, average_bbox_lst
from .face_analysis_diy import FaceAnalysisDIY
from .human_landmark_runner import LandmarkRunner


def create_crop_helpers(cfg) -> Tuple[FaceAnalysisDIY, LandmarkRunner]:
    """Initialize face detection and landmark helpers."""
    if cfg.flag_force_cpu:
        provider = ["CPUExecutionProvider"]
        onnx_provider = "cpu"
    else:
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            provider = ["CUDAExecutionProvider"]
            onnx_provider = "cuda"
        else:
            provider = ["CPUExecutionProvider"]
            onnx_provider = "cpu"
    face = FaceAnalysisDIY(name="buffalo_l", root=cfg.insightface_root, providers=provider)
    face.prepare(ctx_id=cfg.device_id, det_size=(512, 512), det_thresh=cfg.det_thresh)
    face.warmup()
    landmark = LandmarkRunner(ckpt_path=cfg.landmark_ckpt_path, onnx_provider=onnx_provider, device_id=cfg.device_id)
    landmark.warmup()
    return face, landmark


def _get_landmarks(face: FaceAnalysisDIY, landmark: LandmarkRunner, img: np.ndarray, cfg) -> np.ndarray | None:
    faces = face.get(contiguous(img[..., ::-1]), flag_do_landmark_2d_106=True, direction=cfg.direction, max_face_num=cfg.max_face_num)
    if len(faces) == 0:
        return None
    lmk106 = faces[0].landmark_2d_106
    return landmark.run(img, lmk106)


def crop_source_image(img: np.ndarray, cfg, face: FaceAnalysisDIY, landmark: LandmarkRunner):
    lmk = _get_landmarks(face, landmark, img, cfg)
    if lmk is None:
        return None
    ret = crop_image(
        img,
        lmk,
        dsize=cfg.dsize,
        scale=cfg.scale,
        vx_ratio=cfg.vx_ratio,
        vy_ratio=cfg.vy_ratio,
        flag_do_rot=cfg.flag_do_rot,
    )
    ret["img_crop_256x256"] = cv2.resize(ret["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
    ret["lmk_crop"] = lmk
    ret["lmk_crop_256x256"] = ret["pt_crop"] * 256 / cfg.dsize
    return ret


def crop_source_video(frames: List[np.ndarray], cfg, face: FaceAnalysisDIY, landmark: LandmarkRunner):
    frame_crop_lst, lmk_crop_lst, M_c2o_lst = [], [], []
    for fr in frames:
        ret = crop_source_image(fr, cfg, face, landmark)
        if ret is None:
            continue
        frame_crop_lst.append(ret["img_crop_256x256"])
        lmk_crop_lst.append(ret["lmk_crop_256x256"])
        M_c2o_lst.append(ret["M_c2o"])
    return {"frame_crop_lst": frame_crop_lst, "lmk_crop_lst": lmk_crop_lst, "M_c2o_lst": M_c2o_lst}


def crop_driving_video(frames: List[np.ndarray], cfg, face: FaceAnalysisDIY, landmark: LandmarkRunner):
    lmk_lst, bbox_lst = [], []
    for fr in frames:
        lmk = _get_landmarks(face, landmark, fr, cfg)
        if lmk is None:
            continue
        lmk_lst.append(lmk)
        bbox = parse_bbox_from_landmark(
            lmk,
            scale=cfg.scale_crop_driving_video,
            vx_ratio_crop_driving_video=cfg.vx_ratio_crop_driving_video,
            vy_ratio=cfg.vy_ratio_crop_driving_video,
        )["bbox"]
        bbox_lst.append([bbox[0, 0], bbox[0, 1], bbox[2, 0], bbox[2, 1]])
    global_bbox = average_bbox_lst(bbox_lst)
    frame_crop_lst, lmk_crop_lst = [], []
    for fr, lmk in zip(frames, lmk_lst):
        ret = crop_image_by_bbox(fr, global_bbox, lmk=lmk, dsize=cfg.dsize, flag_rot=False, borderValue=(0, 0, 0))
        frame_crop_lst.append(ret["img_crop"])
        lmk_crop_lst.append(ret["lmk_crop"])
    return {"frame_crop_lst": frame_crop_lst, "lmk_crop_lst": lmk_crop_lst}


def calc_lmk_from_cropped_image(img: np.ndarray, face: FaceAnalysisDIY, landmark: LandmarkRunner, cfg):
    return _get_landmarks(face, landmark, img, cfg)


def calc_lmks_from_cropped_video(frames: List[np.ndarray], face: FaceAnalysisDIY, landmark: LandmarkRunner, cfg):
    return [_get_landmarks(face, landmark, fr, cfg) for fr in frames]
