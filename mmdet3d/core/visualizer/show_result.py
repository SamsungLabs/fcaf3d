import mmcv
import numpy as np
from os import path as osp
import matplotlib.pyplot as plt

from .image_vis import (draw_camera_bbox3d_on_img, draw_depth_bbox3d_on_img,
                        draw_lidar_bbox3d_on_img)


def _write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def _write_oriented_bbox(corners, labels, out_filename):
    """Export corners and labels to .obj file for meshlab.

    Args:
        corners(list[ndarray] or ndarray): [B x 8 x 3] corners of
            boxes for each scene
        labels(list[int]): labels of boxes for each scene
        out_filename(str): Filename.
    """
    colors = np.multiply([
        plt.cm.get_cmap('nipy_spectral', 19)((i * 5 + 11) % 18 + 1)[:3] for i in range(18)
    ], 255).astype(np.uint8).tolist()
    with open(out_filename, 'w') as file:
        for i, (corner, label) in enumerate(zip(corners, labels)):
            c = colors[label]
            for p in corner:
                file.write(f'v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n')
            j = i * 8 + 1
            for k in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                      [2, 3, 7, 6], [3, 0, 4, 7], [1, 2, 6, 5]]:
                file.write('f')
                for l in k:
                    file.write(f' {j + l}')
                file.write('\n')
    return


def show_result(points,
                gt_bboxes,
                gt_labels,
                pred_bboxes,
                pred_labels,
                out_dir,
                filename,
                show=True,
                snapshot=False):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool): Visualize the results online. Defaults to False.
        snapshot (bool): Whether to save the online results. Defaults to False.
    """
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:
        from .open3d_vis import Visualizer

        vis = Visualizer(points)
        if pred_bboxes is not None:
            vis.add_bboxes(bbox3d=pred_bboxes)
        if gt_bboxes is not None:
            vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_bboxes is not None:
        _write_oriented_bbox(gt_bboxes, gt_labels,
                             osp.join(result_path, f'{filename}_gt.obj'))

    if pred_bboxes is not None:
        _write_oriented_bbox(pred_bboxes, pred_labels,
                             osp.join(result_path, f'{filename}_pred.obj'))


def show_seg_result(points,
                    gt_seg,
                    pred_seg,
                    out_dir,
                    filename,
                    palette,
                    ignore_index=None,
                    show=False,
                    snapshot=False):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_seg (np.ndarray): Ground truth segmentation mask.
        pred_seg (np.ndarray): Predicted segmentation mask.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        palette (np.ndarray): Mapping between class labels and colors.
        ignore_index (int, optional): The label index to be ignored, e.g. \
            unannotated points. Defaults to None.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results. \
            Defaults to False.
    """
    # we need 3D coordinates to visualize segmentation mask
    if gt_seg is not None or pred_seg is not None:
        assert points is not None, \
            '3D coordinates are required for segmentation visualization'

    # filter out ignored points
    if gt_seg is not None and ignore_index is not None:
        if points is not None:
            points = points[gt_seg != ignore_index]
        if pred_seg is not None:
            pred_seg = pred_seg[gt_seg != ignore_index]
        gt_seg = gt_seg[gt_seg != ignore_index]

    if gt_seg is not None:
        gt_seg_color = palette[gt_seg]
        gt_seg_color = np.concatenate([points[:, :3], gt_seg_color], axis=1)
    if pred_seg is not None:
        pred_seg_color = palette[pred_seg]
        pred_seg_color = np.concatenate([points[:, :3], pred_seg_color],
                                        axis=1)

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    # online visualization of segmentation mask
    # we show three masks in a row, scene_points, gt_mask, pred_mask
    if show:
        from .open3d_vis import Visualizer
        mode = 'xyzrgb' if points.shape[1] == 6 else 'xyz'
        vis = Visualizer(points, mode=mode)
        if gt_seg is not None:
            vis.add_seg_mask(gt_seg_color)
        if pred_seg is not None:
            vis.add_seg_mask(pred_seg_color)
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_seg is not None:
        _write_obj(gt_seg_color, osp.join(result_path, f'{filename}_gt.obj'))

    if pred_seg is not None:
        _write_obj(pred_seg_color, osp.join(result_path,
                                            f'{filename}_pred.obj'))


def show_multi_modality_result(img,
                               gt_bboxes,
                               pred_bboxes,
                               proj_mat,
                               out_dir,
                               filename,
                               box_mode,
                               img_metas=None,
                               show=False,
                               gt_bbox_color=(61, 102, 255),
                               pred_bbox_color=(241, 101, 72)):
    """Convert multi-modality detection results into 2D results.

    Project the predicted 3D bbox to 2D image plane and visualize them.

    Args:
        img (np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str): Coordinate system the boxes are in.
            Should be one of 'depth', 'lidar' and 'camera'.
        img_metas (dict): Used in projecting depth bbox.
        show (bool): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
        pred_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
    """
    if box_mode == 'depth':
        draw_bbox = draw_depth_bbox3d_on_img
    elif box_mode == 'lidar':
        draw_bbox = draw_lidar_bbox3d_on_img
    elif box_mode == 'camera':
        draw_bbox = draw_camera_bbox3d_on_img
    else:
        raise NotImplementedError(f'unsupported box mode {box_mode}')

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:
        show_img = img.copy()
        if gt_bboxes is not None:
            show_img = draw_bbox(
                gt_bboxes, show_img, proj_mat, img_metas, color=gt_bbox_color)
        if pred_bboxes is not None:
            show_img = draw_bbox(
                pred_bboxes,
                show_img,
                proj_mat,
                img_metas,
                color=pred_bbox_color)
        mmcv.imshow(show_img, win_name='project_bbox3d_img', wait_time=0)

    if img is not None:
        mmcv.imwrite(img, osp.join(result_path, f'{filename}_img.png'))

    if gt_bboxes is not None:
        gt_img = draw_bbox(
            gt_bboxes, img, proj_mat, img_metas, color=gt_bbox_color)
        mmcv.imwrite(gt_img, osp.join(result_path, f'{filename}_gt.png'))

    if pred_bboxes is not None:
        pred_img = draw_bbox(
            pred_bboxes, img, proj_mat, img_metas, color=pred_bbox_color)
        mmcv.imwrite(pred_img, osp.join(result_path, f'{filename}_pred.png'))
