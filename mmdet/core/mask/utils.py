import mmcv
import numpy as np


def split_combined_polys(polys, poly_lens, polys_per_mask):
    """Split the combined 1-D polys into masks.

    A mask is represented as a list of polys, and a poly is represented as
    a 1-D array. In dataset, all masks are concatenated into a single 1-D
    tensor. Here we need to split the tensor into original representations.

    Args:
        polys (list): a list (length = image num) of 1-D tensors
        poly_lens (list): a list (length = image num) of poly length
        polys_per_mask (list): a list (length = image num) of poly number
            of each mask

    Returns:
        list: a list (length = image num) of list (length = mask num) of
            list (length = poly num) of numpy array
    """
    mask_polys_list = []
    for img_id in range(len(polys)):
        polys_single = polys[img_id]
        polys_lens_single = poly_lens[img_id].tolist()
        polys_per_mask_single = polys_per_mask[img_id].tolist()

        split_polys = mmcv.slice_list(polys_single, polys_lens_single)
        mask_polys = mmcv.slice_list(split_polys, polys_per_mask_single)
        mask_polys_list.append(mask_polys)
    return mask_polys_list


def mask_expand(masks, expanded_h, expanded_w, top, left):
    height = masks.shape[1]
    width = masks.shape[2]
    if len(masks) == 0:
        expanded_mask = np.empty((0, expanded_h, expanded_w),
                                    dtype=np.uint8)
    else:
        expanded_mask = np.zeros((len(masks), expanded_h, expanded_w),
                                    dtype=np.uint8)
        expanded_mask[:, top:top + height,
                        left:left + width] = masks
    return expanded_mask


def mask_crop(masks, bbox):
    assert isinstance(bbox, np.ndarray)
    assert bbox.ndim == 1

    height = masks.shape[1]
    width = masks.shape[2]

    # clip the boundary
    bbox = bbox.copy()
    bbox[0::2] = np.clip(bbox[0::2], 0, width)
    bbox[1::2] = np.clip(bbox[1::2], 0, height)
    x1, y1, x2, y2 = bbox
    w = np.maximum(x2 - x1, 1)
    h = np.maximum(y2 - y1, 1)

    if len(masks) == 0:
        cropped_masks = np.empty((0, h, w), dtype=np.uint8)
    else:
        cropped_masks = masks[:, y1:y1 + h, x1:x1 + w]
    return cropped_masks
