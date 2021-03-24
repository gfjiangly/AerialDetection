import cvtools

from .coco import CocoDataset


class HRSCDataset(CocoDataset):

    CLASSES = ('ship', )

    def __init__(self,
                ann_file,
                img_prefix,
                img_scale,
                img_norm_cfg,
                multiscale_mode='value',
                size_divisor=None,
                proposal_file=None,
                num_max_proposals=1000,
                flip_ratio=0,
                with_mask=True,
                with_crowd=True,
                with_label=True,
                with_semantic_seg=False,
                seg_prefix=None,
                seg_scale_factor=1,
                extra_aug=None,
                rotate_aug=None,
                rotate_test_aug=None,
                resize_keep_ratio=True,
                test_mode=False,
                level='L1',
                L2_class_map_file=None,
                L3_class_map_file=None):
        super(HRSCDataset, self).__init__(
                ann_file=ann_file,
                img_prefix=img_prefix,
                img_scale=img_scale,
                img_norm_cfg=img_norm_cfg,
                multiscale_mode=multiscale_mode,
                size_divisor=size_divisor,
                proposal_file=proposal_file,
                num_max_proposals=num_max_proposals,
                flip_ratio=flip_ratio,
                with_mask=with_mask,
                with_crowd=with_crowd,
                with_label=with_label,
                with_semantic_seg=with_semantic_seg,
                seg_prefix=seg_prefix,
                seg_scale_factor=seg_scale_factor,
                extra_aug=extra_aug,
                rotate_aug=rotate_aug,
                rotate_test_aug=rotate_test_aug,
                resize_keep_ratio=resize_keep_ratio,
                test_mode=test_mode)
        self.level = level
        self.L2_class_map = None
        self.L3_class_map = None
        self.L2_cat2label = None
        self.L3_cat2label = None
        if level == 'L1':
            for cat in self.cat2label.keys():
                self.cat2label[cat] = 1
        elif level == 'L2':
            self.L2_class_map = cvtools.read_key_value(L2_class_map_file)
            self.L2_cat2label = {
                cat_id: i + 1
                for i, cat_id in enumerate(set(self.L2_class_map.values()))
            }
            ori_to_L2 = dict()
            for cat, cat_id in self.L2_class_map.items():
                ori_to_L2[cat] = self.L2_cat2label[cat_id]
            self.cat2label = ori_to_L2
        elif level == 'L3':
            self.L3_class_map = cvtools.read_key_value(L3_class_map_file)
            self.L3_cat2label = {
                cat_id: i + 1
                for i, cat_id in enumerate(set(self.L3_class_map.values()))
            }
            ori_to_L3 = dict()
            for cat, cat_id in self.L3_class_map.items():
                ori_to_L3[cat] = self.L3_cat2label[cat_id]
            self.cat2label = ori_to_L3


class HRSCL1Dataset(CocoDataset):

    CLASSES = ('ship', )
  

class HRSCL2Dataset(CocoDataset):

    CLASSES = ('ship',  'aircraft carrier', 
    'warcraft', 'merchant ship')


class HRSCL3Dataset(CocoDataset):

    CLASSES = ('ship', )