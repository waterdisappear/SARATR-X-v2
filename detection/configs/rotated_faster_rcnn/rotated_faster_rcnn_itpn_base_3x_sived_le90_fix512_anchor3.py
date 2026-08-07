# 固定训练尺度 512 + RPN ratios 回归基线 的消融版本
_base_ = ['./rotated_faster_rcnn_itpn_base_3x_sived_le90_fix512.py']

model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[0.5, 1.0, 2.0],
        )))

