_base_ = './paconv_ssg.py'

model = dict(
    backbone=dict(
        sa_cfg=dict(
            type='PAConvMUSASAModule',
            scorenet_cfg=dict(mlp_channels=[8, 16, 16]))))
