_base_ = [
    "../_base_/models/vegseg.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    data_preprocessor=data_preprocessor,
    student_training=False,
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/dinov2_b_frozen-simpleAdapter/backbone.pth",
            prefix="backbone.",
        ),
    ),
    decode_head=dict(
        num_classes=5,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/dinov2_b_frozen-simpleAdapter/head.pth",
            prefix="decode_head",
        ),
    ),
    student_adapter=dict(
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/dinov2_b_frozen-simpleAdapter/student_adapter.pth",
            prefix="student_adapter",
        ),
    ),
    auxiliary_head=dict(
        num_classes=5,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/dinov2_b_frozen-simpleAdapter/auxiliary_head.pth",
            prefix="auxiliary_head",
        ),
    ),
    veg_adapter=dict(type="VegAdapter", in_channels=[768, 768, 768, 768]),
)
