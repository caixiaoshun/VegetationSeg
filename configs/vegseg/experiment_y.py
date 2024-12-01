_base_ = [
    "../_base_/models/convnextv2_femto_vit_segformer_vegseg.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    teach_backbone=dict(
        type="mmpretrain.VisionTransformer",
        arch="large",
        frozen_stages=24,
        img_size=256,
        patch_size=14,
        layer_scale_init_value=1e-5,
        out_indices=(7, 11, 15, 23),
        out_type="featmap",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2-large.pth",
            prefix="backbone",
        ),
    ),
    student_adapter=dict(out_channels=1024),
    decode_head=dict(in_channels=[1024, 1024, 1024, 1024],num_classes=5),
    data_preprocessor=data_preprocessor,
    auxiliary_head=dict(num_classes=5,in_channels=1024),
    # veg_adapter=dict(type="VegAdapter", in_channels=[768, 768, 768, 768]),
)
