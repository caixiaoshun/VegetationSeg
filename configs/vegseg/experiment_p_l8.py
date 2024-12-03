_base_ = [
    "../_base_/models/vegseg.py",
    "../_base_/datasets/cloudsen12_high_l1c.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(512, 512))
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4),
    student_adapter=dict(output_size=37),
    veg_adapter=dict(
        type="VegAdapter",
        in_channels=[768, 768, 768, 768],
        model_type="vitBlock",
        mlp_nums=4,
    ),
)
