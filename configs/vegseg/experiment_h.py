_base_ = [
    "../_base_/models/vegseg.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    data_preprocessor=data_preprocessor,
    fuse=True,
    neck=dict(in_channels=[768], scales=[1]),
    decode_head=dict(num_classes=5,in_channels=[768],in_index=[0]),
    auxiliary_head=dict(num_classes=5,in_index=0),
    veg_adapter=dict(type="VegAdapter", in_channels=[768]),
)
