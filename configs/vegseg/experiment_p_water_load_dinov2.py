_base_ = [
    "../_base_/models/vegseg.py",
    "../_base_/datasets/water.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/water_schedule.py",
]

data_preprocessor = dict(size=(512, 512))
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6),
    student_adapter=dict(output_size=37),
    veg_adapter=dict(
        type="VegAdapter",
        in_channels=[768, 768, 768, 768],
        model_type="vitBlock",
        mlp_nums=4,
    ),
)

train_dataloader = dict(batch_size=2,num_workers=2)
val_dataloader = dict(batch_size=2,num_workers=2)
test_dataloader = dict(batch_size=2,num_workers=2)

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
# learning policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=1520*5),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=0.9,
        begin=1520*5,
        end=152000,
        by_epoch=False,
    ),
]
# training schedule for 40k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=152000, val_interval=1520)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=1520, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=1520,
        save_best=["mIoU"],
        rule=["greater"],
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

load_from = "work_dirs/dinov2_upernet_water/best_mIoU_iter_38760.pth"