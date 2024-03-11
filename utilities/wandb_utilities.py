from pytorch_lightning.callbacks import ModelCheckpoint

# Setup the model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val/loss',    # Replace 'val_loss' with your validation metric
    dirpath='checkpoints/',  # Directory to save checkpoints
    filename='best-model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',  # 'min' if the metric should decrease (e.g., loss), 'max' for metrics that should increase (e.g., accuracy)
)
