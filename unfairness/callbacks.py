from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def get_early_stopping_callback(monitor="val_f1_score", mode="max", patience=20, min_delta=0.0):
    return EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        verbose=False
    )

def get_model_checkpoint_callback(monitor="val_f1_score", mode="max", save_top_k=1, dirpath="cv_test/", filename_prefix="best"):
    filename = f"{filename_prefix}-{{epoch:02d}}-{{{monitor}:.4f}}"
    return ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        dirpath=dirpath,
        filename=filename
    )