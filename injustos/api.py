from .utils.config_loader import load_hparams as _load, load_category_from_dataloader
from .trainer import train_one_from_splits as _train_one_from_splits
from .utils.cv import run_cross_validation as _run_cv

def load_hparams(distributed_cfg_path = "configs/distributed_model_config.json",
                 training_cfg_path = "configs/training_config.json"):
    return _load(distributed_cfg_path, training_cfg_path)

def load_category(data_loader_cfg_path = "configs/data_loader.json"):
    return load_category_from_dataloader(data_loader_cfg_path)

def train_from_csv_splits(
    train_csv,
    val_csv,
    test_csv,
    category,
    hparams,
    out_dir,
    max_len = 128,
    batch_size = 32,
    callbacks_config = None,
):
    return _train_one_from_splits(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        hparams=hparams,
        out_dir=out_dir,
        category=category,
        max_len=max_len,
        batch_size=batch_size,
        callbacks_config=callbacks_config or {},
    )

def cross_validate(
    csv_path,
    category,
    distributed_cfg,
    out_dir,
    k = 5,
    prebuilt_folds_json = None,
    seed = 42,
    max_len = 128,
    batch_size = 32,
):
    return _run_cv(
        csv_path=csv_path,
        category=category,
        distributed_cfg=distributed_cfg,
        out_dir=out_dir,
        k=k,
        prebuilt_folds_json=prebuilt_folds_json,
        seed=seed,
        max_len=max_len,
        batch_size=batch_size,
    )