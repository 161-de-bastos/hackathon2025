from .constant import load_training_config, load_distributed_model_config, load_data_loader_config

def load_hparams(distributed_cfg_path = "configs/distributed_model_config.json",
                 training_cfg_path = "configs/training_config.json"):
    train = load_training_config(training_cfg_path)
    dist  = load_distributed_model_config(distributed_cfg_path)
    model_key = "memory_v1"
    model = dist.get(model_key, {})
    return {
        # entrenamiento
        "learning_rate": train.get("learning_rate", 2e-3),
        "batch_size":    train.get("batch_size", 32),
        "epochs":        train.get("epochs", 100),
        "weight_decay":  train.get("weight_decay", 0.0),
        # modelo
        "embedding_dim": model.get("embedding_dim", 100),
        "hidden_dim":    model.get("hidden_dim", 100),
        "dropout":       model.get("dropout", 0.1),
        # strong supervision
        "partial_supervision_info": model.get("partial_supervision_info", {"value": {"flag": False, "coefficient": 1.0, "margin": 0.5}}),
    }

def load_category_from_dataloader(dataloader_cfg_path = "configs/data_loader.json"):
    cfg = load_data_loader_config(dataloader_cfg_path)
    if isinstance(cfg.get("type"), str) and "configs" in cfg:
        t = cfg["type"]
        if t in cfg["configs"]:
            return cfg["configs"][t].get("category", "A")
    return cfg.get("category", "A")

def load_model_and_tokenizer(
    ckpt_path,
    max_len,
    kb_dir="local_database/KB",
    map_location="cpu",
):

    import torch
    from ..lightning import LightMemory
    from ..token import tokenizer_from_state
    from ..dataset import load_kb_bank, load_kb_struct

    device = torch.device(map_location)

    ckpt = torch.load(ckpt_path, map_location=device)
    tok_state = ckpt.get("tokenizer_state")
    if tok_state is None:
        raise RuntimeError("El checkpoint no contiene tokenizer_state. Re-entrena o guarda uno nuevo.")
    tok = tokenizer_from_state(tok_state)

    dummy_ids = {c: torch.zeros(1, max_len, dtype=torch.long) for c in ("A","CH","CR","LTD","TER")}
    dummy_msk = {c: torch.ones(1, dtype=torch.bool) for c in ("A","CH","CR","LTD","TER")}
    lit = LightMemory.load_from_checkpoint(
        ckpt_path,
        vocab_size=tok.vocab_size,
        pad_idx=0,
        hparams=ckpt.get("hyper_parameters", {}).get("hparams", ckpt.get("hyper_parameters", {})),
        kb_ids=dummy_ids,
        kb_mask=dummy_msk,
        map_location=device,
    ).eval()

    kb_ids, kb_mask = load_kb_bank(tok, max_len=max_len, kb_dir=kb_dir)
    for k in kb_ids:
        kb_ids[k] = kb_ids[k].to(next(lit.parameters()).device)
        kb_mask[k] = kb_mask[k].to(next(lit.parameters()).device)
    lit.kb_ids = kb_ids
    lit.kb_mask = kb_mask

    kb_struct = load_kb_struct(kb_dir)  # para racionales con Id/Tag
    return lit, tok, kb_struct