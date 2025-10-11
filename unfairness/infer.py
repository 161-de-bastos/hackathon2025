import torch

from .utils.constant import KB_CATEGORIES

@torch.no_grad()
def predict_with_rationales(
    lit_module,                       
    dataloader,                       
    kb_texts,              
    threshold = 0.5,           
    top_k = 3,                   
    return_for_all = False,     
    use_scores = True,          
    device = None,
):
    model_device = next(lit_module.parameters()).device
    device = device or model_device

    if isinstance(kb_texts, dict):
        kb_dict = {cat: list(kb_texts.get(cat, [])) for cat in KB_CATEGORIES}
    else:
        shared = list(kb_texts)
        kb_dict = {cat: shared for cat in KB_CATEGORIES}

    if isinstance(threshold, dict):
        thr = {cat: float(threshold.get(cat, 0.5)) for cat in KB_CATEGORIES}
    else:
        thr = {cat: float(threshold) for cat in KB_CATEGORIES}

    if isinstance(top_k, dict):
        tk = {cat: int(top_k.get(cat, 3)) for cat in KB_CATEGORIES}
    else:
        tk = {cat: int(top_k) for cat in KB_CATEGORIES}

    results = []
    for batch in dataloader:
        x = batch["input_ids"].to(device, non_blocking=True)

        out = lit_module(x)
        if isinstance(out, tuple) and len(out) == 3:
            logits, att_dict, scores_dict = out
        elif isinstance(out, tuple) and len(out) == 2:
            logits, att_dict = out
            scores_dict = {cat: att_dict[cat] for cat in KB_CATEGORIES}
        else:
            logits = out
            att_dict = {cat: None for cat in KB_CATEGORIES}
            scores_dict = {cat: None for cat in KB_CATEGORIES}

        probs = logits.sigmoid() 
        preds = (probs > torch.tensor([thr[c] for c in KB_CATEGORIES], device=probs.device)).to(torch.int)

        gold_multi = batch.get("labels_multi", None)
        if gold_multi is not None:
            gold_multi = gold_multi.detach().cpu()

        batch_texts = batch.get("raw_text", None)
        B = x.size(0)
        for i in range(B):
            if isinstance(batch_texts, list) and i < len(batch_texts):
                raw_text = batch_texts[i]
            else:
                raw_text = None

            rec = {
                "text": raw_text,
                "per_category": {},
                "general_pred": int(preds[i].max().item()),
                "general_prob": float(probs[i].max().item()),
            }

            for ci, cat in enumerate(KB_CATEGORIES):
                kb_list = kb_dict.get(cat, [])
                M_kb = len(kb_list)
                att_w = att_dict.get(cat, None)
                raw_s = scores_dict.get(cat, None)

                prob = float(probs[i, ci].item())
                pred = int(preds[i, ci].item())
                gold = None
                if gold_multi is not None:
                    gold = float(gold_multi[i, ci].item())

                cat_entry = {
                    "prob": prob,
                    "pred": pred,
                    "gold": gold,
                    "rationales": [],
                }

                need_rationales = return_for_all or (pred == 1)
                if need_rationales and att_w is not None and raw_s is not None and M_kb > 0:
                    takeM = min(att_w.size(1), M_kb)
                    scores = raw_s[i, :takeM].detach().cpu()
                    weights = att_w[i, :takeM].detach().cpu()
                    order = scores.argsort(descending=True) if use_scores else weights.argsort(descending=True)
                    order = order[: tk[cat]].tolist()

                    rats = []
                    for j in order:
                        rats.append({
                            "idx": int(j),
                            "score_raw": float(scores[j].item()),
                            "att_weight": float(weights[j].item()),
                            "text": kb_list[j],
                        })
                    cat_entry["rationales"] = rats

                rec["per_category"][cat] = cat_entry
            results.append(rec)
    return results