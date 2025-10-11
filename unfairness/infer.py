import torch

from .utils.constant import KB_CATEGORIES

@torch.no_grad()
def predict_with_rationales(
    lit_module,                       
    dataloader,                       
    kb_struct_or_texts,              
    threshold = 0.5,           
    top_k = 3,                       
    use_scores = True,          
    device = None
):
    model_device = next(lit_module.parameters()).device
    device = device or model_device

    is_struct = isinstance(kb_struct_or_texts, dict) and all(
        isinstance(kb_struct_or_texts.get(k, {}), dict) for k in KB_CATEGORIES
    )
    if is_struct:
        kb_texts = {cat: kb_struct_or_texts[cat]["text"] for cat in KB_CATEGORIES}
    else:
        kb_texts = {cat: kb_struct_or_texts for cat in KB_CATEGORIES}

    out = []
    for batch in dataloader:
        x = batch["input_ids"].to(device, non_blocking=True)
        logits, att_w_dict, raw_s_dict = lit_module(x)  # dict por cat
        probs_multi = logits.sigmoid()
        preds_multi = (probs_multi > (threshold if isinstance(threshold, (int,float)) else 0.5)).to(torch.int)

        if isinstance(threshold, dict):
            thr_vec = torch.tensor([threshold[c] for c in KB_CATEGORIES], device=probs_multi.device)
            preds_multi = (probs_multi > thr_vec.unsqueeze(0)).to(torch.int)

        B = x.size(0)
        for i in range(B):
            rec = {
                "text": batch.get("raw_text", [""]*B)[i],
                "general_prob": float(probs_multi[i].max().item()),
                "general_pred": int(preds_multi[i].max().item()),
                "per_category": {},
            }
            for c_idx, cat in enumerate(KB_CATEGORIES):
                prob = float(probs_multi[i, c_idx].item())
                pred = int(preds_multi[i, c_idx].item())
                rats = []
                if att_w_dict is not None and raw_s_dict is not None:
                    scores  = raw_s_dict[cat][i].detach().cpu()
                    weights = att_w_dict[cat][i].detach().cpu()
                    order = scores.argsort(descending=True) if use_scores else weights.argsort(descending=True)
                    order = order[:top_k].tolist()
                    kb_list = kb_texts[cat]
                    for j in order:
                        item = {
                            "idx": int(j),
                            "score_raw": float(scores[j].item()),
                            "att_weight": float(weights[j].item()),
                            "text": kb_list[j] if isinstance(kb_list, list) else kb_list.iloc[j],
                        }
                        if is_struct:
                            item["id"]  = kb_struct_or_texts[cat]["id"][j]
                            item["tag"] = kb_struct_or_texts[cat]["tag"][j]
                        rats.append(item)

                rec["per_category"][cat] = {"prob": prob, "pred": pred, "rationales": rats}
            out.append(rec)
    return out