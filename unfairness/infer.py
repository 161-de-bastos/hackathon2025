import torch

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

    out = []
    for batch in dataloader:
        x = batch["input_ids"].to(device, non_blocking=True)
        logits, att_w, raw_s = lit_module(x)  
        probs = logits.sigmoid()              
        preds = (probs > threshold).to(torch.int)  

        gold = batch.get("labels", None)
        if gold is not None:
            gold = gold.detach().cpu()

        B = x.size(0)
        M = att_w.size(1) if att_w is not None else 0

        kbN = len(kb_texts)
        takeM = min(M, kbN)

        for i in range(B):
            rec = {
                "prob": float(probs[i].item()),
                "pred": int(preds[i].item()),
                "gold": (float(gold[i].item()) if gold is not None else None),
                "rationales": [],
            }

            need_rationales = return_for_all or (rec["pred"] == 1)
            if need_rationales and takeM > 0:
                scores = raw_s[i, :takeM].detach().cpu()
                weights = att_w[i, :takeM].detach().cpu()

                order = scores.argsort(descending=True) if use_scores else weights.argsort(descending=True)
                order = order[:top_k].tolist()

                rats = []
                for j in order:
                    rats.append({
                        "idx": int(j),
                        "score_raw": float(scores[j].item()),
                        "att_weight": float(weights[j].item()),
                        "text": kb_texts[j],
                    })
                rec["rationales"] = rats

            out.append(rec)

    return out