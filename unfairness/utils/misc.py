
def mask_ids(df, ids):
    ids = set(map(str, ids))
    return df[df["document_ID"].astype(str).isin(ids)].reset_index(drop=True)