from deep_translator import GoogleTranslator
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import glob

def trans(line,lang='spanish'):
    return GoogleTranslator(source='en', target=lang).translate(line).lower()
    
def load_text(pt):
    with open(pt, 'r') as f:
        text = f.read()
    return text

def sentencizer(txt, threshold = 6):
    idxin, sents = [], []

    for t in txt.split('\n'):
        if len(t.split()) > threshold:
            idxin += [True]
            sents += [t.lower()]
        else:
            idxin += [False]

    return idxin, sents

class exporter:
    def __init__(self,
        input_csv,
        output_csv,
        batch_size,
        distributed = True,
        total_workers = 8,
        wid = 0      
    ):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.batch_size = batch_size
        self.distributed = distributed

        self.__buffer = []
        self.__init_df(total_workers, wid)

    def __worker_csv(self, wid: int) -> str:
        base, ext = os.path.splitext(self.output_csv)
        return f"{base}.w{wid:03d}{ext or '.csv'}"

    def __init_df(self, total_workers, wid):
        self.df_in = pd.read_csv(self.input_csv)

        if self.distributed:
            assert total_workers > wid
            self.output_csv = self.__worker_csv(wid)
            distrib = np.linspace(0,self.df_in.shape[0], total_workers + 1 , dtype = int)
            self.df_in = self.df_in.iloc[distrib[wid]:distrib[wid + 1]]

        self.processed = set()

    def __flush_buffer(self):
        if not self.__buffer:
            return
        out_df = pd.DataFrame(self.__buffer, columns=[
            "i","A","CH","CR","J","LAW","LTD","PINC","TER","USE",
            "document","document_ID","label","text","TER_targets",
            "LTD_targets","A_targets","CH_targets","CR_targets"
        ])
        
        out_df.to_csv(
            self.output_csv,
            index=False,
            mode="a",
            header=not os.path.exists(self.output_csv) or os.path.getsize(self.output_csv) == 0,
            encoding="utf-8",
        )
        self.__buffer.clear()

    def cumulative_export(self):
        for _, row in tqdm(self.df_in.iterrows(), total=len(self.df_in)):
            txt = trans(str(row['text']))
            buff = row.to_dict()
            buff['text'] = txt

            self.__buffer.append(buff)
            if len(self.__buffer) >= self.batch_size:
                self.__flush_buffer()
        self.__flush_buffer()

def merge_workers(dirwids, output_path):
    csvs = glob.glob('*.csv',root_dir=dirwids)
    dfs = pd.concat(
        [pd.read_csv(os.path.join(dirwids,f)) for f in csvs],
        ignore_index=True
    )
    dfs = dfs.drop_duplicates(subset=['i'], keep='last')
    dfs.to_csv(output_path, index=False)