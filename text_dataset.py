import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, text_list, label_list, domain_list, attack_list, model_list, tokenizer, max_len=512, domain_encoder=None, is_eval=False):
        self.texts = text_list
        self.labels = label_list
        self.tokenizer = tokenizer
        self.domain_encoder = LabelEncoder()
        self.attacks = [str(d).strip() if pd.notnull(d) else "unknown" for d in attack_list]
        self.models = [str(d).strip() if pd.notnull(d) else "unknown" for d in model_list]
        self.raw_domains = [str(d).strip() if pd.notnull(d) else "unknown" for d in domain_list]
        self.max_len = max_len
        self.is_eval = is_eval
        if domain_encoder is None:
            self.domain_encoder = LabelEncoder()
            self.domains = self.domain_encoder.fit_transform([str(d).strip() if pd.notnull(d) else "unknown" for d in domain_list])
        else:
            self.domain_encoder = domain_encoder
            # 处理未知领域为-1
            self.domains = np.array([
                self.domain_encoder.transform([str(d).strip()])[0]
                if str(d).strip() in self.domain_encoder.classes_
                else -1
                for d in domain_list
            ])



    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        domains = self.domains[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long),
            'domain': torch.tensor(domains, dtype=torch.long),
            'attack': self.attacks[idx],
            'model': self.models[idx],
            'raw_domain': self.raw_domains[idx]
        }
