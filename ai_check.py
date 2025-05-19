import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
import torch.nn.functional as F
from text_dataset import TextDataset
import os
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


class AITextChecker:
    def __init__(self, model_path, optimizer_path, test_path, max_len=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "./roberta-large"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.test_path = test_path
        # 加载模型和优化器的状态字典
        self.test_data = pd.read_csv(self.test_path).dropna(subset=['answer'])
        self.test_data['answer'] = self.test_data['answer'].astype(str)
        self.max_len = max_len
        test_text_list = self.test_data['answer'].tolist()
        test_label_list = self.test_data['label'].tolist()
        test_domain_list = self.test_data['domain'].tolist()
        test_attack_list = self.test_data['attack'].tolist()
        test_model_list = self.test_data['model'].tolist()
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.batch_size = 32
        self.test_dataset = TextDataset(
            test_text_list,
            test_label_list,
            test_domain_list,
            test_attack_list,
            test_model_list,
            self.tokenizer,
            self.max_len
        )

    def _create_test_loader(self, test_path, batch_size=None):
        batch_size = self.batch_size
        test_data = pd.read_csv(test_path).dropna(subset=['answer'])
        test_data['answer'] = test_data['answer'].astype(str)
        test_text_list = test_data['answer'].tolist()
        test_label_list = test_data['label'].tolist()
        test_domain_list = test_data['domain'].tolist()
        test_attack_list = test_data['attack'].tolist()
        test_model_list = test_data['model'].tolist()
        test_dataset = TextDataset(
            test_text_list,
            test_label_list,
            test_domain_list,
            test_attack_list,
            test_model_list,
            self.tokenizer,
            self.max_len
        )
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def _save_results(self, precision, accuracy, recall, f1, domain_acc, attack_acc, save_path):
        """保存测试结果到txt文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Test Parameters:\n")
            f.write(f"- Precision: {precision:.4f}\n")
            f.write(f"- Accuracy: {accuracy:.4f}\n")
            f.write(f"- Recall: {recall:.4f}\n")
            f.write(f"- F1 Score: {f1:.4f}\n\n")

            f.write("Domain Accuracies:\n")
            for domain, acc in domain_acc.items():
                f.write(f"  {domain}: {acc:.4f}\n")

            f.write("\nAttack Type Accuracies:\n")
            for attack, acc in attack_acc.items():
                f.write(f"  {attack}: {acc:.4f}\n")

    def evaluate(self, test_path, save_path):
        test_loader = self._create_test_loader(test_path)
        self.model.eval()
        all_predictions, all_true_labels = [], []
        all_domains = []
        all_attacks = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Validation on {test_path}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if isinstance(outputs, torch.Tensor) else outputs[0]
                all_predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                all_true_labels.extend(batch['labels'].cpu().tolist())
                all_domains.extend(batch['raw_domain'])
                all_attacks.extend(batch['attack'])

        # 移到循环外：统一计算所有batch的结果
        df = pd.DataFrame({
            'domain': all_domains,
            'attack': all_attacks,
            'true': all_true_labels,
            'pred': all_predictions
        })

        # 计算全局指标
        precision = precision_score(df['true'], df['pred'], average='weighted', zero_division=1)
        accuracy = accuracy_score(df['true'], df['pred'])
        recall = recall_score(df['true'], df['pred'], average='weighted', zero_division=1)
        f1 = f1_score(df['true'], df['pred'], average='weighted', zero_division=1)

        # 计算domain和attack的准确率
        domain_acc = {
            domain: accuracy_score(df[df['domain']==domain]['true'], 
                                  df[df['domain']==domain]['pred'])
            for domain in df['domain'].unique()
        }

        attack_acc = {
            attack: accuracy_score(df[df['attack']==attack]['true'],
                                  df[df['attack']==attack]['pred'])
            for attack in df['attack'].unique()
        }

        self._save_results(precision, accuracy, recall, f1, domain_acc, attack_acc, save_path)
        return precision, accuracy, recall, f1, domain_acc, attack_acc


# 使用示例
if __name__ == "__main__":
    model_path = "./cl_roberta_openai"
    optimizer_path = './cl_roberta_openai'
    save_path= "ablation_study_raid_test.txt"
    test_path = './SentEval/data/downstream/AI_detection/raid_test.csv'

    # 创建AITextChecker实例
    classifier = AITextChecker(model_path, optimizer_path, test_path)
    classifier.evaluate(test_path, save_path)
