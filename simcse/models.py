import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
import logging
import numpy as np
import transformers
import pandas as pd
import os
import re
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup, \
    BertTokenizer, BertForSequenceClassification, XLMRobertaTokenizerFast, LongformerConfig, LongformerForSequenceClassification, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, XLMRobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from torch.utils.data import DataLoader
from text_dataset import TextDataset
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
logger = logging.getLogger(__name__)
import signal
import sys


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


def cl_forward(self,
               encoder,
               input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               labels=None,  # 下游二分类标签（0/1）
               output_attentions=None,
               output_hidden_states=None,
               return_dict=None,
               mlm_input_ids=None,
               mlm_labels=None):
    from transformers.modeling_outputs import SequenceClassifierOutput
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # 如果输入为二维 (batch, seq_len)，扩展为 (batch, 1, seq_len)
    if input_ids.dim() == 2:
        input_ids = input_ids.unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.unsqueeze(1)

    # 强制截断，确保最大长度不超过 512
    max_length = 512
    if input_ids.size(-1) > max_length:
        input_ids = input_ids[..., :max_length]
        attention_mask = attention_mask[..., :max_length]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[..., :max_length]

    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input: (batch_size * num_sent, seq_len)
    input_ids = input_ids.view(-1, input_ids.size(-1))
    attention_mask = attention_mask.view(-1, attention_mask.size(-1))
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if self.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view(-1, mlm_input_ids.size(-1))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling: 恢复为 (batch_size, num_sent, hidden)
    pooler_output = self.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view(batch_size, num_sent, pooler_output.size(-1))

    if self.model_args.pooler_type == "cls":
        pooler_output = self.mlp(pooler_output)

    # 判断分支：如果处于 eval 模式或 finetuning，则走分类分支
    if (not self.training) or self.finetuning:
        # logger.info("Entering classification branch. finetuning={}, training={}".format(self.finetuning, self.training))
        pooled = pooler_output[:, 0]
        # logger.info("Pooled representation stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        #     pooled.min().item(), pooled.max().item(), pooled.mean().item()))
        cls_logits = self.classifier(pooled)
        # print("cls_logits:",cls_logits)
        # print("labels:",labels)
        loss_fct = nn.CrossEntropyLoss()
        if labels is not None:
            labels = labels.view(-1)
            loss = loss_fct(cls_logits, labels)
            # logger.info("Classification loss: {:.4f}".format(loss.item()))
        else:
            # print("label is NONE!")
            loss = torch.tensor(0.0, device=cls_logits.device, requires_grad=True)
        # for name, param in self.classifier.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        #     if param.grad is not None:
        #         logger.info("Gradient norm for {}: {:.4f}".format(name, param.grad.norm().item()))
        #     else:
        #         logger.info("No gradient for parameter: {}".format(name))
        return SequenceClassifierOutput(
            loss=loss,
            logits=cls_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    else:
        # logger.info("Entering contrastive learning branch. training mode.")
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
        if num_sent == 3:
            z3 = pooler_output[:, 2]
        # 此处省略分布式训练相关代码
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        if num_sent >= 3:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
            z3_weight = self.model_args.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                        z1_z3_cos.size(-1) - i - 1)
                 for i in range(z1_z3_cos.size(-1))]
            ).to(self.classifier.weight.device)
            cos_sim = cos_sim + weights

        loss_fct = nn.CrossEntropyLoss()
        sim_labels = torch.arange(cos_sim.size(0)).long().to(self.classifier.weight.device)
        contrast_loss = loss_fct(cos_sim, sim_labels)
        pooled = pooler_output[:, 0]
        pooled = self.dropout(pooled)
        cls_logits = self.classifier(pooled)
        # print("contrasive_cls_logits:",cls_logits)
        # print("contrasive_labels:",sim_labels)
        classification_loss = loss_fct(cls_logits, labels.view(-1)) if labels is not None else 0.0
        total_loss = contrast_loss + self.model_args.cls_weight * classification_loss
        if mlm_outputs is not None and mlm_labels is not None:
            # 将 mlm_labels 展平
            mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            # 通过 lm_head 计算预测分数
            prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
            # 计算 MLM 损失
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1)
            )
            # 根据权重累加到总损失
            total_loss = total_loss + self.model_args.mlm_weight * masked_lm_loss
        # 返回结果
        if not return_dict:
            output = (cos_sim,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        return SequenceClassifierOutput(
            loss=total_loss,
            logits=cls_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Fallback return，如果所有分支都没有执行（理论上不该发生）
    logger.error("cl_forward reached end without returning output. Returning default output.")
    return SequenceClassifierOutput(
        loss=torch.tensor(0.0, device=input_ids.device),
        logits=torch.zeros(batch_size, 2, device=input_ids.device),
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
        cls,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.model_args.num_labels = 2
        self.model_args.cls_weight = 5
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        self.finetuning = False
        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                sent_emb=False,
                mlm_input_ids=None,
                mlm_labels=None,
                ):
        # if sent_emb:
        #     return sentemb_forward(self, self.bert,
        #                            input_ids=input_ids,
        #                            attention_mask=attention_mask,
        #                            token_type_ids=token_type_ids,
        #                            position_ids=position_ids,
        #                            head_mask=head_mask,
        #                            inputs_embeds=inputs_embeds,
        #                            labels=labels,
        #                            output_attentions=output_attentions,
        #                            output_hidden_states=output_hidden_states,
        #                            return_dict=return_dict,
        #                            )
        # else:
        return cl_forward(self, self.bert,
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          position_ids=position_ids,
                          head_mask=head_mask,
                          inputs_embeds=inputs_embeds,
                          labels=labels,
                          output_attentions=output_attentions,
                          output_hidden_states=output_hidden_states,
                          return_dict=return_dict,
                          mlm_input_ids=mlm_input_ids,
                          mlm_labels=mlm_labels,
                          )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.model_args.num_labels = 2
        self.model_args.cls_weight = 5
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        self.finetuning = False
        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                sent_emb=False,
                mlm_input_ids=None,
                mlm_labels=None,
                ):
        # print("Forward received labels:", labels)
        # if sent_emb:
        #     print('sentem_forward!!!!!!')
        #     return sentemb_forward(self, self.roberta,
        #                            input_ids=input_ids,
        #                            attention_mask=attention_mask,
        #                            token_type_ids=token_type_ids,
        #                            position_ids=position_ids,
        #                            head_mask=head_mask,
        #                            inputs_embeds=inputs_embeds,
        #                            labels=labels,
        #                            output_attentions=output_attentions,
        #                            output_hidden_states=output_hidden_states,
        #                            return_dict=return_dict,
        #                            )
        # else:
        # print('cl_forward!!!!!!')
        # print("Calling cl_forward with labels:", labels)
        return cl_forward(self, self.roberta,
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          position_ids=position_ids,
                          head_mask=head_mask,
                          inputs_embeds=inputs_embeds,
                          labels=labels,
                          output_attentions=output_attentions,
                          output_hidden_states=output_hidden_states,
                          return_dict=return_dict,
                          mlm_input_ids=mlm_input_ids,
                          mlm_labels=mlm_labels,
                          )


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def list_factory():
    return defaultdict(list)


class RoBERTaTextTrainer:
    def __init__(self, train_path, val_paths,
                 model_path='./output/roberta_model.pth',
                 optimizer_path='./output/roberta_optimizer.pth',
                 max_len=512,
                 batch_size=64,
                 output_dir='./results',
                 ):
        self.accuracy_history = []
        self.precision_history = []
        self.recall_history = []
        self.f1_history = []
        self.train_path = train_path
        self.val_paths = val_paths
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # 加载数据
        self.train_data = pd.read_csv(self.train_path).dropna(subset=['answer'])
        self.train_data['answer'] = self.train_data['answer'].astype(str)
        self.metrics_history = self._init_metrics_history()
        train_text_list = self.train_data['answer'].tolist()
        train_label_list = self.train_data['label'].tolist()
        train_domain_list = self.train_data['domain'].tolist()
        train_attack_list = self.train_data['attack'].tolist()
        train_model_list = self.train_data['model'].tolist()
        # val_text_list = self.val_data['answer'].tolist()
        # val_label_list = self.val_data['label'].tolist()
        model_id = "./output_directory_ours"
        # 初始化tokenizer和数据集
        self.tokenizer = RobertaTokenizer.from_pretrained(model_id)
        self.train_dataset = TextDataset(
            train_text_list,
            train_label_list,
            train_domain_list,
            train_attack_list,
            train_model_list,
            self.tokenizer,
            self.max_len
        )
        # self.val_dataset = TextDataset(val_text_list, val_label_list, self.tokenizer, self.max_len)

        # 数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
                                       pin_memory=True, persistent_workers=True)
        # self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # 初始化模型和优化器
        self.model = RobertaForSequenceClassification.from_pretrained(model_id, num_labels=2).to(self.device)
        self.num_domains = len(self.train_data['domain'].unique())  # 自动获取领域数量
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_domains)
        ).to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)  # 支持多GPU
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化器
            torch.backends.cudnn.enabled = True  # 启用cudnn加速
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        self.amp_dtype = torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        # 获取优化器和学习率调度器
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler()

        # 类别不平衡：加权损失函数
        class_weights = self.get_class_weights()
        self.loss_fn = CrossEntropyLoss(weight=class_weights.to(self.device))
        self.checkpoint_path = os.path.join(self.output_dir, 'training_checkpoint.pth')
        self._register_signal_handlers()
        self.best_precision = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.dann_alpha = 0.5  # 梯度反转强度
        self.dann_weight = 0.3  # 对抗损失权重

        # 添加领域分类器

    def _register_signal_handlers(self):
        """注册Unix系统信号处理器"""
        if sys.platform != 'win32':  # Windows不支持SIGHUP
            signal.signal(signal.SIGHUP, self._graceful_exit)
        signal.signal(signal.SIGINT, self._graceful_exit)
        signal.signal(signal.SIGTERM, self._graceful_exit)

    def _graceful_exit(self, signum, frame):
        """优雅退出处理"""
        print(f"\n捕获到中断信号 {signum}, 正在保存检查点...")
        self._save_checkpoint(
            epoch=self.current_epoch,
            batch_idx=self.current_batch,
            best_precision=self.best_precision,
            patience_counter=self.patience_counter
        )
        sys.exit(1)

    def _init_metrics_history(self):
        """初始化指标历史数据结构"""
        return {
            val_path: {
                'overall': defaultdict(list),
                'domains': defaultdict(list_factory),
                'attacks': defaultdict(list_factory),
                'models': defaultdict(list_factory)
            } for val_path in self.val_paths
        }

    @staticmethod
    def _create_domain_dict():
        """可序列化的domain字典工厂函数"""
        return {'accuracy': []}

    @staticmethod
    def _create_attack_dict():
        """可序列化的attack字典工厂函数"""
        return {'accuracy': []}

    @staticmethod
    def _create_model_dict():
        """可序列化的attack字典工厂函数"""
        return {'accuracy': []}

    def _check_numpy_availability(self):
        """检查numpy可用性"""
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError("Numpy is required but not available. Please install numpy first.")

    def _save_checkpoint(self, epoch, batch_idx, best_precision, patience_counter):
        """改进的检查点保存方法"""

        # 转换defaultdict为普通dict以便序列化
        def convert_defaultdict(d):
            if isinstance(d, defaultdict):
                return {k: convert_defaultdict(v) for k, v in d.items()}
            return d

        temp_path = self.checkpoint_path + ".tmp"
        checkpoint_metrics = {}
        for val_path, metrics in self.metrics_history.items():
            checkpoint_metrics[val_path] = {
                'overall': convert_defaultdict(metrics['overall']),
                'domains': convert_defaultdict(metrics['domains']),
                'attacks': convert_defaultdict(metrics['attacks']),
                'models': convert_defaultdict(metrics['models'])
            }

        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_precision': self.best_precision,
            'patience_counter': self.patience_counter,
            'metrics_history': checkpoint_metrics
            # 使用转换后的可序列化数据
        }

        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, self.checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
        except Exception as e:
            print(f"Failed to save checkpoint: {str(e)}")
            raise

    def _load_checkpoint(self):
        """修正后的检查点加载方法"""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # 1. 模型参数加载与验证
            state_dict = checkpoint['model_state']
            # 处理多GPU参数名
            if not isinstance(self.model, torch.nn.DataParallel) and any(k.startswith('module.') for k in state_dict):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            elif isinstance(self.model, torch.nn.DataParallel) and not any(k.startswith('module.') for k in state_dict):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}

            # 严格加载并打印警告
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=True)
            if missing_keys:
                logger.warning(f"Missing keys when loading model: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")

            # 2. 优化器状态设备迁移
            optimizer_state = checkpoint['optimizer_state']
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.optimizer.load_state_dict(optimizer_state)

            # 3. 学习率调度器完整恢复
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.scheduler.optimizer = self.optimizer  # 必须重新绑定优化器

            # 4. 指标历史数据结构转换

            def recursive_convert(d):
                if isinstance(d, dict):
                    return defaultdict(list_factory, {k: recursive_convert(v) for k, v in d.items()})
                return d

            loaded_metrics = {}
            for val_path, metrics in checkpoint['metrics_history'].items():
                loaded_metrics[val_path] = {
                    'overall': defaultdict(list, metrics['overall']),
                    'domains': recursive_convert(metrics['domains']),
                    'attacks': recursive_convert(metrics['attacks']),
                    'models': recursive_convert(metrics['models'])
                }

            return (
                checkpoint.get('epoch', 0),
                checkpoint.get('batch_idx', 0),
                checkpoint.get('best_precision', 0.0),
                checkpoint.get('patience_counter', 0),
                loaded_metrics
            )

        return 0, 0, 0.0, 0, None

    def _create_val_loader(self, val_path):
        val_data = pd.read_csv(val_path).dropna(subset=['answer'])
        val_data['answer'] = val_data['answer'].astype(str)
        val_text_list = val_data['answer'].tolist()
        val_label_list = val_data['label'].tolist()
        val_domain_list = val_data['domain'].tolist()
        # val_domain_list = val_data['raw_domain'].apply(
        #     lambda x: x if x in self.train_dataset.domain_encoder.classes_ else "unknown"
        # )
        # 直接使用已扩展的encoder
        # val_domain_encoded = self.train_dataset.domain_encoder.transform(val_domain_list)
        val_attack_list = val_data['attack'].tolist()
        val_model_list = val_data['model'].tolist()
        val_dataset = TextDataset(
            val_text_list,
            val_label_list,
            val_domain_list,
            val_attack_list,
            val_model_list,
            self.tokenizer,
            self.max_len,
            domain_encoder=self.train_dataset.domain_encoder
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # 获取优化器和调度器
    def get_optimizer_and_scheduler(self, lr=2e-5, warmup_ratio=0.1):
        optimizer = AdamW([
            {'params': self.model.parameters()},
            {'params': self.domain_classifier.parameters(), 'lr': 1e-4}
        ], lr=lr, weight_decay=0.01)
        total_steps = len(self.train_loader) * 10
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.3
        )
        return optimizer, scheduler

    # 计算类别权重
    def get_class_weights(self):
        labels = self.train_data['label'].values
        labels = labels.astype(int)
        class_sample_count = np.bincount(labels)  # 获取每个类别的样本数量
        class_sample_count = np.where(class_sample_count == 0, 1e-6, class_sample_count)  # 防止除0错误
        weight = 1.0 / class_sample_count  # 样本少的类别权重大
        return torch.tensor(weight, dtype=torch.float)  # 直接返回类别权重

    def _save_epoch_results(self, val_path, epoch, overall_metrics, domain_acc, attack_acc, model_acc):
        """保存单个epoch的评估结果到txt文件"""
        safe_val_name = self._get_safe_filename(val_path)
        file_path = os.path.join(self.output_dir, f"{safe_val_name}_metrics.txt")

        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"=== Epoch {epoch + 1} ===\n")
                f.write(f"[Overall]\n")
                f.write(f"Accuracy: {overall_metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {overall_metrics['precision']:.4f}\n")
                f.write(f"Recall: {overall_metrics['recall']:.4f}\n")
                f.write(f"F1-Score: {overall_metrics['f1']:.4f}\n\n")

                f.write(f"[Domain Accuracies]\n")
                for domain, acc in sorted(domain_acc.items()):
                    acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
                    f.write(f"- {domain}: {acc_str}\n")
                f.write("\n" + "=" * 40 + "\n\n")
                f.write(f"[Attack Accuracies]\n")
                for attack_type, acc in sorted(attack_acc.items()):
                    acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
                    f.write(f"- {attack_type}: {acc_str}\n")
                f.write("\n" + "=" * 40 + "\n\n")
                f.write(f"[Model Accuracies]\n")
                for model_name, acc in sorted(model_acc.items()):
                    acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
                    f.write(f"- {model_name}: {acc_str}\n")
        except Exception as e:
            print(f"无法保存评估结果到 {file_path}: {str(e)}")

    def _save_final_results(self, val_path):
        """保存最终评估结果到独立文件"""
        safe_val_name = self._get_safe_filename(val_path)
        final_path = os.path.join(self.output_dir, f"{safe_val_name}_final.txt")
        metrics = self.metrics_history[val_path]

        try:
            with open(final_path, 'w', encoding='utf-8') as f:
                # 整体指标
                f.write("=== Final Evaluation Results ===\n")
                f.write(f"[Overall Metrics History]\n")
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    values = metrics['overall'][metric]
                    f.write(f"{metric.capitalize()}: {[round(v, 4) for v in values]}\n")

                # 各domain最终准确率
                f.write("\n[Final Domain Accuracies]\n")
                for domain, data in metrics['domains'].items():
                    final_acc = data['accuracy'][-1] if len(data['accuracy']) > 0 else np.nan
                    acc_str = f"{final_acc:.4f}" if not np.isnan(final_acc) else "N/A"
                    f.write(f"- {domain}: {acc_str}\n")
                f.write("\n[Final Attack Accuracies]\n")
                for attack_type, data in metrics['attacks'].items():
                    final_acc = data['accuracy'][-1] if len(data['accuracy']) > 0 else np.nan
                    acc_str = f"{final_acc:.4f}" if not np.isnan(final_acc) else "N/A"
                    f.write(f"- {attack_type}: {acc_str}\n")
                f.write("\n[Final Model Accuracies]\n")
                for model_name, data in metrics['models'].items():
                    final_acc = data['accuracy'][-1] if len(data['accuracy']) > 0 else np.nan
                    acc_str = f"{final_acc:.4f}" if not np.isnan(final_acc) else "N/A"
                    f.write(f"- {model_name}: {acc_str}\n")
                # 添加统计信息
                f.write("\n[Statistics]\n")
                f.write(f"Total Epochs: {len(metrics['overall']['accuracy'])}\n")
                f.write(f"Domains Count: {len(metrics['domains'])}\n")
                f.write(f"Attacks Count: {len(metrics['attacks'])}\n")
                f.write(f"Models Count: {len(metrics['models'])}\n")
        except Exception as e:
            print(f"无法保存最终结果到 {final_path}: {str(e)}")

    # 训练函数
    def _get_safe_filename(self, path):
        """生成安全的文件名"""
        base_name = os.path.basename(path)
        return re.sub(r'[\\/*?:"<>|]', "_", base_name.split('.')[0])

    def _compute_dann_loss(self, features, domains):
        """计算领域对抗损失"""
        # 梯度反转
        reversed_features = GradientReversalFunction.apply(features, self.dann_alpha)
        domain_logits = self.domain_classifier(reversed_features)
        return F.cross_entropy(domain_logits, domains)

    def train(self, num_epochs=10, early_stopping_patience=3):
        start_epoch, start_batch, self.best_precision, self.patience_counter, loaded_metrics = self._load_checkpoint()
        if loaded_metrics:
            self.metrics_history = loaded_metrics
            print(f"Resuming training from epoch {start_epoch + 1}")
        try:
            for epoch in range(start_epoch, num_epochs):
                self.current_epoch = epoch
                try:  # 训练阶段
                    self.model.train()
                    total_loss = 0
                    loader_enumerator = enumerate(tqdm(self.train_loader,
                                                       desc=f'Training Epoch {epoch + 1}/{num_epochs}',
                                                       initial=start_batch if epoch == start_epoch else 0))
                    for batch_idx, batch in loader_enumerator:
                        self.current_batch = batch_idx
                        if epoch == start_epoch and batch_idx < start_batch:
                            continue
                        self.optimizer.zero_grad()
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        domains = batch['domain'].to(self.device)
                        with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=True):
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
                            features = outputs.hidden_states[-1][:, 0, :]
                            logits = outputs.logits
                            main_loss = outputs.loss if outputs.loss is not None else self.loss_fn(logits, labels)
                            domain_loss = self._compute_dann_loss(features, domains)
                            # 总损失
                            total_loss = main_loss + self.dann_weight * domain_loss

                        self.scaler.scale(total_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                        total_loss += total_loss.item()
                        if batch_idx % 1000 == 0:
                            self._save_checkpoint(
                                epoch=epoch,
                                batch_idx=self.current_batch,
                                best_precision=self.best_precision,
                                patience_counter=self.patience_counter
                            )
                    average_loss = total_loss / len(self.train_loader)
                    print(f'Epoch {epoch + 1}, Loss: {average_loss:.4f}')
                    start_batch = 0
                    # 更新学习率
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.scheduler.step()

                    # 验证阶段
                    for val_path in self.val_paths:
                        precision, accuracy, recall, f1, domain_acc, attack_acc, model_acc = self.evaluate(
                            val_path=val_path)
                        print(f'Validation on {val_path} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
                              f'Recall: {recall:.4f}, F1: {f1:.4f}')

                        # 将当前 epoch 的指标添加到历史记录中
                        self.metrics_history[val_path]['overall']['accuracy'].append(accuracy)
                        self.metrics_history[val_path]['overall']['precision'].append(precision)
                        self.metrics_history[val_path]['overall']['recall'].append(recall)
                        self.metrics_history[val_path]['overall']['f1'].append(f1)

                        for domain, acc in domain_acc.items():
                            self.metrics_history[val_path]['domains'][domain]['accuracy'].append(acc)
                        for attack_type, acc in attack_acc.items():
                            self.metrics_history[val_path]['attacks'][attack_type]['accuracy'].append(acc)
                        for model_name, acc in model_acc.items():
                            self.metrics_history[val_path]['models'][model_name]['accuracy'].append(acc)
                        # 早停机制
                        if precision > self.best_precision:
                            self.best_precision = precision
                            self.patience_counter = 0
                            # 保存模型和优化器状态
                            torch.save(self.model.state_dict(), self.model_path)
                            torch.save(self.optimizer.state_dict(), self.optimizer_path)
                            print(f'Model and optimizer saved with precision {self.best_precision:.4f}')
                        else:
                            self.patience_counter += 1
                            if self.patience_counter >= early_stopping_patience:
                                print(f'Early stopping triggered at epoch {epoch + 1}')
                                break
                        self._save_epoch_results(
                            val_path=val_path,
                            epoch=epoch,
                            overall_metrics={
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            },
                            domain_acc=domain_acc,
                            attack_acc=attack_acc,
                            model_acc=model_acc
                        )
                except Exception as e:
                    print(f"Error occurred at epoch {epoch}: {str(e)}")
                    self._save_checkpoint(
                        epoch=self.current_epoch,
                        batch_idx=self.current_batch,
                        best_precision=self.best_precision,  # 传递实例变量
                        patience_counter=self.patience_counter
                    )
                    raise  # 重新抛出异常以便上层处理
        finally:  # 训练结束后清理检查点
            for val_path in self.val_paths:
                self._save_final_results(val_path)
            # 在所有 epoch 训练完成后绘制指标变化图
            self.plot_metrics()

    def evaluate(self, val_path):
        val_loader = self._create_val_loader(val_path)
        self.model.eval()
        all_predictions, all_true_labels = [], []
        all_domains = []
        all_attacks = []
        all_models = []
        # domain_names = []
        # for d in all_domains:
        #     if d == -1:  # 未知领域
        #         domain_names.append("unknown")
        #     else:
        #         domain_names.append(self.train_dataset.domain_encoder.inverse_transform([d])[0])
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation on {val_path}', leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if isinstance(outputs, torch.Tensor) else outputs[0]
                all_predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                all_true_labels.extend(batch['labels'].cpu().tolist())
                # all_domains.extend(batch['domain'])
                all_attacks.extend(batch['attack'])
                all_models.extend(batch['model'])
                all_domains.extend(batch['raw_domain'])
        precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=1)
        accuracy = accuracy_score(all_true_labels, all_predictions)
        recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=1)
        f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=1)
        df = pd.DataFrame({
            'domain': all_domains,
            'attack': all_attacks,
            'model': all_models,
            'true': all_true_labels,
            'pred': all_predictions,
        })
        domain_acc = {}
        attack_acc = {}
        model_acc = {}
        for domain in df['domain'].unique():
            domain_df = df[df['domain'] == domain]
            domain_acc[domain] = accuracy_score(domain_df['true'], domain_df['pred'])
        for attack_type in df['attack'].unique():
            attack_df = df[df['attack'] == attack_type]
            attack_acc[attack_type] = accuracy_score(attack_df['true'], attack_df['pred'])
        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            model_acc[model_name] = accuracy_score(model_df['true'], model_df['pred'])
        return precision, accuracy, recall, f1, domain_acc, attack_acc, model_acc

    def plot_metrics(self):
        """绘制每个验证数据集的指标变化曲线"""
        for val_path, metrics in self.metrics_history.items():
            plt.figure(figsize=(12, 6))
            epochs = range(1, len(metrics['overall']['accuracy']) + 1)

            # 绘制四条曲线
            plt.plot(epochs, metrics['overall']['accuracy'], 'b-', label='Accuracy')
            plt.plot(epochs, metrics['overall']['precision'], 'r--', label='Precision')
            plt.plot(epochs, metrics['overall']['recall'], 'g-.', label='Recall')
            plt.plot(epochs, metrics['overall']['f1'], 'm:', label='F1-Score')

            # 添加图表元素
            plt.title(f'Model Performance on {val_path} Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # 保存并显示图表
            plt.savefig(f'training_metrics_{val_path.split("/")[-1].split(".")[0]}.png')
            plt.show()
            for domain, data in metrics['domains'].items():
                plt.plot(epochs, data['accuracy'], linestyle='--',
                         label=f'Domain {domain} Accuracy')
            plt.legend()
            safe_name = os.path.splitext(os.path.basename(val_path))[0]
            plt.savefig(f'overall_metrics_{safe_name}.png')
            for attack_type, data in metrics['attacks'].items():
                plt.plot(epochs, data['accuracy'], linestyle='--',
                         label=f'Attack {attack_type} Accuracy')
            plt.legend()
            safe_name = os.path.splitext(os.path.basename(val_path))[0]
            plt.savefig(f'attack_metrics_{safe_name}.png')
            for model_name, data in metrics['models'].items():
                plt.plot(epochs, data['accuracy'], linestyle='--',
                         label=f'Model {model_name} Accuracy')
            plt.legend()
            safe_name = os.path.splitext(os.path.basename(val_path))[0]
            plt.savefig(f'model_metrics_{safe_name}.png')
            # 新增domain准确率柱状图
            plt.figure(figsize=(10, 5))
            domains = list(metrics['domains'].keys())
            final_acc = [data['accuracy'][-1] for data in metrics['domains'].values()]
            plt.bar(domains, final_acc, color=['blue', 'green', 'red'])
            plt.ylim(0, 1)
            plt.title(f'Final Domain Accuracy - {safe_name}')
            plt.savefig(f'domain_acc_{safe_name}.png')

            plt.figure(figsize=(20, 6))
            attacks = list(metrics['attacks'].keys())
            final_acc = [data['accuracy'][-1] for data in metrics['attacks'].values()]
            plt.bar(attacks, final_acc, color=['blue', 'green', 'red'])
            plt.ylim(0, 1)
            plt.title(f'Final Attack Accuracy - {safe_name}')
            plt.savefig(f'attack_acc_{safe_name}.png')

            plt.figure(figsize=(20, 6))
            models = list(metrics['models'].keys())
            final_acc = [data['accuracy'][-1] for data in metrics['models'].values()]
            plt.bar(models, final_acc, color=['blue', 'green', 'red'])
            plt.ylim(0, 1)
            plt.title(f'Final Model Accuracy - {safe_name}')
            plt.savefig(f'model_acc_{safe_name}.png')



class BERTTextTrainer:
    def __init__(self, train_path, val_paths,
                 model_path='./output/bert_model.pth',
                 optimizer_path='./output/bert_optimizer.pth',
                 max_len=512,
                 batch_size=64,
                 output_dir='./results',
                 ):
        self.accuracy_history = []
        self.precision_history = []
        self.recall_history = []
        self.f1_history = []
        self.train_path = train_path
        self.val_paths = val_paths
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # 加载数据
        self.train_data = pd.read_csv(self.train_path).dropna(subset=['answer'])
        self.train_data['answer'] = self.train_data['answer'].astype(str)
        self.metrics_history = self._init_metrics_history()
        train_text_list = self.train_data['answer'].tolist()
        train_label_list = self.train_data['label'].tolist()
        train_domain_list = self.train_data['domain'].tolist()
        train_attack_list = self.train_data['attack'].tolist()
        train_model_list = self.train_data['model'].tolist()
        # val_text_list = self.val_data['answer'].tolist()
        # val_label_list = self.val_data['label'].tolist()
        model_id = "./output_directory_bert"
        # 初始化tokenizer和数据集
        self.tokenizer = BertTokenizer.from_pretrained(model_id)
        self.train_dataset = TextDataset(
            train_text_list,
            train_label_list,
            train_domain_list,
            train_attack_list,
            train_model_list,
            self.tokenizer,
            self.max_len
        )
        # self.val_dataset = TextDataset(val_text_list, val_label_list, self.tokenizer, self.max_len)

        # 数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
                                       pin_memory=True, persistent_workers=True)
        # self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # 初始化模型和优化器
        self.model = BertForSequenceClassification.from_pretrained(model_id, num_labels=2).to(self.device)
        self.num_domains = len(self.train_data['domain'].unique())  # 自动获取领域数量
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_domains)
        ).to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)  # 支持多GPU
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化器
            torch.backends.cudnn.enabled = True  # 启用cudnn加速
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        self.amp_dtype = torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        # 获取优化器和学习率调度器
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler()

        # 类别不平衡：加权损失函数
        class_weights = self.get_class_weights()
        self.loss_fn = CrossEntropyLoss(weight=class_weights.to(self.device))
        self.checkpoint_path = os.path.join(self.output_dir, 'training_checkpoint.pth')
        self._register_signal_handlers()
        self.best_precision = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.dann_alpha = 0.5  # 梯度反转强度
        self.dann_weight = 0.3  # 对抗损失权重

        # 添加领域分类器

    def _register_signal_handlers(self):
        """注册Unix系统信号处理器"""
        if sys.platform != 'win32':  # Windows不支持SIGHUP
            signal.signal(signal.SIGHUP, self._graceful_exit)
        signal.signal(signal.SIGINT, self._graceful_exit)
        signal.signal(signal.SIGTERM, self._graceful_exit)

    def _graceful_exit(self, signum, frame):
        """优雅退出处理"""
        print(f"\n捕获到中断信号 {signum}, 正在保存检查点...")
        self._save_checkpoint(
            epoch=self.current_epoch,
            batch_idx=self.current_batch,
            best_precision=self.best_precision,
            patience_counter=self.patience_counter
        )
        sys.exit(1)

    def _init_metrics_history(self):
        """初始化指标历史数据结构"""
        return {
            val_path: {
                'overall': defaultdict(list),
                'domains': defaultdict(list_factory),
                'attacks': defaultdict(list_factory),
                'models': defaultdict(list_factory)
            } for val_path in self.val_paths
        }

    @staticmethod
    def _create_domain_dict():
        """可序列化的domain字典工厂函数"""
        return {'accuracy': []}

    @staticmethod
    def _create_attack_dict():
        """可序列化的attack字典工厂函数"""
        return {'accuracy': []}

    @staticmethod
    def _create_model_dict():
        """可序列化的attack字典工厂函数"""
        return {'accuracy': []}

    def _check_numpy_availability(self):
        """检查numpy可用性"""
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError("Numpy is required but not available. Please install numpy first.")

    def _save_checkpoint(self, epoch, batch_idx, best_precision, patience_counter):
        """改进的检查点保存方法"""

        # 转换defaultdict为普通dict以便序列化
        def convert_defaultdict(d):
            if isinstance(d, defaultdict):
                return {k: convert_defaultdict(v) for k, v in d.items()}
            return d

        temp_path = self.checkpoint_path + ".tmp"
        checkpoint_metrics = {}
        for val_path, metrics in self.metrics_history.items():
            checkpoint_metrics[val_path] = {
                'overall': convert_defaultdict(metrics['overall']),
                'domains': convert_defaultdict(metrics['domains']),
                'attacks': convert_defaultdict(metrics['attacks']),
                'models': convert_defaultdict(metrics['models'])
            }

        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_precision': self.best_precision,
            'patience_counter': self.patience_counter,
            'metrics_history': checkpoint_metrics
            # 使用转换后的可序列化数据
        }

        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, self.checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
        except Exception as e:
            print(f"Failed to save checkpoint: {str(e)}")
            raise

    def _load_checkpoint(self):
        """修正后的检查点加载方法"""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # 1. 模型参数加载与验证
            state_dict = checkpoint['model_state']
            # 处理多GPU参数名
            if not isinstance(self.model, torch.nn.DataParallel) and any(k.startswith('module.') for k in state_dict):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            elif isinstance(self.model, torch.nn.DataParallel) and not any(k.startswith('module.') for k in state_dict):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}

            # 严格加载并打印警告
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=True)
            if missing_keys:
                logger.warning(f"Missing keys when loading model: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")

            # 2. 优化器状态设备迁移
            optimizer_state = checkpoint['optimizer_state']
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.optimizer.load_state_dict(optimizer_state)

            # 3. 学习率调度器完整恢复
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.scheduler.optimizer = self.optimizer  # 必须重新绑定优化器

            # 4. 指标历史数据结构转换

            def recursive_convert(d):
                if isinstance(d, dict):
                    return defaultdict(list_factory, {k: recursive_convert(v) for k, v in d.items()})
                return d

            loaded_metrics = {}
            for val_path, metrics in checkpoint['metrics_history'].items():
                loaded_metrics[val_path] = {
                    'overall': defaultdict(list, metrics['overall']),
                    'domains': recursive_convert(metrics['domains']),
                    'attacks': recursive_convert(metrics['attacks']),
                    'models': recursive_convert(metrics['models'])
                }

            return (
                checkpoint.get('epoch', 0),
                checkpoint.get('batch_idx', 0),
                checkpoint.get('best_precision', 0.0),
                checkpoint.get('patience_counter', 0),
                loaded_metrics
            )

        return 0, 0, 0.0, 0, None

    def _create_val_loader(self, val_path):
        val_data = pd.read_csv(val_path).dropna(subset=['answer'])
        val_data['answer'] = val_data['answer'].astype(str)
        val_text_list = val_data['answer'].tolist()
        val_label_list = val_data['label'].tolist()
        val_domain_list = val_data['domain'].tolist()
        # val_domain_list = val_data['raw_domain'].apply(
        #     lambda x: x if x in self.train_dataset.domain_encoder.classes_ else "unknown"
        # )
        # 直接使用已扩展的encoder
        # val_domain_encoded = self.train_dataset.domain_encoder.transform(val_domain_list)
        val_attack_list = val_data['attack'].tolist()
        val_model_list = val_data['model'].tolist()
        val_dataset = TextDataset(
            val_text_list,
            val_label_list,
            val_domain_list,
            val_attack_list,
            val_model_list,
            self.tokenizer,
            self.max_len,
            domain_encoder=self.train_dataset.domain_encoder
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # 获取优化器和调度器
    def get_optimizer_and_scheduler(self, lr=2e-5, warmup_ratio=0.1):
        optimizer = AdamW([
            {'params': self.model.parameters()},
            {'params': self.domain_classifier.parameters(), 'lr': 1e-4}
        ], lr=lr, weight_decay=0.01)
        total_steps = len(self.train_loader) * 10
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.3
        )
        return optimizer, scheduler

    # 计算类别权重
    def get_class_weights(self):
        labels = self.train_data['label'].values
        labels = labels.astype(int)
        class_sample_count = np.bincount(labels)  # 获取每个类别的样本数量
        class_sample_count = np.where(class_sample_count == 0, 1e-6, class_sample_count)  # 防止除0错误
        weight = 1.0 / class_sample_count  # 样本少的类别权重大
        return torch.tensor(weight, dtype=torch.float)  # 直接返回类别权重

    def _save_epoch_results(self, val_path, epoch, overall_metrics, domain_acc, attack_acc, model_acc):
        """保存单个epoch的评估结果到txt文件"""
        safe_val_name = self._get_safe_filename(val_path)
        file_path = os.path.join(self.output_dir, f"{safe_val_name}_metrics.txt")

        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"=== Epoch {epoch + 1} ===\n")
                f.write(f"[Overall]\n")
                f.write(f"Accuracy: {overall_metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {overall_metrics['precision']:.4f}\n")
                f.write(f"Recall: {overall_metrics['recall']:.4f}\n")
                f.write(f"F1-Score: {overall_metrics['f1']:.4f}\n\n")

                f.write(f"[Domain Accuracies]\n")
                for domain, acc in sorted(domain_acc.items()):
                    acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
                    f.write(f"- {domain}: {acc_str}\n")
                f.write("\n" + "=" * 40 + "\n\n")
                f.write(f"[Attack Accuracies]\n")
                for attack_type, acc in sorted(attack_acc.items()):
                    acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
                    f.write(f"- {attack_type}: {acc_str}\n")
                f.write("\n" + "=" * 40 + "\n\n")
                f.write(f"[Model Accuracies]\n")
                for model_name, acc in sorted(model_acc.items()):
                    acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
                    f.write(f"- {model_name}: {acc_str}\n")
        except Exception as e:
            print(f"无法保存评估结果到 {file_path}: {str(e)}")

    def _save_final_results(self, val_path):
        """保存最终评估结果到独立文件"""
        safe_val_name = self._get_safe_filename(val_path)
        final_path = os.path.join(self.output_dir, f"{safe_val_name}_final.txt")
        metrics = self.metrics_history[val_path]

        try:
            with open(final_path, 'w', encoding='utf-8') as f:
                # 整体指标
                f.write("=== Final Evaluation Results ===\n")
                f.write(f"[Overall Metrics History]\n")
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    values = metrics['overall'][metric]
                    f.write(f"{metric.capitalize()}: {[round(v, 4) for v in values]}\n")

                # 各domain最终准确率
                f.write("\n[Final Domain Accuracies]\n")
                for domain, data in metrics['domains'].items():
                    final_acc = data['accuracy'][-1] if len(data['accuracy']) > 0 else np.nan
                    acc_str = f"{final_acc:.4f}" if not np.isnan(final_acc) else "N/A"
                    f.write(f"- {domain}: {acc_str}\n")
                f.write("\n[Final Attack Accuracies]\n")
                for attack_type, data in metrics['attacks'].items():
                    final_acc = data['accuracy'][-1] if len(data['accuracy']) > 0 else np.nan
                    acc_str = f"{final_acc:.4f}" if not np.isnan(final_acc) else "N/A"
                    f.write(f"- {attack_type}: {acc_str}\n")
                f.write("\n[Final Model Accuracies]\n")
                for model_name, data in metrics['models'].items():
                    final_acc = data['accuracy'][-1] if len(data['accuracy']) > 0 else np.nan
                    acc_str = f"{final_acc:.4f}" if not np.isnan(final_acc) else "N/A"
                    f.write(f"- {model_name}: {acc_str}\n")
                # 添加统计信息
                f.write("\n[Statistics]\n")
                f.write(f"Total Epochs: {len(metrics['overall']['accuracy'])}\n")
                f.write(f"Domains Count: {len(metrics['domains'])}\n")
                f.write(f"Attacks Count: {len(metrics['attacks'])}\n")
                f.write(f"Models Count: {len(metrics['models'])}\n")
        except Exception as e:
            print(f"无法保存最终结果到 {final_path}: {str(e)}")

    # 训练函数
    def _get_safe_filename(self, path):
        """生成安全的文件名"""
        base_name = os.path.basename(path)
        return re.sub(r'[\\/*?:"<>|]', "_", base_name.split('.')[0])

    def _compute_dann_loss(self, features, domains):
        """计算领域对抗损失"""
        # 梯度反转
        reversed_features = GradientReversalFunction.apply(features, self.dann_alpha)
        domain_logits = self.domain_classifier(reversed_features)
        return F.cross_entropy(domain_logits, domains)

    def train(self, num_epochs=10, early_stopping_patience=3):
        start_epoch, start_batch, self.best_precision, self.patience_counter, loaded_metrics = self._load_checkpoint()
        if loaded_metrics:
            self.metrics_history = loaded_metrics
            print(f"Resuming training from epoch {start_epoch + 1}")
        try:
            for epoch in range(start_epoch, num_epochs):
                self.current_epoch = epoch
                try:  # 训练阶段
                    self.model.train()
                    total_loss = 0
                    loader_enumerator = enumerate(tqdm(self.train_loader,
                                                       desc=f'Training Epoch {epoch + 1}/{num_epochs}',
                                                       initial=start_batch if epoch == start_epoch else 0))
                    for batch_idx, batch in loader_enumerator:
                        self.current_batch = batch_idx
                        if epoch == start_epoch and batch_idx < start_batch:
                            continue
                        self.optimizer.zero_grad()
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        domains = batch['domain'].to(self.device)
                        with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=True):
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
                            features = outputs.hidden_states[-1][:, 0, :]
                            logits = outputs.logits
                            main_loss = outputs.loss if outputs.loss is not None else self.loss_fn(logits, labels)
                            domain_loss = self._compute_dann_loss(features, domains)
                            # 总损失
                            total_loss = main_loss + self.dann_weight * domain_loss

                        self.scaler.scale(total_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                        total_loss += total_loss.item()
                        if batch_idx % 1000 == 0:
                            self._save_checkpoint(
                                epoch=epoch,
                                batch_idx=self.current_batch,
                                best_precision=self.best_precision,
                                patience_counter=self.patience_counter
                            )
                    average_loss = total_loss / len(self.train_loader)
                    print(f'Epoch {epoch + 1}, Loss: {average_loss:.4f}')
                    start_batch = 0
                    # 更新学习率
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.scheduler.step()

                    # 验证阶段
                    for val_path in self.val_paths:
                        precision, accuracy, recall, f1, domain_acc, attack_acc, model_acc = self.evaluate(
                            val_path=val_path)
                        print(f'Validation on {val_path} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
                              f'Recall: {recall:.4f}, F1: {f1:.4f}')

                        # 将当前 epoch 的指标添加到历史记录中
                        self.metrics_history[val_path]['overall']['accuracy'].append(accuracy)
                        self.metrics_history[val_path]['overall']['precision'].append(precision)
                        self.metrics_history[val_path]['overall']['recall'].append(recall)
                        self.metrics_history[val_path]['overall']['f1'].append(f1)

                        for domain, acc in domain_acc.items():
                            self.metrics_history[val_path]['domains'][domain]['accuracy'].append(acc)
                        for attack_type, acc in attack_acc.items():
                            self.metrics_history[val_path]['attacks'][attack_type]['accuracy'].append(acc)
                        for model_name, acc in model_acc.items():
                            self.metrics_history[val_path]['models'][model_name]['accuracy'].append(acc)
                        # 早停机制
                        if precision > self.best_precision:
                            self.best_precision = precision
                            self.patience_counter = 0
                            # 保存模型和优化器状态
                            torch.save(self.model.state_dict(), self.model_path)
                            torch.save(self.optimizer.state_dict(), self.optimizer_path)
                            print(f'Model and optimizer saved with precision {self.best_precision:.4f}')
                        else:
                            self.patience_counter += 1
                            if self.patience_counter >= early_stopping_patience:
                                print(f'Early stopping triggered at epoch {epoch + 1}')
                                break
                        self._save_epoch_results(
                            val_path=val_path,
                            epoch=epoch,
                            overall_metrics={
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            },
                            domain_acc=domain_acc,
                            attack_acc=attack_acc,
                            model_acc=model_acc
                        )
                except Exception as e:
                    print(f"Error occurred at epoch {epoch}: {str(e)}")
                    self._save_checkpoint(
                        epoch=self.current_epoch,
                        batch_idx=self.current_batch,
                        best_precision=self.best_precision,  # 传递实例变量
                        patience_counter=self.patience_counter
                    )
                    raise  # 重新抛出异常以便上层处理
        finally:  # 训练结束后清理检查点
            for val_path in self.val_paths:
                self._save_final_results(val_path)
            # 在所有 epoch 训练完成后绘制指标变化图
            self.plot_metrics()

    def evaluate(self, val_path):
        val_loader = self._create_val_loader(val_path)
        self.model.eval()
        all_predictions, all_true_labels = [], []
        all_domains = []
        all_attacks = []
        all_models = []
        # domain_names = []
        # for d in all_domains:
        #     if d == -1:  # 未知领域
        #         domain_names.append("unknown")
        #     else:
        #         domain_names.append(self.train_dataset.domain_encoder.inverse_transform([d])[0])
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation on {val_path}', leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if isinstance(outputs, torch.Tensor) else outputs[0]
                all_predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                all_true_labels.extend(batch['labels'].cpu().tolist())
                # all_domains.extend(batch['domain'])
                all_attacks.extend(batch['attack'])
                all_models.extend(batch['model'])
                all_domains.extend(batch['raw_domain'])
        precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=1)
        accuracy = accuracy_score(all_true_labels, all_predictions)
        recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=1)
        f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=1)
        df = pd.DataFrame({
            'domain': all_domains,
            'attack': all_attacks,
            'model': all_models,
            'true': all_true_labels,
            'pred': all_predictions,
        })
        domain_acc = {}
        attack_acc = {}
        model_acc = {}
        for domain in df['domain'].unique():
            domain_df = df[df['domain'] == domain]
            domain_acc[domain] = accuracy_score(domain_df['true'], domain_df['pred'])
        for attack_type in df['attack'].unique():
            attack_df = df[df['attack'] == attack_type]
            attack_acc[attack_type] = accuracy_score(attack_df['true'], attack_df['pred'])
        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            model_acc[model_name] = accuracy_score(model_df['true'], model_df['pred'])
        return precision, accuracy, recall, f1, domain_acc, attack_acc, model_acc

    def plot_metrics(self):
        """绘制每个验证数据集的指标变化曲线"""
        for val_path, metrics in self.metrics_history.items():
            plt.figure(figsize=(12, 6))
            epochs = range(1, len(metrics['overall']['accuracy']) + 1)

            # 绘制四条曲线
            plt.plot(epochs, metrics['overall']['accuracy'], 'b-', label='Accuracy')
            plt.plot(epochs, metrics['overall']['precision'], 'r--', label='Precision')
            plt.plot(epochs, metrics['overall']['recall'], 'g-.', label='Recall')
            plt.plot(epochs, metrics['overall']['f1'], 'm:', label='F1-Score')

            # 添加图表元素
            plt.title(f'Model Performance on {val_path} Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # 保存并显示图表
            plt.savefig(f'training_metrics_{val_path.split("/")[-1].split(".")[0]}.png')
            plt.show()
            for domain, data in metrics['domains'].items():
                plt.plot(epochs, data['accuracy'], linestyle='--',
                         label=f'Domain {domain} Accuracy')
            plt.legend()
            safe_name = os.path.splitext(os.path.basename(val_path))[0]
            plt.savefig(f'overall_metrics_{safe_name}.png')
            for attack_type, data in metrics['attacks'].items():
                plt.plot(epochs, data['accuracy'], linestyle='--',
                         label=f'Attack {attack_type} Accuracy')
            plt.legend()
            safe_name = os.path.splitext(os.path.basename(val_path))[0]
            plt.savefig(f'attack_metrics_{safe_name}.png')
            for model_name, data in metrics['models'].items():
                plt.plot(epochs, data['accuracy'], linestyle='--',
                         label=f'Model {model_name} Accuracy')
            plt.legend()
            safe_name = os.path.splitext(os.path.basename(val_path))[0]
            plt.savefig(f'model_metrics_{safe_name}.png')
            # 新增domain准确率柱状图
            plt.figure(figsize=(10, 5))
            domains = list(metrics['domains'].keys())
            final_acc = [data['accuracy'][-1] for data in metrics['domains'].values()]
            plt.bar(domains, final_acc, color=['blue', 'green', 'red'])
            plt.ylim(0, 1)
            plt.title(f'Final Domain Accuracy - {safe_name}')
            plt.savefig(f'domain_acc_{safe_name}.png')

            plt.figure(figsize=(20, 6))
            attacks = list(metrics['attacks'].keys())
            final_acc = [data['accuracy'][-1] for data in metrics['attacks'].values()]
            plt.bar(attacks, final_acc, color=['blue', 'green', 'red'])
            plt.ylim(0, 1)
            plt.title(f'Final Attack Accuracy - {safe_name}')
            plt.savefig(f'attack_acc_{safe_name}.png')

            plt.figure(figsize=(20, 6))
            models = list(metrics['models'].keys())
            final_acc = [data['accuracy'][-1] for data in metrics['models'].values()]
            plt.bar(models, final_acc, color=['blue', 'green', 'red'])
            plt.ylim(0, 1)
            plt.title(f'Final Model Accuracy - {safe_name}')
            plt.savefig(f'model_acc_{safe_name}.png')


class LongformerTextTrainer:
    def __init__(self, train_path, val_paths,
                 model_path='./output/cl_model.pth',
                 optimizer_path='./output/cl_tokenizer.pth',
                 max_len=512,
                 batch_size=32,
                 output_dir='./results',
                 ):
        self.accuracy_history = []
        self.precision_history = []
        self.recall_history = []
        self.f1_history = []
        self.auroc_history = []
        self.train_path = train_path
        self.val_paths = val_paths
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # 加载数据
        self.train_data = pd.read_csv(self.train_path).dropna(subset=['answer'])
        self.train_data['answer'] = self.train_data['answer'].astype(str)
        self.metrics_history = self._init_metrics_history()
        train_text_list = self.train_data['answer'].tolist()
        train_label_list = self.train_data['label'].tolist()
        train_domain_list = self.train_data['domain'].tolist()
        train_attack_list = self.train_data['attack'].tolist()
        train_model_list = self.train_data['model'].tolist()
        # val_text_list = self.val_data['answer'].tolist()
        # val_label_list = self.val_data['label'].tolist()
        model_id = "./cl_roberta_large"
        # 初始化tokenizer和数据集
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.train_dataset = TextDataset(
            train_text_list,
            train_label_list,
            train_domain_list,
            train_attack_list,
            train_model_list,
            self.tokenizer,
            self.max_len
        )
        # self.val_dataset = TextDataset(val_text_list, val_label_list, self.tokenizer, self.max_len)

        # 数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
                                       pin_memory=True, persistent_workers=True)
        # self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # 初始化模型和优化器
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2
        )
        self.model.to(self.device)
        self.num_domains = len(self.train_data['domain'].unique())  # 自动获取领域数量
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_domains)
        ).to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)  # 支持多GPU
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化器
            torch.backends.cudnn.enabled = True  # 启用cudnn加速
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        self.amp_dtype = torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        # 获取优化器和学习率调度器
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler()

        # 类别不平衡：加权损失函数
        class_weights = self.get_class_weights()
        self.loss_fn = CrossEntropyLoss(weight=class_weights.to(self.device))
        self.checkpoint_path = os.path.join(self.output_dir, 'training_checkpoint.pth')
        self._register_signal_handlers()
        self.best_precision = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.dann_alpha = 0.5  # 梯度反转强度
        self.dann_weight = 0.3  # 对抗损失权重

        # 添加领域分类器

    def _register_signal_handlers(self):
        """注册Unix系统信号处理器"""
        if sys.platform != 'win32':  # Windows不支持SIGHUP
            signal.signal(signal.SIGHUP, self._graceful_exit)
        signal.signal(signal.SIGINT, self._graceful_exit)
        signal.signal(signal.SIGTERM, self._graceful_exit)

    def _graceful_exit(self, signum, frame):
        """优雅退出处理"""
        print(f"\n捕获到中断信号 {signum}, 正在保存检查点...")
        self._save_checkpoint(
            epoch=self.current_epoch,
            batch_idx=self.current_batch,
            best_precision=self.best_precision,
            patience_counter=self.patience_counter
        )
        sys.exit(1)

    def _init_metrics_history(self):
        """初始化指标历史数据结构"""
        return {
            val_path: {
                'overall': defaultdict(list),
                'domains': defaultdict(list_factory),
                'attacks': defaultdict(list_factory),
                'models': defaultdict(list_factory)
            } for val_path in self.val_paths
        }

    @staticmethod
    def _create_domain_dict():
        """可序列化的domain字典工厂函数"""
        return {'accuracy': []}

    @staticmethod
    def _create_attack_dict():
        """可序列化的attack字典工厂函数"""
        return {'accuracy': []}

    @staticmethod
    def _create_model_dict():
        """可序列化的attack字典工厂函数"""
        return {'accuracy': []}

    def _check_numpy_availability(self):
        """检查numpy可用性"""
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError("Numpy is required but not available. Please install numpy first.")

    def _save_checkpoint(self, epoch, batch_idx, best_precision, patience_counter):
        """改进的检查点保存方法"""

        # 转换defaultdict为普通dict以便序列化
        def convert_defaultdict(d):
            if isinstance(d, defaultdict):
                return {k: convert_defaultdict(v) for k, v in d.items()}
            return d

        temp_path = self.checkpoint_path + ".tmp"
        checkpoint_metrics = {}
        for val_path, metrics in self.metrics_history.items():
            checkpoint_metrics[val_path] = {
                'overall': convert_defaultdict(metrics['overall']),
                'domains': convert_defaultdict(metrics['domains']),
                'attacks': convert_defaultdict(metrics['attacks']),
                'models': convert_defaultdict(metrics['models'])
            }

        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_precision': self.best_precision,
            'patience_counter': self.patience_counter,
            'metrics_history': checkpoint_metrics
            # 使用转换后的可序列化数据
        }

        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, self.checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
        except Exception as e:
            print(f"Failed to save checkpoint: {str(e)}")
            raise

    def _load_checkpoint(self):
        """修正后的检查点加载方法"""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # 1. 模型参数加载与验证
            state_dict = checkpoint['model_state']
            # 处理多GPU参数名
            if not isinstance(self.model, torch.nn.DataParallel) and any(k.startswith('module.') for k in state_dict):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            elif isinstance(self.model, torch.nn.DataParallel) and not any(k.startswith('module.') for k in state_dict):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}

            # 严格加载并打印警告
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=True)
            if missing_keys:
                logger.warning(f"Missing keys when loading model: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")

            # 2. 优化器状态设备迁移
            optimizer_state = checkpoint['optimizer_state']
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.optimizer.load_state_dict(optimizer_state)

            # 3. 学习率调度器完整恢复
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.scheduler.optimizer = self.optimizer  # 必须重新绑定优化器

            # 4. 指标历史数据结构转换

            def recursive_convert(d):
                if isinstance(d, dict):
                    return defaultdict(list_factory, {k: recursive_convert(v) for k, v in d.items()})
                return d

            loaded_metrics = {}
            for val_path, metrics in checkpoint['metrics_history'].items():
                loaded_metrics[val_path] = {
                    'overall': defaultdict(list, metrics['overall']),
                    'domains': recursive_convert(metrics['domains']),
                    'attacks': recursive_convert(metrics['attacks']),
                    'models': recursive_convert(metrics['models'])
                }

            return (
                checkpoint.get('epoch', 0),
                checkpoint.get('batch_idx', 0),
                checkpoint.get('best_precision', 0.0),
                checkpoint.get('patience_counter', 0),
                loaded_metrics
            )

        return 0, 0, 0.0, 0, None

    def _create_val_loader(self, val_path):
        val_data = pd.read_csv(val_path).dropna(subset=['answer'])
        val_data['answer'] = val_data['answer'].astype(str)
        val_text_list = val_data['answer'].tolist()
        val_label_list = val_data['label'].tolist()
        val_domain_list = val_data['domain'].tolist()
        # val_domain_list = val_data['raw_domain'].apply(
        #     lambda x: x if x in self.train_dataset.domain_encoder.classes_ else "unknown"
        # )
        # 直接使用已扩展的encoder
        # val_domain_encoded = self.train_dataset.domain_encoder.transform(val_domain_list)
        val_attack_list = val_data['attack'].tolist()
        val_model_list = val_data['model'].tolist()
        val_dataset = TextDataset(
            val_text_list,
            val_label_list,
            val_domain_list,
            val_attack_list,
            val_model_list,
            self.tokenizer,
            self.max_len,
            domain_encoder=self.train_dataset.domain_encoder
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # 获取优化器和调度器
    def get_optimizer_and_scheduler(self, lr=2e-5, warmup_ratio=0.1):
        optimizer = AdamW([
            {'params': self.model.parameters()},
            {'params': self.domain_classifier.parameters(), 'lr': 1e-4}
        ], lr=lr, weight_decay=0.01)
        total_steps = len(self.train_loader) * 10
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.3
        )
        return optimizer, scheduler

    # 计算类别权重
    def get_class_weights(self):
        labels = self.train_data['label'].values
        labels = labels.astype(int)
        class_sample_count = np.bincount(labels)  # 获取每个类别的样本数量
        class_sample_count = np.where(class_sample_count == 0, 1e-6, class_sample_count)  # 防止除0错误
        weight = 1.0 / class_sample_count  # 样本少的类别权重大
        return torch.tensor(weight, dtype=torch.float)  # 直接返回类别权重

    def _save_epoch_results(self, val_path, epoch, overall_metrics, domain_acc, attack_acc, model_acc):
        """保存单个epoch的评估结果到txt文件"""
        safe_val_name = self._get_safe_filename(val_path)
        file_path = os.path.join(self.output_dir, f"{safe_val_name}_metrics.txt")

        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"=== Epoch {epoch + 1} ===\n")
                f.write(f"[Overall]\n")
                f.write(f"Accuracy: {overall_metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {overall_metrics['precision']:.4f}\n")
                f.write(f"Recall: {overall_metrics['recall']:.4f}\n")
                f.write(f"F1-Score: {overall_metrics['f1']:.4f}\n\n")
                f.write(f"Validation AUROC: {overall_metrics['auroc']:.4f}\n")

                f.write(f"[Domain Accuracies]\n")
                for domain, acc in sorted(domain_acc.items()):
                    acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
                    f.write(f"- {domain}: {acc_str}\n")
                f.write("\n" + "=" * 40 + "\n\n")
                f.write(f"[Attack Accuracies]\n")
                for attack_type, acc in sorted(attack_acc.items()):
                    acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
                    f.write(f"- {attack_type}: {acc_str}\n")
                f.write("\n" + "=" * 40 + "\n\n")
                f.write(f"[Model Accuracies]\n")
                for model_name, acc in sorted(model_acc.items()):
                    acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
                    f.write(f"- {model_name}: {acc_str}\n")
        except Exception as e:
            print(f"无法保存评估结果到 {file_path}: {str(e)}")

    def _save_final_results(self, val_path):
        """保存最终评估结果到独立文件"""
        safe_val_name = self._get_safe_filename(val_path)
        final_path = os.path.join(self.output_dir, f"{safe_val_name}_final.txt")
        metrics = self.metrics_history[val_path]

        try:
            with open(final_path, 'w', encoding='utf-8') as f:
                # 整体指标
                f.write("=== Final Evaluation Results ===\n")
                f.write(f"[Overall Metrics History]\n")
                for metric in ['accuracy', 'precision', 'recall', 'f1','auroc']:
                    values = metrics['overall'][metric]
                    f.write(f"{metric.capitalize()}: {[round(v, 4) for v in values]}\n")

                # 各domain最终准确率
                f.write("\n[Final Domain Accuracies]\n")
                for domain, data in metrics['domains'].items():
                    final_acc = data['accuracy'][-1] if len(data['accuracy']) > 0 else np.nan
                    acc_str = f"{final_acc:.4f}" if not np.isnan(final_acc) else "N/A"
                    f.write(f"- {domain}: {acc_str}\n")
                f.write("\n[Final Attack Accuracies]\n")
                for attack_type, data in metrics['attacks'].items():
                    final_acc = data['accuracy'][-1] if len(data['accuracy']) > 0 else np.nan
                    acc_str = f"{final_acc:.4f}" if not np.isnan(final_acc) else "N/A"
                    f.write(f"- {attack_type}: {acc_str}\n")
                f.write("\n[Final Model Accuracies]\n")
                for model_name, data in metrics['models'].items():
                    final_acc = data['accuracy'][-1] if len(data['accuracy']) > 0 else np.nan
                    acc_str = f"{final_acc:.4f}" if not np.isnan(final_acc) else "N/A"
                    f.write(f"- {model_name}: {acc_str}\n")
                # 添加统计信息
                f.write("\n[Statistics]\n")
                f.write(f"Total Epochs: {len(metrics['overall']['accuracy'])}\n")
                f.write(f"Domains Count: {len(metrics['domains'])}\n")
                f.write(f"Attacks Count: {len(metrics['attacks'])}\n")
                f.write(f"Models Count: {len(metrics['models'])}\n")
        except Exception as e:
            print(f"无法保存最终结果到 {final_path}: {str(e)}")

    # 训练函数
    def _get_safe_filename(self, path):
        """生成安全的文件名"""
        base_name = os.path.basename(path)
        return re.sub(r'[\\/*?:"<>|]', "_", base_name.split('.')[0])

    def _compute_dann_loss(self, features, domains):
        """计算领域对抗损失"""
        # 梯度反转
        reversed_features = GradientReversalFunction.apply(features, self.dann_alpha)
        domain_logits = self.domain_classifier(reversed_features)
        return F.cross_entropy(domain_logits, domains)

    def train(self, num_epochs=10):
        start_epoch, start_batch, self.best_precision, self.patience_counter, loaded_metrics = self._load_checkpoint()
        if loaded_metrics:
            self.metrics_history = loaded_metrics
            print(f"Resuming training from epoch {start_epoch + 1}")
        try:
            for epoch in range(start_epoch, num_epochs):
                self.current_epoch = epoch
                try:  # 训练阶段
                    self.model.train()
                    total_loss = 0
                    loader_enumerator = enumerate(tqdm(self.train_loader,
                                                       desc=f'Training Epoch {epoch + 1}/{num_epochs}',
                                                       initial=start_batch if epoch == start_epoch else 0))
                    for batch_idx, batch in loader_enumerator:
                        self.current_batch = batch_idx
                        if epoch == start_epoch and batch_idx < start_batch:
                            continue
                        self.optimizer.zero_grad()
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        domains = batch['domain'].to(self.device)
                        with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=True):
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
                            features = outputs.hidden_states[-1][:, 0, :]
                            logits = outputs.logits
                            main_loss = outputs.loss if outputs.loss is not None else self.loss_fn(logits, labels)
                            domain_loss = self._compute_dann_loss(features, domains)
                            # 总损失
                            total_loss = main_loss

                        self.scaler.scale(total_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                        total_loss += total_loss.item()
                        if batch_idx % 1000 == 0:
                            self._save_checkpoint(
                                epoch=epoch,
                                batch_idx=self.current_batch,
                                best_precision=self.best_precision,
                                patience_counter=self.patience_counter
                            )
                    average_loss = total_loss / len(self.train_loader)
                    print(f'Epoch {epoch + 1}, Loss: {average_loss:.4f}')
                    start_batch = 0
                    # 更新学习率
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.scheduler.step()

                    # 验证阶段
                    for val_path in self.val_paths:
                        precision, accuracy, recall, f1, auroc, domain_acc, attack_acc, model_acc = self.evaluate(
                            val_path=val_path)
                        print(f'Validation on {val_path} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
                              f'Recall: {recall:.4f}, F1: {f1:.4f},AUROC: {auroc:.4f}')

                        # 将当前 epoch 的指标添加到历史记录中
                        self.metrics_history[val_path]['overall']['accuracy'].append(accuracy)
                        self.metrics_history[val_path]['overall']['precision'].append(precision)
                        self.metrics_history[val_path]['overall']['recall'].append(recall)
                        self.metrics_history[val_path]['overall']['f1'].append(f1)
                        self.metrics_history[val_path]['overall']['auroc'].append(auroc)

                        for domain, acc in domain_acc.items():
                            self.metrics_history[val_path]['domains'][domain]['accuracy'].append(acc)
                        for attack_type, acc in attack_acc.items():
                            self.metrics_history[val_path]['attacks'][attack_type]['accuracy'].append(acc)
                        for model_name, acc in model_acc.items():
                            self.metrics_history[val_path]['models'][model_name]['accuracy'].append(acc)
                        # 早停机制
                        if precision > self.best_precision:
                            self.best_precision = precision
                            self.patience_counter = 0
                            # 保存模型和优化器状态
                            torch.save(self.model.state_dict(), self.model_path)
                            torch.save(self.optimizer.state_dict(), self.optimizer_path)
                            print(f'Model and optimizer saved with precision {self.best_precision:.4f}')
                        self._save_epoch_results(
                            val_path=val_path,
                            epoch=epoch,
                            overall_metrics={
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'auroc': auroc
                            },
                            domain_acc=domain_acc,
                            attack_acc=attack_acc,
                            model_acc=model_acc
                        )
                except Exception as e:
                    print(f"Error occurred at epoch {epoch}: {str(e)}")
                    self._save_checkpoint(
                        epoch=self.current_epoch,
                        batch_idx=self.current_batch,
                        best_precision=self.best_precision,  # 传递实例变量
                        patience_counter=self.patience_counter
                    )
                    raise  # 重新抛出异常以便上层处理
        finally:  # 训练结束后清理检查点
            for val_path in self.val_paths:
                self._save_final_results(val_path)
            # 在所有 epoch 训练完成后绘制指标变化图
            torch.save(self.model.state_dict(), self.model_path.replace(".pth", "_final.pth"))
            torch.save(self.optimizer.state_dict(), self.optimizer_path.replace(".pth", "_final.pth"))
            print("Final model and optimizer saved after all epochs.")
            self.plot_metrics()

    def evaluate(self, val_path):
        val_loader = self._create_val_loader(val_path)
        self.model.eval()
        all_predictions, all_true_labels, all_probabilities = [], [], []
        all_domains = []
        all_attacks = []
        all_models = []
        # domain_names = []
        # for d in all_domains:
        #     if d == -1:  # 未知领域
        #         domain_names.append("unknown")
        #     else:
        #         domain_names.append(self.train_dataset.domain_encoder.inverse_transform([d])[0])
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation on {val_path}', leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if isinstance(outputs, torch.Tensor) else outputs[0]
                probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # 保存正类概率
                all_probabilities.extend(probabilities)
                all_predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                all_true_labels.extend(batch['labels'].cpu().tolist())
                # all_domains.extend(batch['domain'])
                all_attacks.extend(batch['attack'])
                all_models.extend(batch['model'])
                all_domains.extend(batch['raw_domain'])
        precision = precision_score(all_true_labels, all_predictions, average='binary', zero_division=1)
        accuracy = accuracy_score(all_true_labels, all_predictions)
        recall = recall_score(all_true_labels, all_predictions, average='binary', zero_division=1)
        f1 = f1_score(all_true_labels, all_predictions, average='binary', zero_division=1)
        # 新增AUROC计算 - 关键修改点
        try:
            auroc = roc_auc_score(all_true_labels, all_probabilities)  # 使用概率计算AUROC
        except ValueError as e:
            print(f"AUROC计算错误: {e}")
            auroc = np.nan  # 异常情况填充NaN
        df = pd.DataFrame({
            'domain': all_domains,
            'attack': all_attacks,
            'model': all_models,
            'true': all_true_labels,
            'pred': all_predictions,
        })
        domain_acc = {}
        attack_acc = {}
        model_acc = {}
        for domain in df['domain'].unique():
            domain_df = df[df['domain'] == domain]
            domain_acc[domain] = accuracy_score(domain_df['true'], domain_df['pred'])
        for attack_type in df['attack'].unique():
            attack_df = df[df['attack'] == attack_type]
            attack_acc[attack_type] = accuracy_score(attack_df['true'], attack_df['pred'])
        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            model_acc[model_name] = accuracy_score(model_df['true'], model_df['pred'])
        return precision, accuracy, recall, f1, auroc, domain_acc, attack_acc, model_acc

    def plot_metrics(self):
        """绘制每个验证数据集的指标变化曲线"""
        for val_path, metrics in self.metrics_history.items():
            plt.figure(figsize=(12, 6))
            epochs = range(1, len(metrics['overall']['accuracy']) + 1)

            # 绘制四条曲线
            plt.plot(epochs, metrics['overall']['accuracy'], 'b-', label='Accuracy')
            plt.plot(epochs, metrics['overall']['precision'], 'r--', label='Precision')
            plt.plot(epochs, metrics['overall']['recall'], 'g-.', label='Recall')
            plt.plot(epochs, metrics['overall']['f1'], 'm:', label='F1-Score')
            plt.plot(epochs, metrics['overall']['auroc'], 'k--', label='AUROC')
            # 添加图表元素
            plt.title(f'Model Performance on {val_path} Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # 保存并显示图表
            plt.savefig(f'training_metrics_{val_path.split("/")[-1].split(".")[0]}.png')
            plt.show()
            for domain, data in metrics['domains'].items():
                plt.plot(epochs, data['accuracy'], linestyle='--',
                         label=f'Domain {domain} Accuracy')
            plt.legend()
            safe_name = os.path.splitext(os.path.basename(val_path))[0]
            plt.savefig(f'overall_metrics_{safe_name}.png')
            for attack_type, data in metrics['attacks'].items():
                plt.plot(epochs, data['accuracy'], linestyle='--',
                         label=f'Attack {attack_type} Accuracy')
            plt.legend()
            safe_name = os.path.splitext(os.path.basename(val_path))[0]
            plt.savefig(f'attack_metrics_{safe_name}.png')
            for model_name, data in metrics['models'].items():
                plt.plot(epochs, data['accuracy'], linestyle='--',
                         label=f'Model {model_name} Accuracy')
            plt.legend()
            safe_name = os.path.splitext(os.path.basename(val_path))[0]
            plt.savefig(f'model_metrics_{safe_name}.png')
            # 新增domain准确率柱状图
            plt.figure(figsize=(10, 5))
            domains = list(metrics['domains'].keys())
            final_acc = [data['accuracy'][-1] for data in metrics['domains'].values()]
            plt.bar(domains, final_acc, color=['blue', 'green', 'red'])
            plt.ylim(0, 1)
            plt.title(f'Final Domain Accuracy - {safe_name}')
            plt.savefig(f'domain_acc_{safe_name}.png')

            plt.figure(figsize=(20, 6))
            attacks = list(metrics['attacks'].keys())
            final_acc = [data['accuracy'][-1] for data in metrics['attacks'].values()]
            plt.bar(attacks, final_acc, color=['blue', 'green', 'red'])
            plt.ylim(0, 1)
            plt.title(f'Final Attack Accuracy - {safe_name}')
            plt.savefig(f'attack_acc_{safe_name}.png')

            plt.figure(figsize=(20, 6))
            models = list(metrics['models'].keys())
            final_acc = [data['accuracy'][-1] for data in metrics['models'].values()]
            plt.bar(models, final_acc, color=['blue', 'green', 'red'])
            plt.ylim(0, 1)
            plt.title(f'Final Model Accuracy - {safe_name}')
            plt.savefig(f'model_acc_{safe_name}.png')

            plt.figure(figsize=(12, 6))
            plt.plot(epochs, metrics['overall']['auroc'], 'k--', label='AUROC')
            plt.title(f'AUROC Curve on {val_path}')
            plt.xlabel('Epochs')
            plt.ylabel('AUROC')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'auroc_curve_{safe_name}.png')
