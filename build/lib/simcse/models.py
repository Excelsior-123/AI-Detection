import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
import transformers
from transformers import RobertaTokenizer
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
logger = logging.getLogger(__name__)


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
        print("cls_logits:",cls_logits)
        print("labels:",labels)
        loss_fct = nn.CrossEntropyLoss()
        if labels is not None:
            labels = labels.view(-1)
            loss = loss_fct(cls_logits, labels)
            # logger.info("Classification loss: {:.4f}".format(loss.item()))
        else:
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
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1)
                 for i in range(z1_z3_cos.size(-1))]
            ).to(self.classifier.weight.device)
            cos_sim = cos_sim + weights

        loss_fct = nn.CrossEntropyLoss()
        sim_labels = torch.arange(cos_sim.size(0)).long().to(self.classifier.weight.device)
        contrast_loss = loss_fct(cos_sim, sim_labels)
        pooled = pooler_output[:, 0]
        cls_logits = self.classifier(pooled)
        # print("contrasive_cls_logits:",cls_logits)
        # print("contrasive_labels:",sim_labels)
        classification_loss = loss_fct(cls_logits, labels.view(-1)) if labels is not None else 0.0
        total_loss = contrast_loss + self.model_args.cls_weight * classification_loss

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
        self.bert = BertModel(config, add_pooling_layer=False)

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
        if sent_emb:
            return sentemb_forward(self, self.bert,
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
                                   )
        else:
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
        self.classifier = nn.Linear(self.config.hidden_size, 4)
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
