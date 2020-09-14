import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import SequenceSummary
from transformers import (BertForSequenceClassification, BertModel,
                          XLNetForSequenceClassification, XLNetModel,
                          RobertaForSequenceClassification, RobertaModel,
                          AlbertForSequenceClassification, AlbertModel,
                          PreTrainedModel)


class BaseModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if "tags" in config.__dict__:
            self.spec_tag1, self.spec_tag2, self.spec_tag3, self.spec_tag4 = config.tags
        else:
            self.spec_tag1, self.spec_tag2, self.spec_tag3, self.spec_tag4 = None, None, None, None
        if "scheme" in config.__dict__:
            self.scheme = config.scheme
        else:
            self.scheme = 0
        if "num_labels" in config.__dict__:
            self.num_labels = config.num_labels
        else:
            self.num_labels = 2

        self.loss_fct = CrossEntropyLoss()
        self.classifier1 = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier2 = nn.Linear(config.hidden_size * 3, config.num_labels)
        self.classifier3 = nn.Linear(config.hidden_size * 5, config.num_labels)

    @staticmethod
    def special_tag_representation(seq_output, input_ids, special_tag):
        spec_idx = (input_ids == special_tag).nonzero(as_tuple=False)
        temp = []
        for idx in spec_idx:
            temp.append(seq_output[idx[0], idx[1], :])
        return torch.stack(temp, dim=0)

    def output2logits(self, pooled_output, seq_output, input_ids):
        if self.scheme == 1:
            seq_tags = []
            for each_tag in [self.spec_tag1, self.spec_tag2]:
                seq_tags.append(self.special_tag_representation(seq_output, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
            logits = self.classifier2(new_pooled_output)
        elif self.scheme == 2:
            seq_tags = []
            for each_tag in [self.spec_tag1, self.spec_tag2, self.spec_tag3, self.spec_tag4]:
                seq_tags.append(self.special_tag_representation(seq_output, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
            logits = self.classifier3(new_pooled_output)
        else:
            logits = self.classifier1(pooled_output)

        return logits


class BertForRelationIdentification(BertForSequenceClassification, BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                **kwargs):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        seq_output = outputs[0]
        seq_output = self.dropout(seq_output)

        logits = self.output2logits(pooled_output, seq_output, input_ids)

        outputs = (logits,) + outputs[2:]

        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (loss,) + outputs

        return outputs


class RoBERTaForRelationIdentification(RobertaForSequenceClassification, BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

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
                **kwargs):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            utput_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        seq_output = outputs[0]
        seq_output = self.dropout(seq_output)

        logits = self.output2logits(pooled_output, seq_output, input_ids)

        outputs = (logits,) + outputs[2:]

        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (loss,) + outputs

        return outputs


class AlbertForRelationIdentification(AlbertForSequenceClassification, BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

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
                **kwargs):

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        seq_output = outputs[0]
        seq_output = self.dropout(seq_output)

        logits = self.output2logits(pooled_output, seq_output, input_ids)

        outputs = (logits,) + outputs[2:]

        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (loss,) + outputs

        return outputs


class XLNetForRelationIdentification(XLNetForSequenceClassification, BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.xlnet = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier1 = nn.Linear(config.d_model, config.num_labels)
        self.classifier2 = nn.Linear(config.d_model * 3, config.num_labels)
        self.classifier3 = nn.Linear(config.d_model * 5, config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                mems=None,
                perm_mask=None,
                target_mapping=None,
                token_type_ids=None,
                input_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        outputs = self.xlnet(
                input_ids,
                attention_mask=attention_mask,
                mems=mems,
                perm_mask=perm_mask,
                target_mapping=target_mapping,
                token_type_ids=token_type_ids,
                input_mask=input_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)

        seq_output = outputs[0]
        seq_output = self.dropout(seq_output)
        pooled_output = self.sequence_summary(seq_output)

        logits = self.output2logits(pooled_output, seq_output, input_ids)

        outputs = (logits,) + outputs[1:]

        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (loss,) + outputs

        return outputs
