import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from .utils import TransformerLogger
from transformers.modeling_utils import SequenceSummary
from transformers import (BertForSequenceClassification, BertModel,
                          XLNetForSequenceClassification, XLNetModel,
                          RobertaForSequenceClassification, RobertaModel,
                          AlbertForSequenceClassification, AlbertModel,
                          LongformerForSequenceClassification, LongformerModel,
                          DebertaForSequenceClassification, DebertaModel,
                          MegatronBertForSequenceClassification, MegatronBertModel,
                          PreTrainedModel)
from .model_utils import StableDropout, FocalLoss, BCEFocalLoss


logger = TransformerLogger(logger_level='i').get_logger()


class BaseModel(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.spec_tag1, self.spec_tag2, self.spec_tag3, self.spec_tag4 = config.tags
        self.scheme = config.scheme
        self.num_labels = config.num_labels

        if hasattr(config, 'balance_sample_weights'):
            sample_weights = config.sample_weights if config.balance_sample_weights else None
            sample_weights = torch.tensor(sample_weights, dtype=torch.float32) if sample_weights else None
        else:
            sample_weights = None

        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            # TODO: the sample weights need to be tuned for focal loss functions; we do not support currently
            if config.binary_mode:
                self.loss_fct = BCEFocalLoss(gamma=config.focal_loss_gamma)
            else:
                self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            if hasattr(config, 'binary_mode') and config.binary_mode:
                self.loss_fct = BCEWithLogitsLoss(weight=sample_weights)
            else:
                self.loss_fct = CrossEntropyLoss(weight=sample_weights)

        self.drop_out = StableDropout(config.hidden_dropout_prob)

        if self.scheme == 1:
            self.classifier_dim = config.hidden_size * 3
        elif self.scheme == 2:
            self.classifier_dim = config.hidden_size * 5
        elif self.scheme == 3:
            self.classifier_dim = config.hidden_size * 2
        else:
            self.classifier_dim = config.hidden_size

        self.base_classifier = nn.Linear(self.classifier_dim, self.num_labels)

    @staticmethod
    def special_tag_representation(seq_output, input_ids, special_tag):
        spec_idx = (input_ids == special_tag).nonzero(as_tuple=False)

        temp = []
        for idx in spec_idx:
            temp.append(seq_output[idx[0], idx[1], :])
        tags_rep = torch.stack(temp, dim=0)

        return tags_rep

    def output2logits(self, pooled_output, seq_output, input_ids):
        if self.scheme == 1:
            seq_tags = []
            for each_tag in [self.spec_tag1, self.spec_tag3]:
                seq_tags.append(self.special_tag_representation(seq_output, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
        elif self.scheme == 2:
            seq_tags = []
            for each_tag in [self.spec_tag1, self.spec_tag2, self.spec_tag3, self.spec_tag4]:
                seq_tags.append(self.special_tag_representation(seq_output, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
        elif self.scheme == 3:
            seq_tags = []
            for each_tag in [self.spec_tag1, self.spec_tag3]:
                seq_tags.append(self.special_tag_representation(seq_output, input_ids, each_tag))
            new_pooled_output = torch.cat(seq_tags, dim=1)
        else:
            new_pooled_output = pooled_output

        logits = self.base_classifier(self.drop_out(new_pooled_output))

        return logits

    def calc_loss(self, logits, outputs, labels):
        new_outputs = (logits,) + outputs[2:]
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        new_outputs = (loss,) + new_outputs

        return new_outputs


class BertForRelationIdentification(BertForSequenceClassification, BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
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
        seq_output = outputs[0]
        logits = self.output2logits(pooled_output, seq_output, input_ids)

        return self.calc_loss(logits, outputs, labels)


class RoBERTaForRelationIdentification(RobertaForSequenceClassification, BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        pooled_output = outputs[1]
        seq_output = outputs[0]
        logits = self.output2logits(pooled_output, seq_output, input_ids)

        return self.calc_loss(logits, outputs, labels)


class AlbertForRelationIdentification(AlbertForSequenceClassification, BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
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
        seq_output = outputs[0]
        logits = self.output2logits(pooled_output, seq_output, input_ids)

        return self.calc_loss(logits, outputs, labels)


class XLNetForRelationIdentification(XLNetForSequenceClassification, BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
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
                use_cache=True,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                **kwargs):

        outputs = self.transformer(
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
                **kwargs)

        seq_output = outputs[0]
        pooled_output = self.sequence_summary(seq_output)
        logits = self.output2logits(pooled_output, seq_output, input_ids)

        return self.calc_loss(logits, outputs, labels)


class LongFormerForRelationIdentification(LongformerForSequenceClassification, BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                global_attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                **kwargs):

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        pooled_output = outputs[1]
        seq_output = outputs[0]
        logits = self.output2logits(pooled_output, seq_output, input_ids)

        return self.calc_loss(logits, outputs, labels)


class DebertaForRelationIdentification(DebertaForSequenceClassification, BaseModel):
    def __init__(self, config):
        from model_utils import ContextPooler
        super().__init__(config)
        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        seq_output = outputs[0]
        pooled_output = self.pooler(seq_output)
        logits = self.output2logits(pooled_output, seq_output, input_ids)

        return self.calc_loss(logits, outputs, labels)


class MegatronForRelationIdentification(MegatronBertForSequenceClassification, BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = MegatronBertModel(config)
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
        seq_output = outputs[0]
        logits = self.output2logits(pooled_output, seq_output, input_ids)

        return self.calc_loss(logits, outputs, labels)
