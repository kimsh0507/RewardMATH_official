'''
https://github.com/GAIR-NLP/ReasonEval
'''
import torch
import torch.nn as nn
from transformers import MistralModel, MistralPreTrainedModel, LlamaModel, LlamaPreTrainedModel, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

class ReasonEval_7B(MistralPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = MistralModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dimension, bias=config.use_bias)
        self.post_init()  # Initialize weights and apply final processing

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (batch_size, sequence_length, dim)
        scores = self.score_head(hidden_states)  # (batch_size, sequence_length, class)
        return scores
    
class ReasonEval_34B(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dim, bias=config.bias)
        self.post_init() # Initialize weights and apply final processing

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
      
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (batch_size, sequence_length, dim)
        scores = self.score_head(hidden_states)  # (batch_size, sequence_length, class)
        return scores