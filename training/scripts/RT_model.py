import transformers
from transformers.models.llama.modeling_llama import *
from dataclasses import dataclass

class TTLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        input_size,
        output_size,
    ):
        super().__init__()
        self.tt_layernorm = LlamaRMSNorm(input_size, eps=config.rms_norm_eps)
        self.tt_linear = nn.Linear(input_size, output_size, bias=False)
        self.tt_act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        x: torch.Tensor,
        residual = False,
    ):
        y = self.tt_layernorm(x)
        y = self.tt_linear(y)
        y = self.tt_act_fn(y)
        if residual:
            y = y + x[..., -y.shape[-1]:]
        return y


@dataclass
class CausalLMRewardTransformerOutputWithPast(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sr_logits: torch.FloatTensor = None

class LlamaForCausalLMRewardTransformer(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        tt_hidden_size = getattr(config, 'tt_hidden_size', 512)
        tt_layers = []
        for i in range(config.num_hidden_layers):
            if i == 0:
                tt_layers.append(TTLayer(config, input_size=config.hidden_size, output_size=tt_hidden_size))
            else:
                tt_layers.append(TTLayer(config, input_size=config.hidden_size+tt_hidden_size, output_size=tt_hidden_size))
        self.tt_layers = nn.ModuleList(tt_layers)

        ### if we set sr_out_dim=1, the `HF Trainer` will silently fail.
        ### so we set sr_out_dim>1, but for now we only use the first dim
        # self.sr_head = nn.Linear(config.hidden_size, 1, bias=False) 

        sr_out_dim = getattr(config, 'sr_out_dim', 2)
        sr_head_layer = getattr(config, 'sr_head_layer', 1)
        sr_hidden_size_rate = getattr(config, 'sr_hidden_size_rate', 1)

        if sr_head_layer == 1:
            self.sr_head = nn.Linear(config.hidden_size+tt_hidden_size, sr_out_dim, bias=False)
        else:
            sr_hidden_size = sr_hidden_size_rate * config.hidden_size
            layers = []
            layers.append(nn.Linear(config.hidden_size+tt_hidden_size, sr_hidden_size, bias=False)) 
            layers.append(ACT2FN[config.hidden_act]) 
            for _ in range(sr_head_layer-2):
                layers.append(nn.Linear(sr_hidden_size, sr_hidden_size, bias=False)) 
                layers.append(ACT2FN[config.hidden_act])
            layers.append(nn.Linear(sr_hidden_size, sr_out_dim, bias=False))
            self.sr_head = nn.Sequential(*layers)

        # Initialize weights and apply final processing
        self.post_init()

    def reset_tt_sr(self, tt_hidden_size=512, sr_head_layer=1, sr_out_dim=2, sr_hidden_size_rate=1):
        self.config.tt_hidden_size = tt_hidden_size
        self.config.sr_head_layer = sr_head_layer
        self.config.sr_out_dim = sr_out_dim
        self.config.sr_hidden_size_rate = sr_hidden_size_rate
        config = self.config

        tt_layers = []
        for i in range(config.num_hidden_layers):
            if i == 0:
                tt_layers.append(TTLayer(config, input_size=config.hidden_size, output_size=tt_hidden_size))
            else:
                tt_layers.append(TTLayer(config, input_size=config.hidden_size+tt_hidden_size, output_size=tt_hidden_size))
        self.tt_layers = nn.ModuleList(tt_layers)

        if sr_head_layer == 1:
            self.sr_head = nn.Linear(config.hidden_size+tt_hidden_size, sr_out_dim, bias=False)
        else:
            sr_hidden_size = sr_hidden_size_rate * config.hidden_size
            layers = []
            layers.append(nn.Linear(config.hidden_size+tt_hidden_size, sr_hidden_size, bias=False)) 
            layers.append(ACT2FN[config.hidden_act]) 
            for _ in range(sr_head_layer-2):
                layers.append(nn.Linear(sr_hidden_size, sr_hidden_size, bias=False)) 
                layers.append(ACT2FN[config.hidden_act])
            layers.append(nn.Linear(sr_hidden_size, sr_out_dim, bias=False))
            self.sr_head = nn.Sequential(*layers)

        # Initialize weights
        for layer in self.tt_layers:
            self._init_weights(layer)
        self._init_weights(self.sr_head)


    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        # for `generat()` function
        return {"input_ids": input_ids}
        
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="LlamaConfig")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True
        return_dict = True

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            cache_position=cache_position,
        )

        last_hidden_state = outputs[0] # outputs[0]==outputs.hidden_states[-1]==outputs.last_hidden_state True [batch_size, seq_len, hidden_size]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(last_hidden_state, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(last_hidden_state[:, -num_logits_to_keep:, :])

        # Calculate the intermediate reward channel feature
        hidden_states=outputs.hidden_states
        assert len(self.tt_layers) == len(hidden_states)-1
        for i in range(len(self.tt_layers)):
            if i == 0:
                tt_hidden_state = self.tt_layers[0](hidden_states[0][:, -num_logits_to_keep:, :], residual=False)
            else:
                input_ = torch.cat([hidden_states[i][:, -num_logits_to_keep:, :], tt_hidden_state],dim=-1)
                tt_hidden_state = self.tt_layers[i](input_, residual=True)

        # The final reward is calculated by the last layer hidden_state and the last layer reward channel feature
        input_ = torch.cat([last_hidden_state[:, -num_logits_to_keep:, :], tt_hidden_state],dim=-1)
        sr_logits = self.sr_head(input_)
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMRewardTransformerOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sr_logits=sr_logits,
        )
