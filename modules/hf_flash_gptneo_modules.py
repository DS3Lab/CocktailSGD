import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention as _GPTNeoSelfAttention
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoMLP
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig as GPTConfig

try:
    from flash_attn.flash_attention import FlashAttention
    flash_attn_installed = True
    print('>>>>> flash attention')
except ImportError:
    flash_attn_installed = False
    
    
# @torch.jit.script
def gpt_loss_func(input, target):
    lm_logits, labels = input, target
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embed_dropout)
        
    def forward(self, input_ids, *args, **kargs):
        
        device = input_ids.device
        
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        
        # position ids
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)
        
        return hidden_states
    
    
class GPTNeoSelfAttention(_GPTNeoSelfAttention):
    
    def __init__(self, config, attention_type):
        super().__init__(config, attention_type)
        self.attention_type = attention_type
        
        self.scaling = 1 #self.head_dim**-0.5 # looks like neo does not scale
        
        if flash_attn_installed:
            self.flash_attn = FlashAttention(softmax_scale=self.scaling, attention_dropout=config.attention_dropout)
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None, prefix_masks=None):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        
        # do not change for local attention
        if prefix_masks is not None and self.attention_type != 'local':
            for _prefix_masks in prefix_masks.bool():
                causal_mask[:, :, :, _prefix_masks] = 1
        
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        prefix_masks=None,
    ):

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        bsz, tgt_len, _ = hidden_states.size()
        
        assert flash_attn_installed
        
        qkv = torch.stack(
            [
                query_states.view((bsz, tgt_len, self.num_heads, self.head_dim)),
                key_states.view((bsz, tgt_len, self.num_heads, self.head_dim)),
                value_states.view((bsz, tgt_len, self.num_heads, self.head_dim)),
            ],
            dim=2
        )

        attn_output, _ = self.flash_attn(qkv, causal=True) # assuming that these are autoregressive!!
        attn_output = attn_output.reshape((bsz, tgt_len, self.embed_dim))

        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        present = None
        attn_weights = None

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
    
    
class GPTNeoAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]

        if self.attention_type in ["global", "local"]:
            self.attention = GPTNeoSelfAttention(config, self.attention_type)
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )
            
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        prefix_masks=None,
    ):
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            prefix_masks=prefix_masks
        )
    

class GPTBlock(nn.Module):
    
    def __init__(self, config, layer_id, *args, use_checkpoint=True, **kargs):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTNeoAttention(config, layer_id)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPTNeoMLP(inner_dim, config)

        self.config = config
        self.use_checkpoint = use_checkpoint
        
        def attn_res(x: torch.Tensor, prefix_masks=None, attention_mask=None) -> torch.Tensor:
            res = x
            x = self.ln_1(x)
            x = self.attn(x, prefix_masks=prefix_masks, attention_mask=attention_mask)[0]
            return x + res
        self.attn_res = attn_res
        
        def mlp_res(x: torch.Tensor) -> torch.Tensor:
            res = x
            x = self.ln_2(x)
            x = self.mlp(x)
            return x + res
        self.mlp_res = mlp_res

    def forward(self, x: torch.Tensor, prefix_masks=None, mask=None, *args, **kargs) -> torch.Tensor:
        
        if mask is not None:
            # bool -> float
            attention_mask = (1e4)*(mask[:, None, None, :]-1.0)
        else:
            attention_mask = None
        
        if not self.training:
            x = self.attn_res(x, prefix_masks=prefix_masks, attention_mask=attention_mask)
            x = self.mlp_res(x)
            return x
        
        if self.use_checkpoint:
            x.requires_grad_(True)
            x = checkpoint(self.attn_res, x, prefix_masks, attention_mask)
        else:
            x = self.attn_res(x, prefix_masks, attention_mask)
            
        if self.use_checkpoint:
            x.requires_grad_(True)
            x = checkpoint(self.mlp_res, x)
        else:
            x = self.mlp_res(x)
        return x
    
    
class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, x, input_ids=None, *args, **kargs):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
    
        
class GPTClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        
    def forward(self, hidden_states, input_ids=None, *args, **kargs):
        
        batch_size, sequence_length = hidden_states.shape[:2]
        if input_ids is not None:
            sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        else:
            sequence_lengths = -1
        
        pooled_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        logits = self.score(self.ln_f(pooled_hidden_states))
        
        return 