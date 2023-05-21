import os
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
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention as _GPTNeoXAttention
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer as _GPTNeoXBlock
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel as _GPTNeoXModel
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig as GPTConfig
# from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding

from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.flash_attention import FlashAttention
flash_attn_installed = True
print('>>>>> flash attention')

try:
    import apex.contrib.layer_norm
    # LayerNorm = apex.normalization.FusedLayerNorm
    LayerNorm = apex.contrib.layer_norm.FastLayerNorm
    print('>>>>> Apex FastLayerNorm')
except:
    LayerNorm = nn.LayerNorm

from einops import rearrange

class GPTNeoXAttention(_GPTNeoXAttention):
    
    def __init__(self, config):
        super(_GPTNeoXAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims, base=config.rotary_emb_base, interleaved=False)
        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype()),
            persistent=False,
        )
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        assert flash_attn_installed
        self.flash_attn = FlashAttention(softmax_scale=1.0/self.norm_factor, attention_dropout = 0)
    
    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        offset=None,
        output_attentions=False,
    ):
        has_layer_past = layer_past is not None
        
        bsz, tgt_len, _ = hidden_states.shape

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)
        
        qkv = rearrange(qkv, '... (h three d) -> ... h three d', three=3, d=self.head_size)
        qkv = qkv.permute(0, 1, 3, 2, 4)
        qkv = self.rotary_emb(qkv)

        # Compute attention
        attn_weights = None
        present = None
        
        attn_output, _ = self.flash_attn(qkv, causal=True)
        attn_output = attn_output.view(bsz, tgt_len, self.num_attention_heads * self.head_size)

        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.embed_in = nn.Embedding(config.vocab_size, self.embed_dim)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_embs.pt',
            )))
        except:
            print(f'Cannot load from <model_path>. The model is randomly initialized.')
        return module
        
    @torch.compile
    def forward(self, input_ids, *args, **kargs):
        
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embed_in(input_ids)
        return hidden_states
    

class GPTBlock(_GPTNeoXBlock):
    def __init__(self, config, *args, use_checkpoint=True, **kargs):
        super(_GPTNeoXBlock, self).__init__()
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = GPTNeoXAttention(config)
        self.mlp = GPTNeoXMLP(config)
        self.config = config
        self.use_checkpoint = use_checkpoint
        
        def mha_fw(x: torch.Tensor, res: torch.Tensor, attention_mask: torch.Tensor):
            attention_layer_output = self.attention(self.input_layernorm(x), attention_mask=attention_mask)
            attn_output = attention_layer_output[0]
            return attn_output + res
        
        @torch.compile()
        def mlp_fw(x: torch.Tensor, res: torch.Tensor):
            mlp_out = self.mlp(self.post_attention_layernorm(x))
            return mlp_out + res
        
        """
        To be compatible with https://github.com/huggingface/transformers/blob/a0ae2310ec46a2c592950babc85cf02e325bf6a7/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L336-L347
        """
        if self.config.use_parallel_residual:
            # @torch.compile()
            def block_forward(x: torch.Tensor, attention_mask: torch.Tensor,
                              prefix_masks: torch.Tensor) -> torch.Tensor:
                attn_output = mha_fw(x, res=x, attention_mask=attention_mask)

                # x = x + attn(ln1(x)) + mlp(ln2(x))
                # x_a = attn_output, 
                mlp_out = mlp_fw(x, res=attn_output)
                return mlp_out
        else:
            # @torch.compile()
            def block_forward(x: torch.Tensor, attention_mask: torch.Tensor,
                              prefix_masks: torch.Tensor) -> torch.Tensor:
                
                attn_output = mha_fw(x, res=x, attention_mask=attention_mask)
                
                # x = x + attn(ln1(x)) 
                # x = x + mlp(ln2(x))
                mlp_out = mlp_fw(attn_output, res=attn_output)
                return mlp_out

        self.block_forward = block_forward

    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval().half()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, f'pytorch_{layer_index}.pt',
            )))
        except Exception as e:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module
    
    def forward(self, x: torch.Tensor, layer_past=None, mask=None, **kargs) -> torch.Tensor:
        
        if mask is not None:
            # bool -> float
            attention_mask = 1e9*(mask[:, None, None, :]-1)
        else:
            attention_mask = None
                
        if self.training:
            
            if self.use_checkpoint:
                x.requires_grad_(True)
                x = checkpoint(self.block_forward, x, attention_mask, None)
            else:
                x = self.block_forward(x, attention_mask, None)
            
            return x
           
        else:
        
            residual = x
            ln_out = self.input_layernorm(x)
            attention_layer_outputs = self.attention(
                ln_out,
                attention_mask=attention_mask,
            )
            attn_output = attention_layer_outputs[0]  # output_attn: a, present, ...

            mlp_output = self.mlp(self.post_attention_layernorm(x))
            x = mlp_output + attn_output + residual

            return x
    
    
class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_lm_head.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module
        
    @torch.compile
    def forward(self, x, *args, **kargs):
        x = self.final_layer_norm(x)
        x = self.embed_out(x)
        return x