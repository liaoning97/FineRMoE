import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer, Qwen2MoeRMSNorm, Qwen2MoeRotaryEmbedding, Qwen2MoeMLP, QWEN2MOE_ATTENTION_CLASSES, Qwen2MoeModel, Qwen2MoeForCausalLM
from transformers.activations import ACT2FN
from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig 

def keep_max_and_mask(tensor):
    max_indices = torch.argmax(tensor, dim=-1, keepdim=True)  # (N, M, 1)
    mask = torch.zeros_like(tensor, dtype=torch.bool).scatter_(-1, max_indices, True)
    
    return mask

def finermoe_router(scores, num_tokens, num_experts, G_I, G_O, top_k, R_I, R_O):
    # scores [num_tokens, num_experts]
    assert num_experts % R_I % G_I % R_O % G_O == 0
    concat_shard_scores = scores.view(num_tokens, R_O * G_O, -1) # [num_token, R_O * G_O, R_I * G_I]
    _, add_shard_top_indices = torch.topk(concat_shard_scores, top_k//G_O, dim=-1)
    activate_add_shard_mask = torch.zeros_like(concat_shard_scores, dtype=torch.bool)
    activate_add_shard_mask.scatter_(-1, add_shard_top_indices, True)

    concat_shard_scores = torch.sum(concat_shard_scores, dim=-1) # [num_token, R_O * G_O]
    concat_shard_scores = concat_shard_scores.view(num_tokens, G_O, -1) # [num_token, G_O, R_O]
    activate_concat_shard_mask = keep_max_and_mask(concat_shard_scores) # [num_token, G_O, R_O]
    activate_concat_shard_mask = activate_concat_shard_mask.view(num_tokens, -1) # [num_token, G_O * R_O]
    activate_concat_shard_mask = activate_concat_shard_mask.unsqueeze(-1).repeat(1,1,R_I*G_I)
    
    final_mask = activate_add_shard_mask & activate_concat_shard_mask
    final_mask = final_mask.view(num_tokens, -1)

    masked_scores = scores.masked_fill(~final_mask.bool(), float('-inf'))
    probs, top_indices = torch.topk(masked_scores, k=top_k, dim=-1)
    
    return probs, top_indices

class FineRMoeConfig(Qwen2MoeConfig):
    model_type = "finer_moe"

class FineRMoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size//config.G_O, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
class FineRMoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.G_I = config.G_I
        self.G_O = config.G_O
        self.moe_concat_proj = config.moe_concat_proj
        self.R_I = config.R_I
        self.R_O = config.R_O

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FineRMoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        if self.moe_concat_proj:
            self.concat_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights, selected_experts = finermoe_router(router_logits, batch_size * sequence_length, self.num_experts, self.G_I, self.G_O, self.top_k, self.R_I, self.R_O)
        routing_weights = F.softmax(routing_weights, dim=1, dtype=torch.float)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero(as_tuple=True)[0].tolist()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            # final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            num_experts_per_concat_shard = self.R_O * self.G_I * self.R_I
            concat_shard_dim = hidden_dim // self.G_O
            final_hidden_states[:, (expert_idx//num_experts_per_concat_shard)*concat_shard_dim : ((expert_idx//num_experts_per_concat_shard)+1)*concat_shard_dim].index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        if self.moe_concat_proj:
            final_hidden_states = self.concat_proj(final_hidden_states)
            
        if self.shared_expert:
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
    
            final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

class FineRMoeDecoderLayer(Qwen2MoeDecoderLayer):
    def __init__(self, config: FineRMoeConfig, layer_idx: int):
        super(Qwen2MoeDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = QWEN2MOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = FineRMoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen2MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class FineRMoeModel(Qwen2MoeModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`FineRMoeDecoderLayer`]

    Args:
        config: FineRMoeConfig
    """
    config_class = FineRMoeConfig
    _no_split_modules = ["FineRMoeDecoderLayer"]

    def __init__(self, config: FineRMoeConfig):
        super(Qwen2MoeModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [FineRMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2MoeRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class FineRMoeForCausalLM(Qwen2MoeForCausalLM):
    config_class = FineRMoeConfig
    _no_split_modules = ["FineRMoeDecoderLayer"]

    def __init__(self, config):
        super(Qwen2MoeForCausalLM, self).__init__(config)
        self.model = FineRMoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

