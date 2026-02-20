# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Union
from functools import partial
from contextlib import nullcontext
import torch
import torch._dynamo

from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron_patch.model.qwen2_moe.layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron_patch.model.qwen2_moe.transformer_config import core_transformer_config_from_args
from megatron_patch.arguments import get_patch_args
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from megatron_patch.data import train_valid_test_datasets_provider
from megatron.training import get_args, pretrain, print_rank_0, get_timers, print_rank_last, is_last_rank

torch._dynamo.config.suppress_errors = True


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    build_tokenizer(args)
    use_te = args.transformer_impl == "transformer_engine"
    assert use_te

    print_rank_0('building Qwen2-Megatron model ...')

    config = core_transformer_config_from_args(args)

    if args.num_experts:
        # Define the decoder block spec
        transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
    else:
        # Define the decoder layer spec
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                args.num_experts, args.moe_grouped_gemm,
                args.qk_layernorm, moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(
                args.num_experts, args.moe_grouped_gemm,
                args.qk_layernorm, moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm)

    build_model_context = nullcontext
    build_model_context_args = {}

    with build_model_context(**build_model_context_args):
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling
        )

    if args.only_train_list == 'all' and args.freeze_list == "":
        pass
    else:
        def _contain(string, keywords):
            for k in keywords: 
                if k in string: return True
            return False
        if args.only_train_list != 'all':
            only_train_list = args.only_train_list.split('+')
            print_rank_0(f"only_train_list: {only_train_list}")
            for n, p in model.named_parameters():
                if not _contain(n, only_train_list): 
                    print_rank_0(f"freeze parameter: {n}")
                    p.requires_grad = False
        else:
            freeze_list = args.freeze_list.split('+')
            print_rank_0(f"freeze_list: {freeze_list}")
            for n, p in model.named_parameters():
                if _contain(n, freeze_list): 
                    print_rank_0(f"freeze parameter: {n}")
                    p.requires_grad = False

#    if args.only_train_router:
#        for n, p in model.named_parameters():
#            if not 'router' in n: p.requires_grad = False
#    else:
#        for n, p in model.named_parameters():
#            if args.freeze_embedding and 'embedding' in n:
#                p.requires_grad = False
#            if args.freeze_attention and 'attention' in n:
#                p.requires_grad = False
#            if args.freeze_shared_expert and 'shared_experts' in n:
#                p.requires_grad = False

    return model

class CosineAnnealingScheduler:
    def __init__(self, eta_min, eta_max, T_max, global_batch_size):
        """
        初始化 Cosine Annealing 调度器。
        
        :param eta_min: 学习率的最小值
        :param eta_max: 学习率的最大值
        :param T_max: 一个周期的总 epoch 数
        """
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_max = T_max
        self.T_cur = 0
        self.global_batch_size = global_batch_size
        self.micro_batch = 0

    def step(self):
        """
        更新当前的 epoch，并返回新的学习率。
        """
        self.micro_batch = (self.micro_batch + 1) % self.global_batch_size
        if self.micro_batch == 0:
            self.T_cur += 1
        if self.T_cur > self.T_max:
            self.T_cur = 0  # 重置周期
        return self.get_value()

    def get_value(self):
        """
        根据余弦函数计算当前的学习率。
        """
        return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_max))

from megatron_patch.template.helper import get_batch, loss_func
from torch.optim.lr_scheduler import CosineAnnealingLR
def my_forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    args = get_args()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params = get_batch(data_iterator)
    timers("batch-generator").stop()
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)
    # update moe_aux_loss_coeff
    if args.moe_aux_loss_coeff_scheduler is not None and is_last_rank():
        if not hasattr(args, "__moe_aux_loss_coeff_scheduler"):
            data_parallel_num = torch.distributed.get_world_size() // args.pipeline_model_parallel_size // args.tensor_model_parallel_size
            args.__moe_aux_loss_coeff_scheduler = CosineAnnealingScheduler(T_max=args.train_iters, eta_min=0.001, eta_max=model.config.moe_aux_loss_coeff, global_batch_size=args.global_batch_size//args.micro_batch_size//data_parallel_num)
        args.__moe_aux_loss_coeff_scheduler.step()
        if args.__moe_aux_loss_coeff_scheduler.micro_batch == 0:
            model.config.moe_aux_loss_coeff = args.__moe_aux_loss_coeff_scheduler.get_value()
            print_rank_last(f"moe_aux_loss_coeff = {model.config.moe_aux_loss_coeff}, current_iter={args.__moe_aux_loss_coeff_scheduler.T_cur}")

    return output_tensor, partial(loss_func, loss_mask, num_seqs)


if __name__ == "__main__":
    from megatron_patch.template.helper import forward_step
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        my_forward_step,
        extra_args_provider=get_patch_args,
    )
