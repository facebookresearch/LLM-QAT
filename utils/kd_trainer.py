# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import inspect

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from . import utils
import torch
from apex import amp
from fairscale.nn.data_parallel import (
    FullyShardedDataParallel as FullyShardedDDP,
    ShardedDataParallel as ShardedDDP,
)
from fairscale.nn.wrap import auto_wrap
from torch import nn
from torch.nn import functional as F, MSELoss
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_pt_utils import (
    get_module_class_from_name,
)
from transformers.trainer_utils import FSDPOption, has_length, ShardedDDPOption
from transformers.utils import is_torch_neuroncore_available, logging

logger = logging.get_logger(__name__)
local_rank = utils.get_local_rank()

mse_loss = MSELoss()


class KDTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ce_loss(self, size_average, student_logits, teacher_logits):

        model_output_log_prob = F.log_softmax(student_logits, dim=2)
        real_output_soft = F.softmax(teacher_logits, dim=2)

        loss = F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")
        return loss

    def mse_loss(self, student_logits, teacher_logits):
        return mse_loss(student_logits, teacher_logits)

    def compute_loss_train(self, model, inputs, return_outputs=False):

        with torch.no_grad():
            teacher_outputs = model.teacher(
                **inputs
                # **inputs, output_hidden_states=True, output_attentions=True
            )
        teacher_logits = teacher_outputs.get("logits")
        del teacher_outputs

        # forward pass
        student_outputs = model(**inputs)
        # get attributes
        student_logits = student_outputs.get("logits")

        if not return_outputs:
            del student_outputs

        kd_loss = 0.0
        size_average = True
        if model.kd_loss_scale > 0.0:
            kd_loss = self.ce_loss(size_average, student_logits, teacher_logits)

        del teacher_logits
        del student_logits

        tok_loss = model.kd_loss_scale * kd_loss

        return (tok_loss, student_outputs) if return_outputs else tok_loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss_train(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = self.ipex_optimize_model(model, training, dtype=dtype)

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            model, self.optimizer = amp.initialize(
                model, self.optimizer, opt_level=self.args.fp16_opt_level
            )

        # Multi-gpu training (should be after apex fp16 initialization) / 8bit models does not support DDP
        if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
            model = nn.DataParallel(model)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        # if not training:
        #     return model

        # Distributed training (should be after apex fp16 initialization)
        if self.sharded_ddp is not None:
            # Sharded DDP!
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                model = ShardedDDP(model, self.optimizer)
            else:
                mixed_precision = self.args.fp16 or self.args.bf16
                cpu_offload = ShardedDDPOption.OFFLOAD in self.args.sharded_ddp
                zero_3 = self.sharded_ddp == ShardedDDPOption.ZERO_DP_3
                # XXX: Breaking the self.model convention but I see no way around it for now.
                if ShardedDDPOption.AUTO_WRAP in self.args.sharded_ddp:
                    model = auto_wrap(model)
                self.model = model = FullyShardedDDP(
                    model,
                    mixed_precision=mixed_precision,
                    reshard_after_forward=zero_3,
                    cpu_offload=cpu_offload,
                ).to(self.args.device)
        # Distributed training using PyTorch FSDP
        elif self.fsdp is not None:
            if not self.args.fsdp_config["xla"]:
                # PyTorch FSDP!
                from torch.distributed.fsdp.fully_sharded_data_parallel import (
                    CPUOffload,
                    FullyShardedDataParallel as FSDP,
                    MixedPrecision,
                )
                from torch.distributed.fsdp.wrap import (
                    size_based_auto_wrap_policy,
                    transformer_auto_wrap_policy,
                )

                if FSDPOption.OFFLOAD in self.args.fsdp:
                    cpu_offload = CPUOffload(offload_params=True)
                else:
                    cpu_offload = CPUOffload(offload_params=False)

                auto_wrap_policy = None

                if FSDPOption.AUTO_WRAP in self.args.fsdp:
                    if self.args.fsdp_config["fsdp_min_num_params"] > 0:
                        auto_wrap_policy = functools.partial(
                            size_based_auto_wrap_policy,
                            min_num_params=self.args.fsdp_config["fsdp_min_num_params"],
                        )
                    elif (
                        self.args.fsdp_config.get(
                            "fsdp_transformer_layer_cls_to_wrap", None
                        )
                        is not None
                    ):
                        transformer_cls_to_wrap = set()
                        for layer_class in self.args.fsdp_config[
                            "fsdp_transformer_layer_cls_to_wrap"
                        ]:
                            transformer_cls = get_module_class_from_name(
                                model, layer_class
                            )
                            if transformer_cls is None:
                                raise Exception(
                                    "Could not find the transformer layer class to wrap in the model."
                                )
                            else:
                                transformer_cls_to_wrap.add(transformer_cls)
                        auto_wrap_policy = functools.partial(
                            transformer_auto_wrap_policy,
                            # Transformer layer class to wrap
                            transformer_layer_cls=transformer_cls_to_wrap,
                        )
                mixed_precision_policy = None
                dtype = None
                if self.args.fp16:
                    dtype = torch.float16
                elif self.args.bf16:
                    dtype = torch.bfloat16
                if dtype is not None:
                    mixed_precision_policy = MixedPrecision(
                        param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
                    )
                if type(model) != FSDP:
                    # XXX: Breaking the self.model convention but I see no way around it for now.
                    signature = inspect.signature(FSDP.__init__).parameters.keys()
                    kwargs = {}
                    for arg in [
                        "limit_all_gathers",
                        "forward_prefetch",
                        "backward_prefetch",
                    ]:
                        if arg in signature:
                            kwargs[arg] = getattr(self, arg)
                    kwargs["limit_all_gathers"] = True
                    self.model = model = FSDP(
                        model,
                        sharding_strategy=self.fsdp,
                        cpu_offload=cpu_offload,
                        auto_wrap_policy=auto_wrap_policy,
                        mixed_precision=mixed_precision_policy,
                        device_id=self.args.device,
                        ignored_modules=None
                        if getattr(model, "teacher", None) is None
                        else [model.teacher],
                        **kwargs,
                    )

        elif self.args.local_rank != -1:
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
            if is_torch_neuroncore_available():
                return model
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank] if self.args._n_gpu != 0 else None,
                output_device=self.args.local_rank if self.args._n_gpu != 0 else None,
                **kwargs,
            )

        # torch.compile() needs to be called after wrapping the model with FSDP or DDP
        # to ensure that it accounts for the graph breaks required by those wrappers
        if self.args.torch_compile:
            model = torch.compile(
                model,
                backend=self.args.torch_compile_backend,
                mode=self.args.torch_compile_mode,
            )

        return model
