# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ControlNet Encoder based on GeneralDIT
"""
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import nn
from torchvision import transforms

from cosmos_transfer1.diffusion.conditioner import DataType
from cosmos_transfer1.diffusion.module.blocks import zero_module
from cosmos_transfer1.diffusion.module.parallel import split_inputs_cp
from cosmos_transfer1.diffusion.training.modules.blocks import PatchEmbed
from cosmos_transfer1.diffusion.training.networks.general_dit_video_conditioned import (
    VideoExtendGeneralDIT as GeneralDIT,
)
from cosmos_transfer1.diffusion.training.tensor_parallel import scatter_along_first_dim
from cosmos_transfer1.utils import log


class GeneralDITEncoder(GeneralDIT):
    """
    ControlNet Encoder based on GeneralDIT. Heavily borrowed from GeneralDIT with minor modifications.
    """

    def __init__(self, *args, **kwargs):
        hint_channels = kwargs.pop("hint_channels", 16)
        self.dropout_ctrl_branch = kwargs.pop("dropout_ctrl_branch", 0.5)
        num_control_blocks = kwargs.pop("num_control_blocks", None)
        if num_control_blocks is not None:
            assert num_control_blocks > 0 and num_control_blocks <= kwargs["num_blocks"]
            kwargs["layer_mask"] = [False] * num_control_blocks + [True] * (kwargs["num_blocks"] - num_control_blocks)
        self.random_drop_control_blocks = kwargs.pop("random_drop_control_blocks", False)
        obj_net_kwargs = kwargs.pop("obj_ctrl")
        super().__init__(*args, **kwargs)
        num_blocks = self.num_blocks
        model_channels = self.model_channels
        layer_mask = kwargs.get("layer_mask", None)
        layer_mask = [False] * num_blocks if layer_mask is None else layer_mask
        self.layer_mask = layer_mask
        self.hint_channels = hint_channels
        self.build_hint_patch_embed()
        hint_nf = [16, 16, 32, 32, 96, 96, 256]
        nonlinearity = nn.SiLU()
        input_hint_block = [nn.Linear(model_channels, hint_nf[0]), nonlinearity]
        for i in range(len(hint_nf) - 1):
            input_hint_block += [nn.Linear(hint_nf[i], hint_nf[i + 1]), nonlinearity]
        self.input_hint_block = nn.Sequential(*input_hint_block)
        # Initialize weights
        self.init_weights()
        self.zero_blocks = nn.ModuleDict()
        for idx in range(num_blocks):
            if layer_mask[idx]:
                continue
            self.zero_blocks[f"block{idx}"] = zero_module(nn.Linear(model_channels, model_channels))
        self.input_hint_block.append(zero_module(nn.Linear(hint_nf[-1], model_channels)))
        self.net_obj = ObjectDITEncoder(
            *args,
            **(kwargs | dict(obj_net_kwargs)),
        )

    def _set_sequence_parallel(self, status: bool):
        self.zero_blocks.sequence_parallel = status
        self.input_hint_block.sequence_parallel = status
        super()._set_sequence_parallel(status)

    def build_hint_patch_embed(self):
        concat_padding_mask, in_channels, patch_spatial, patch_temporal, model_channels = (
            self.concat_padding_mask,
            self.hint_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder2 = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
            keep_spatio=True,
            legacy_patch_emb=self.legacy_patch_emb,
        )

        if self.legacy_patch_emb:
            # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.x_embedder2.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def prepare_hint_embedded_sequence(
        self, x_B_C_T_H_W: torch.Tensor, fps: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[2], 1, 1)],
                dim=1,
            )

        x_B_T_H_W_D = self.x_embedder2(x_B_C_T_H_W)

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps)

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D, None

    def encode_hint(
        self,
        hint: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
    ) -> torch.Tensor:
        assert hint.size(1) <= self.hint_channels, f"Expected hint channels <= {self.hint_channels}, got {hint.size(1)}"
        if hint.size(1) < self.hint_channels:
            padding_shape = list(hint.shape)
            padding_shape[1] = self.hint_channels - hint.size(1)
            hint = torch.cat([hint, torch.zeros(*padding_shape, dtype=hint.dtype, device=hint.device)], dim=1)
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."

        hint_B_T_H_W_D, _ = self.prepare_hint_embedded_sequence(hint, fps=fps, padding_mask=padding_mask)

        hint = rearrange(hint_B_T_H_W_D, "B T H W D -> T H W B D")
        if self.sequence_parallel:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            T, H, W, B, D = hint.shape
            hint = hint.view(T * H * W, 1, 1, B, -1)
            hint = scatter_along_first_dim(hint, tp_group)

        guided_hint = self.input_hint_block(hint)
        return guided_hint

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        hint_key: Optional[str] = None,
        base_model: Optional[nn.Module] = None,
        control_weight: Optional[float] = 1.0,
        num_layers_to_use: Optional[int] = -1,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
        """
        # record the input as they are replaced in this forward
        obj_ctrl = self.net_obj.forward(
            x=kwargs["control_input_pristine"],
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            hint_key="control_input_degraded",
            base_model=base_model,
            control_weight=control_weight,
            num_layers_to_use=num_layers_to_use,
            condition_video_input_mask=condition_video_input_mask,
            **kwargs,
        )
        condition_video_input_mask = None

        x_input = x
        crossattn_emb_input = crossattn_emb
        crossattn_mask_input = crossattn_mask
        condition_video_input_mask_input = condition_video_input_mask

        hint = kwargs.pop(hint_key)
        if hint is None:
            print("using none hint")
            return base_model.net.forward(
                x=x_input,
                timesteps=timesteps,
                crossattn_emb=crossattn_emb_input,
                crossattn_mask=crossattn_mask_input,
                fps=fps,
                image_size=image_size,
                padding_mask=padding_mask,
                scalar_feature=scalar_feature,
                data_type=data_type,
                condition_video_input_mask=condition_video_input_mask_input,
                **kwargs,
            )
        if hasattr(self, "hint_encoders"):  # for multicontrol
            guided_hints = []
            for i in range(hint.shape[1]):
                self.input_hint_block = self.hint_encoders[i].input_hint_block
                self.pos_embedder = self.hint_encoders[i].pos_embedder
                self.x_embedder2 = self.hint_encoders[i].x_embedder2
                guided_hints += [self.encode_hint(hint[:, i], fps=fps, padding_mask=padding_mask, data_type=data_type)]
        else:
            guided_hints = self.encode_hint(hint, fps=fps, padding_mask=padding_mask, data_type=data_type)
            guided_hints = torch.chunk(guided_hints, hint.shape[0] // x.shape[0], dim=3)
            # Only support multi-control at inference time
            assert len(guided_hints) == 1 or not torch.is_grad_enabled()

        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."

        B, C, T, H, W = x.shape
        if data_type == DataType.VIDEO:
            if condition_video_input_mask is not None:
                if self.cp_group is not None:
                    condition_video_input_mask = split_inputs_cp(
                        condition_video_input_mask, seq_dim=2, cp_group=self.cp_group
                    )
                input_list = [x, condition_video_input_mask]
                x = torch.cat(input_list, dim=1)

        elif data_type == DataType.IMAGE:
            # For image, we dont have condition_video_input_mask, or condition_video_pose
            # We need to add the extra channel for video condition mask
            padding_channels = self.in_channels - x.shape[1]
            x = torch.cat([x, torch.zeros((B, padding_channels, T, H, W), dtype=x.dtype, device=x.device)], dim=1)
        else:
            assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels, got {x.shape[1]}"

        self.crossattn_emb = crossattn_emb
        self.crossattn_mask = crossattn_mask

        if self.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")
        if crossattn_mask:
            crossattn_mask = rearrange(crossattn_mask, "B M -> M B")

        outs = {}

        # (Experimental, not used in the released model) if also training base model, sometimes drop the
        # controlnet branch to only train base branch. This is to prevent the network become dependent on
        # controlnet branch and make control weight useless.
        is_training = torch.is_grad_enabled()
        is_training_base_model = any(p.requires_grad for p in base_model.parameters())
        if is_training and is_training_base_model:
            coin_flip = torch.rand(B).to(x.device) > self.dropout_ctrl_branch  # prob for only training base model
            if self.blocks["block0"].x_format == "THWBD":
                coin_flip = coin_flip[None, None, None, :, None]
            elif self.blocks["block0"].x_format == "BTHWD":
                coin_flip = coin_flip[:, None, None, None, None]
        else:
            coin_flip = 1

        num_control_blocks = self.layer_mask.index(True)
        num_layers_to_use = num_control_blocks
        control_gate_per_layer = [i < num_layers_to_use for i in range(num_control_blocks)]

        if isinstance(control_weight, torch.Tensor):
            if control_weight.ndim == 0:  # Single scalar tensor
                control_weight = [float(control_weight)] * len(guided_hints)
            elif control_weight.ndim == 1:  # List of scalar weights
                control_weight = [float(w) for w in control_weight]
            else:  # Spatial-temporal weight maps
                control_weight = [w for w in control_weight]  # Keep as tensor
        else:
            control_weight = [control_weight] * len(guided_hints)

        x_before_blocks = x.clone()
        for i, guided_hint in enumerate(guided_hints):
            x = x_before_blocks
            if hasattr(self, "hint_encoders"):  # for multicontrol
                blocks = self.hint_encoders[i].blocks
                zero_blocks = self.hint_encoders[i].zero_blocks
                t_embedder = self.hint_encoders[i].t_embedder
                affline_norm = self.hint_encoders[i].affline_norm
                self.x_embedder = self.hint_encoders[i].x_embedder
                self.extra_pos_embedder = self.hint_encoders[i].extra_pos_embedder
            else:
                blocks = self.blocks
                zero_blocks = self.zero_blocks
                t_embedder = self.t_embedder
                affline_norm = self.affline_norm

            x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
                x, fps=fps, padding_mask=padding_mask
            )
            # logging affline scale information
            affline_scale_log_info = {}

            timesteps_B_D, adaln_lora_B_3D = t_embedder(timesteps.flatten())
            affline_emb_B_D = timesteps_B_D
            affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

            if scalar_feature is not None:
                raise NotImplementedError("Scalar feature is not implemented yet.")

            if self.additional_timestamp_channels:
                additional_cond_B_D = self.prepare_additional_timestamp_embedder(
                    bs=x.shape[0],
                    fps=fps,
                    h=image_size[:, 0],
                    w=image_size[:, 1],
                    org_h=image_size[:, 2],
                    org_w=image_size[:, 3],
                )

                affline_emb_B_D += additional_cond_B_D
                affline_scale_log_info["additional_cond_B_D"] = additional_cond_B_D.detach()

            affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
            affline_emb_B_D = affline_norm(affline_emb_B_D)

            # for logging purpose
            self.affline_scale_log_info = affline_scale_log_info
            self.affline_emb = affline_emb_B_D

            x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
            if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
                )

            if self.sequence_parallel:
                tp_group = parallel_state.get_tensor_model_parallel_group()
                # Sequence parallel requires the input tensor to be scattered along the first dimension.
                assert self.block_config == "FA-CA-MLP"  # Only support this block config for now
                T, H, W, B, D = x.shape
                # variable name x_T_H_W_B_D is no longer valid. x is reshaped to THW*1*1*b*D and will be reshaped back in FinalLayer
                x = x.view(T * H * W, 1, 1, B, D)
                assert x.shape[0] % parallel_state.get_tensor_model_parallel_world_size() == 0
                x = scatter_along_first_dim(x, tp_group)
                if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.view(
                        T * H * W, 1, 1, B, D
                    )
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = scatter_along_first_dim(
                        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, tp_group
                    )

            for idx, (name, block) in enumerate(blocks.items()):
                assert (
                    blocks["block0"].x_format == block.x_format
                ), f"First block has x_format {blocks[0].x_format}, got {block.x_format}"
                x = block(
                    x,
                    affline_emb_B_D,
                    crossattn_emb,
                    crossattn_mask,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                    adaln_lora_B_3D=adaln_lora_B_3D,
                    extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                )
                if guided_hint is not None:
                    x = x + guided_hint
                    guided_hint = None

                gate = control_gate_per_layer[idx]
                if isinstance(control_weight[i], (float, int)) or control_weight[i].ndim < 2:
                    hint_val = zero_blocks[name](x) * control_weight[i] * coin_flip * gate
                else:  # Spatial-temporal weights [num_controls, B, 1, T, H, W]
                    control_feat = zero_blocks[name](x)
                    # Get current feature dimensions
                    weight_map = control_weight[i]  # [B, 1, T, H, W]
                    # Reshape to match THWBD format
                    weight_map = weight_map.permute(2, 3, 4, 0, 1)  # [T, H, W, B, 1]
                    weight_map = weight_map.view(T * H * W, 1, 1, B, 1)

                    if self.sequence_parallel:
                        weight_map = scatter_along_first_dim(weight_map, tp_group)

                    hint_val = control_feat * weight_map * coin_flip * gate

                if name not in outs:
                    outs[name] = hint_val
                else:
                    outs[name] += hint_val

        output = base_model.net.forward(
            x=x_input,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb_input,
            crossattn_mask=crossattn_mask_input,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            x_ctrl=outs,
            obj_ctrl=obj_ctrl,
            condition_video_input_mask=condition_video_input_mask_input,
            **kwargs,
        )
        return output


class ObjectDITEncoder(GeneralDIT):
    """
        ControlNet Encoder based on GeneralDIT. Heavily borrowed from GeneralDIT with minor modifications.
        """

    def __init__(self, *args, in_channels, is_extend_model=False, **kwargs):
        self.is_extend_model = is_extend_model
        if is_extend_model:
            new_input_channels = in_channels + 1
            log.info(f"Updating input channels to {new_input_channels} to accomodate cond_mask")
        else:
            new_input_channels = in_channels

        if kwargs.get("add_augment_sigma_embedding", None) is not None:
            self.add_augment_sigma_embedding = kwargs.pop("add_augment_sigma_embedding")
        else:
            self.add_augment_sigma_embedding = False
        self.dropout_ctrl_branch = kwargs.pop("dropout_ctrl_branch", 0.5)
        num_control_blocks = kwargs.pop("num_control_blocks", None)
        if num_control_blocks is not None:
            assert num_control_blocks > 0 and num_control_blocks <= kwargs["num_blocks"]
            kwargs["layer_mask"] = [False] * num_control_blocks + [True] * (
                        kwargs["num_blocks"] - num_control_blocks)
        self.random_drop_control_blocks = kwargs.pop("random_drop_control_blocks", False)

        num_ctrl_blocks = kwargs.pop("num_ctrl_blocks")
        kwargs['layer_mask'] = ([False] * num_ctrl_blocks + [True] * (kwargs["num_blocks"] - num_ctrl_blocks))
        super().__init__(*args, in_channels=new_input_channels, **kwargs)

        original_x_embedder = self.x_embedder
        super().build_patch_embed()
        self.x_embedder2 = self.x_embedder
        self.x_embedder = original_x_embedder

        num_blocks = self.num_blocks
        model_channels = self.model_channels
        layer_mask = kwargs.get("layer_mask", None)
        layer_mask = [False] * num_blocks if layer_mask is None else layer_mask
        self.layer_mask = layer_mask

        # Initialize weights
        self.init_weights()
        self.zero_blocks = nn.ModuleDict()
        for idx in range(num_blocks):
            if layer_mask[idx]:
                continue
            self.zero_blocks[f"block{idx}"] = zero_module(nn.Linear(model_channels, model_channels))

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        hint_key: Optional[str] = None,
        base_model: Optional[nn.Module] = None,
        control_weight: Optional[float] = 1.0,
        num_layers_to_use: Optional[int] = -1,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
        """
        # record the input as they are replaced in this forward
        frame_repeat = kwargs.get("frame_repeat", None)
        hint = kwargs.pop(hint_key)

        B, C, T, H, W = x.shape
        assert condition_video_input_mask is not None
        if self.cp_group is not None:
            condition_video_input_mask = rearrange(
                condition_video_input_mask, "B C (V T) H W -> B C V T H W", V=self.n_views
            )
            condition_video_input_mask = split_inputs_cp(
                condition_video_input_mask, seq_dim=3, cp_group=self.cp_group
            )
            condition_video_input_mask = rearrange(
                condition_video_input_mask, "B C V T H W -> B C (V T) H W", V=self.n_views
            )
        x_list = [x, condition_video_input_mask]
        x = torch.cat(
            x_list,
            dim=1,
        )

        hint_list = [hint, condition_video_input_mask]
        hint = torch.cat(
            hint_list,
            dim=1,
        )

        self.crossattn_emb = crossattn_emb
        self.crossattn_mask = crossattn_mask

        if self.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        if self.blocks["block0"].x_format == "THWBD":
            crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")
            if crossattn_mask:
                crossattn_mask = rearrange(crossattn_mask, "B M -> M B")

        outs = {}

        # If also training base model, sometimes drop the controlnet branch to only train base branch.
        # This is to prevent the network become dependent on controlnet branch and make control weight useless.
        is_training = torch.is_grad_enabled()
        is_training_base_model = False
        if is_training and is_training_base_model:
            coin_flip = torch.rand(B).to(x.device) > self.dropout_ctrl_branch  # prob for only training base model
            if self.blocks["block0"].x_format == "THWBD":
                coin_flip = coin_flip[None, None, None, :, None]
            elif self.blocks["block0"].x_format == "BTHWD":
                coin_flip = coin_flip[:, None, None, None, None]
        else:
            coin_flip = 1

        num_control_blocks = self.layer_mask.index(True)
        if self.random_drop_control_blocks:
            if is_training:  # Use a random number of layers during training.
                num_layers_to_use = np.random.randint(num_control_blocks) + 1
            elif num_layers_to_use == -1:  # Evaluate using all the layers.
                num_layers_to_use = num_control_blocks
            else:  # Use the specified number of layers during inference.
                pass
        else:  # Use all of the layers.
            num_layers_to_use = num_control_blocks
        control_gate_per_layer = [i < num_layers_to_use for i in range(num_control_blocks)]

        blocks = self.blocks
        zero_blocks = self.zero_blocks
        t_embedder = self.t_embedder
        affline_norm = self.affline_norm

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x,
            fps=fps,
            padding_mask=padding_mask,
        )

        hint_B_T_H_W_D, _, _ = self.prepare_embedded_sequence(
            hint,
            fps=fps,
            padding_mask=padding_mask,
        )

        # logging affline scale information
        affline_scale_log_info = {}

        timesteps_B_D, adaln_lora_B_3D = t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

        if scalar_feature is not None:
            raise NotImplementedError("Scalar feature is not implemented yet.")

        if self.additional_timestamp_channels:
            additional_cond_B_D = self.prepare_additional_timestamp_embedder(
                bs=x.shape[0],
                fps=fps,
                h=image_size[:, 0],
                w=image_size[:, 1],
                org_h=image_size[:, 2],
                org_w=image_size[:, 3],
            )

            affline_emb_B_D += additional_cond_B_D
            affline_scale_log_info["additional_cond_B_D"] = additional_cond_B_D.detach()

        affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
        affline_emb_B_D = affline_norm(affline_emb_B_D)

        # for logging purpose
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = affline_emb_B_D

        x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
        hint = rearrange(hint_B_T_H_W_D, "B T H W D -> T H W B D")
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
            )

        if self.sequence_parallel:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            # Sequence parallel requires the input tensor to be scattered along the first dimension.
            assert self.block_config == 'FA-MA-CA-MLP'  # Only support this block config for now
            T, H, W, B, D = x.shape
            # variable name x_T_H_W_B_D is no longer valid. x is reshaped to THW*1*1*b*D and will be reshaped back in FinalLayer
            x = x.view(T * H * W, 1, 1, B, D)
            hint = hint.view(T * H * W, 1, 1, B, D)

            assert x.shape[0] % parallel_state.get_tensor_model_parallel_world_size() == 0
            x = scatter_along_first_dim(x, tp_group)

            assert hint.shape[0] % parallel_state.get_tensor_model_parallel_world_size() == 0
            hint = scatter_along_first_dim(hint, tp_group)

            if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.view(
                    T * H * W, 1, 1, B, D
                )
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = scatter_along_first_dim(
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, tp_group
                )

        for idx, (name, block) in enumerate(blocks.items()):
            assert (
                blocks["block0"].x_format == block.x_format
            ), f"First block has x_format {blocks[0].x_format}, got {block.x_format}"
            x, hint = block(
                x,
                affline_emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                hint=hint,
            )

            gate = control_gate_per_layer[idx]
            if isinstance(control_weight, (float, int)) or control_weight.ndim < 2:
                hint_val = zero_blocks[name](x) * control_weight * coin_flip * gate
            else:  # Spatial-temporal weights [num_controls, B, 1, T, H, W]
                control_feat = zero_blocks[name](x)

                # Get current feature dimensions
                if self.blocks["block0"].x_format == "THWBD":
                    weight_map = control_weight # [B, 1, T, H, W]

                    if weight_map.shape[2:5] != (T, H, W):
                        assert weight_map.shape[2] == 8 * (T - 1) + 1
                        weight_map_i = [
                            torch.nn.functional.interpolate(
                                weight_map[:, :, :1, :, :],
                                size=(1, H, W),
                                mode="trilinear",
                                align_corners=False,
                            )
                        ]
                        for wi in range(1, weight_map.shape[2], 8):
                            weight_map_i += [
                                torch.nn.functional.interpolate(
                                    weight_map[:, :, wi : wi + 8],
                                    size=(1, H, W),
                                    mode="trilinear",
                                    align_corners=False,
                                )
                            ]
                        weight_map = torch.cat(weight_map_i, dim=2)

                    # Reshape to match THWBD format
                    weight_map = weight_map.permute(2, 3, 4, 0, 1)  # [T, H, W, B, 1]
                    weight_map = weight_map.view(T * H * W, 1, 1, B, 1)
                    if self.sequence_parallel:
                        weight_map = scatter_along_first_dim(weight_map, tp_group)

                else:  # BTHWD format
                    raise NotImplementedError("BTHWD format for weight map is not implemented yet.")
                hint_val = control_feat * weight_map * coin_flip * gate

            if name not in outs:
                outs[name] = hint_val
            else:
                outs[name] += hint_val

        return outs

    def load_state_dict_from_base_model(self, base_model_state_dict: dict):
        """
        将预训练的 base_model 权重加载到 object_net (控制分支) 中。
        此版本会最后打印出未能加载的权重（missing_keys）。

        object_net.blocks.blockX.blocks[0] (VideoAttn) ← base_model.net.blocks.blockX.blocks[0] (VideoAttn)
        object_net.blocks.blockX.blocks[1] (MultiAttn) ← base_model.net.blocks.blockX.blocks[0] (VideoAttn) （特殊规则）
        object_net.blocks.blockX.blocks[2] (VideoAttn) ← base_model.net.blocks.blockX.blocks[1] (VideoAttn)
        object_net.blocks.blockX.blocks[3] (GPT2FeedForward) ← base_model.net.blocks.blockX.blocks[2] (GPT2FeedForward)

        Args:
            base_model_state_dict (OrderedDict): 从预训练 base_model 加载的 state_dict。
                                                 注意：这个 state_dict 应该包含 'net' 前缀，
                                                 例如 'net.x_embedder...'.
        """

        object_net_state_dict = self.state_dict()
        loaded_keys = set()
        for key, value in base_model_state_dict.items():
            if not key.startswith('net.'):
                continue

            base_key = key[4:]
            match = re.match(r'blocks\.block(\d+)\.blocks\.(\d+)\.(.*)', base_key)

            if match:
                block_num = match.group(1)
                sub_block_num = int(match.group(2))
                rest_of_key = match.group(3)

                if sub_block_num == 0:
                    target_key_0 = f'blocks.block{block_num}.blocks.0.{rest_of_key}'
                    if target_key_0 in object_net_state_dict:
                        object_net_state_dict[target_key_0] = value.clone()
                        loaded_keys.add(target_key_0)

                    target_key_1 = f'blocks.block{block_num}.blocks.1.{rest_of_key}'
                    if target_key_1 in object_net_state_dict:
                        object_net_state_dict[target_key_1] = value.clone()
                        loaded_keys.add(target_key_1)

                elif sub_block_num == 1:
                    target_key_2 = f'blocks.block{block_num}.blocks.2.{rest_of_key}'
                    if target_key_2 in object_net_state_dict:
                        object_net_state_dict[target_key_2] = value.clone()
                        loaded_keys.add(target_key_2)

                elif sub_block_num == 2:
                    target_key_3 = f'blocks.block{block_num}.blocks.3.{rest_of_key}'
                    if target_key_3 in object_net_state_dict:
                        object_net_state_dict[target_key_3] = value.clone()
                        loaded_keys.add(target_key_3)

                continue

            # 3. 统一处理 x_embedder -> x_embedder 和 x_embedder2 的映射
            if base_key.startswith('x_embedder.'):
                target_key_1 = base_key
                target_key_2 = base_key.replace('x_embedder.', 'x_embedder2.', 1)
                target_keys_to_process = {target_key_1, target_key_2}

                for target_key in target_keys_to_process:
                    if target_key in object_net_state_dict:
                        source_tensor = value
                        target_tensor = object_net_state_dict[target_key]
                        if source_tensor.shape != target_tensor.shape:
                            log.critical(f"Padding unmatched tensor: {target_key} "
                                         f" (source tensor: {source_tensor.shape}, "
                                         f" target tensor: {target_tensor.shape})", False)
                            new_tensor = torch.zeros_like(target_tensor)
                            slice_obj = tuple(slice(0, min(s_dim, t_dim))
                                              for s_dim, t_dim in zip(source_tensor.shape, target_tensor.shape))
                            new_tensor[slice_obj] = source_tensor[slice_obj]
                            object_net_state_dict[target_key] = new_tensor
                        else:
                            object_net_state_dict[target_key] = source_tensor.clone()
                        loaded_keys.add(target_key)

                continue

            if base_key in object_net_state_dict:
                if object_net_state_dict[base_key].shape == value.shape:
                    object_net_state_dict[base_key] = value.clone()
                    loaded_keys.add(base_key)
                else:
                    log.critical(f" [警告] obj_ctrl与base_model尺寸不匹配，跳过权重: {base_key} "
                          f" (目标尺寸: {object_net_state_dict[base_key].shape}, "
                          f" 源尺寸: {value.shape})", False)

        self.load_state_dict(object_net_state_dict)
        all_object_net_keys = set(self.state_dict().keys())
        missing_keys = all_object_net_keys - loaded_keys

        assert all(['zero_blocks' in key for key in missing_keys]), \
            "There are unexpected missing keys in the object_net. " \
            "Please check the mapping logic or the base_model state_dict."
        log.success(f"Complete loading obj_ctrl model checkpoint from the base_model, total: {len(loaded_keys)}", False)
