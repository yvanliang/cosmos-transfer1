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

from typing import Optional, Tuple

import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import nn
from torchvision import transforms

from cosmos_transfer1.diffusion.conditioner import DataType
from cosmos_transfer1.diffusion.module.parallel import split_inputs_cp
from cosmos_transfer1.diffusion.module.position_embedding import (
    LearnableEmb3D,
    LearnableEmb3D_FPS_Aware,
    LearnablePosEmbAxis,
    MultiCameraSinCosPosEmbAxis,
    MultiCameraVideoRopePosition3DEmb,
    SinCosPosEmb,
    SinCosPosEmb_FPS_Aware,
    SinCosPosEmbAxis,
    VideoRopePosition3DEmb,
    VideoRopePositionEmb,
)
from cosmos_transfer1.diffusion.training.modules.blocks import (
    GeneralDITTransformerBlock,
    PatchEmbed,
    SDXLTimestepEmbedding,
    SDXLTimesteps,
)
from cosmos_transfer1.diffusion.training.networks.general_dit import GeneralDIT
from cosmos_transfer1.diffusion.training.tensor_parallel import scatter_along_first_dim
from cosmos_transfer1.utils import log


class MultiCameraGeneralDIT(GeneralDIT):
    def __init__(
        self,
        *args,
        n_views: int = 3,
        n_views_emb: int = -1,
        view_condition_dim: int = 3,
        traj_condition_dim: int = 0,
        concat_view_embedding: bool = True,
        concat_traj_embedding: bool = False,
        add_repeat_frame_embedding: bool = False,
        **kwargs,
    ):
        if kwargs.get("add_augment_sigma_embedding", None) is not None:
            kwargs.pop("add_augment_sigma_embedding")
        self.n_views = n_views
        if n_views_emb < 0:
            self.n_views_emb = n_views
        else:
            self.n_views_emb = n_views_emb
        self.view_condition_dim = view_condition_dim
        self.concat_view_embedding = concat_view_embedding
        self.traj_condition_dim = traj_condition_dim
        self.concat_traj_embedding = concat_traj_embedding
        self.add_repeat_frame_embedding = add_repeat_frame_embedding

        super().__init__(*args, **kwargs)
        # reinit self.blocks
        del self.blocks  # 原本的模型未指定n_views，现在是n_views=3。而该参数只作用于ca，ca时将view合并到batch维度，这样每个view的图像token只会和对应视角的文本token计算注意力
        self.blocks = nn.ModuleDict()

        layer_mask = [False] * self.num_blocks if kwargs["layer_mask"] is None else kwargs["layer_mask"]
        assert (
            len(layer_mask) == self.num_blocks
        ), f"Layer mask length {len(layer_mask)} does not match num_blocks { self.num_blocks}"
        for idx in range(self.num_blocks):
            if layer_mask[idx]:
                continue
            self.blocks[f"block{idx}"] = GeneralDITTransformerBlock(
                x_dim=self.model_channels,
                context_dim=kwargs["crossattn_emb_channels"],
                num_heads=self.num_heads,
                block_config=self.block_config,
                window_sizes=(
                    kwargs["window_sizes"] if idx in kwargs["window_block_indexes"] else []
                ),  # There will be bug if using "WA-CA-MLP"
                mlp_ratio=kwargs["mlp_ratio"],
                spatial_attn_win_size=kwargs["spatial_attn_win_size"],
                temporal_attn_win_size=kwargs["temporal_attn_win_size"],
                x_format=self.block_x_format,
                use_adaln_lora=self.use_adaln_lora,
                adaln_lora_dim=self.adaln_lora_dim,
                n_views=self.n_views,
            )
        self.view_embeddings = nn.Embedding(self.n_views_emb, view_condition_dim)  # Learnable embedding layer

        if self.concat_traj_embedding:
            self.traj_embeddings = nn.Linear(192, self.traj_condition_dim)  # Learnable embedding layer
        if self.add_repeat_frame_embedding:
            self.repeat_frame_embedding = nn.Linear(1, view_condition_dim)  # Learnable embedding layer

        self.init_weights()

    def build_patch_embed(self):
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
            view_condition_dim,
            traj_condition_dim,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
            self.view_condition_dim,
            self.traj_condition_dim,
        )
        if self.concat_view_embedding:
            in_channels = in_channels + view_condition_dim if view_condition_dim > 0 else in_channels

        if self.concat_traj_embedding:
            in_channels = in_channels + traj_condition_dim if traj_condition_dim > 0 else in_channels

        in_channels = in_channels + 1 if concat_padding_mask else in_channels

        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
            keep_spatio=True,
            legacy_patch_emb=self.legacy_patch_emb,
        )

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        if self.legacy_patch_emb:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def build_pos_embed(self):
        if self.pos_emb_cls == "rope3d":
            cls_type = MultiCameraVideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        log.critical(f"Building positional embedding with {self.pos_emb_cls} class, impl {cls_type}")
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            n_views=self.n_views,
        )
        self.pos_embedder = cls_type(
            **kwargs,
        )

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = MultiCameraSinCosPosEmbAxis(
                **kwargs,
            )

    def forward_before_blocks(
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
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
        """
        trajectory = kwargs.get("trajectory", None)
        frame_repeat = kwargs.get("frame_repeat", None)

        del kwargs
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        original_shape = x.shape
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x,
            fps=fps,
            padding_mask=padding_mask,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
            trajectory=trajectory,
            frame_repeat=frame_repeat,
            view_indices_B_T=view_indices_B_T,
        )
        # logging affline scale information
        affline_scale_log_info = {}

        timesteps_B_D, adaln_lora_B_3D = self.t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

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
        affline_emb_B_D = self.affline_norm(affline_emb_B_D)

        # for logging purpose
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = affline_emb_B_D
        self.crossattn_emb = crossattn_emb
        self.crossattn_mask = crossattn_mask

        if self.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        if self.blocks["block0"].x_format == "THWBD":
            x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
            if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
                )
            crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")

            if crossattn_mask:
                crossattn_mask = rearrange(crossattn_mask, "B M -> M B")

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

        elif self.blocks["block0"].x_format == "BTHWD":
            x = x_B_T_H_W_D
        else:
            raise ValueError(f"Unknown x_format {self.blocks[0].x_format}")
        output = {
            "x": x,
            "affline_emb_B_D": affline_emb_B_D,
            "crossattn_emb": crossattn_emb,
            "crossattn_mask": crossattn_mask,
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_3D": adaln_lora_B_3D,
            "original_shape": original_shape,
            "extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        return output

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        trajectory: Optional[torch.Tensor] = None,
        frame_repeat: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the `self.pos_embedder`
                with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )

        if view_indices_B_T is None:
            view_indices = torch.arange(self.n_views).clamp(max=self.n_views_emb - 1)  # View indices [0, 1, ..., V-1]
            view_indices = view_indices.to(x_B_C_T_H_W.device)
            view_embedding = self.view_embeddings(view_indices)  # Shape: [V, embedding_dim]
            view_embedding = rearrange(view_embedding, "V D -> D V")
            view_embedding = (
                view_embedding.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)
            )  # Shape: [1, D, V, 1, 1, 1]
        else:
            view_indices_B_T = view_indices_B_T.clamp(max=self.n_views_emb - 1)
            view_indices_B_T = view_indices_B_T.to(x_B_C_T_H_W.device).long()
            view_embedding = self.view_embeddings(view_indices_B_T)  # B, (V T), D
            view_embedding = rearrange(view_embedding, "B (V T) D -> B D V T", V=self.n_views)
            view_embedding = view_embedding.unsqueeze(-1).unsqueeze(-1)  # Shape: [B, D, V, T, 1, 1]

        if self.add_repeat_frame_embedding:
            if frame_repeat is None:
                frame_repeat = (
                    torch.zeros([x_B_C_T_H_W.shape[0], view_embedding.shape[1]])
                    .to(view_embedding.device)
                    .to(view_embedding.dtype)
                )
            frame_repeat_embedding = self.repeat_frame_embedding(frame_repeat.unsqueeze(-1))
            frame_repeat_embedding = rearrange(frame_repeat_embedding, "B V D -> B D V")
            view_embedding = view_embedding + frame_repeat_embedding.unsqueeze(3).unsqueeze(4).unsqueeze(5)

        x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=self.n_views)
        view_embedding = view_embedding.expand(
            x_B_C_V_T_H_W.shape[0],
            view_embedding.shape[1],
            view_embedding.shape[2],
            x_B_C_V_T_H_W.shape[3],
            x_B_C_V_T_H_W.shape[4],
            x_B_C_V_T_H_W.shape[5],
        )  # Shape: [B, V, 3, t, H, W]
        if self.concat_traj_embedding:
            traj_emb = self.traj_embeddings(trajectory)
            traj_emb = traj_emb.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)
            traj_emb = traj_emb.expand(
                x_B_C_V_T_H_W.shape[0],
                traj_emb.shape[1],
                view_embedding.shape[2],
                x_B_C_V_T_H_W.shape[3],
                x_B_C_V_T_H_W.shape[4],
                x_B_C_V_T_H_W.shape[5],
            )  # Shape: [B, V, 3, t, H, W]

            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding, traj_emb], dim=1)
        else:
            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)

        x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, " B C V T H W -> B C (V T) H W", V=self.n_views)
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D, None, extra_pos_emb


class VideoExtendGeneralDIT(MultiCameraGeneralDIT):
    def __init__(self, *args, in_channels=16 + 1, add_augment_sigma_embedding=False, **kwargs):
        self.add_augment_sigma_embedding = add_augment_sigma_embedding
        # extra channel for video condition mask
        super().__init__(*args, in_channels=in_channels, **kwargs)
        log.info(f"VideoExtendGeneralDIT in_channels: {in_channels }")

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
        video_cond_bool: Optional[torch.Tensor] = None,
        condition_video_indicator: Optional[torch.Tensor] = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        condition_video_pose: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Args:
        condition_video_augment_sigma: (B) tensor of sigma value for the conditional input augmentation
        condition_video_pose: (B, 1, T, H, W) tensor of pose condition
        """
        B, C, T, H, W = x.shape

        if data_type == DataType.VIDEO:
            assert (
                condition_video_input_mask is not None
            ), "condition_video_input_mask is required for video data type; check if your model_obj is extend_model.FSDPDiffusionModel or the base DiffusionModel"
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
            input_list = [x, condition_video_input_mask]
            if condition_video_pose is not None:
                if condition_video_pose.shape[2] > T:
                    log.warning(
                        f"condition_video_pose has more frames than the input video: {condition_video_pose.shape} > {x.shape}"
                    )
                    condition_video_pose = condition_video_pose[:, :, :T, :, :].contiguous()
                input_list.append(condition_video_pose)
            x = torch.cat(
                input_list,
                dim=1,
            )

        return super().forward(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            condition_video_augment_sigma=condition_video_augment_sigma,
            view_indices_B_T=view_indices_B_T,
            **kwargs,
        )
