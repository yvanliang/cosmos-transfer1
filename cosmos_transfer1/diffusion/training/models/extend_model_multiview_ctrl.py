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

from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_transfer1.diffusion.conditioner import CosmosCondition, DataType, ViewConditionedVideoConditionerWithCtrl
from cosmos_transfer1.diffusion.diffusion.modules.res_sampler import COMMON_SOLVER_OPTIONS
from cosmos_transfer1.diffusion.inference.inference_utils import (
    merge_patches_into_video,
    non_strict_load_model,
    split_video_into_patches,
)
from cosmos_transfer1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_transfer1.diffusion.training.models.extend_model_multiview import FSDPExtendDiffusionModel
from cosmos_transfer1.diffusion.training.models.model import DiffusionModel as VideoDiffusionModel
from cosmos_transfer1.diffusion.training.models.model import _broadcast, broadcast_condition
from cosmos_transfer1.diffusion.training.models.model_image import diffusion_fsdp_class_decorator
from cosmos_transfer1.utils import log, misc
from cosmos_transfer1.utils.lazy_config import instantiate as lazy_instantiate

IS_PREPROCESSED_KEY = "is_preprocessed"


class MultiVideoDiffusionModelWithCtrl(FSDPExtendDiffusionModel):
    def __init__(self, config, fsdp_checkpointer=None):
        self.pixel_corruptor = None
        if fsdp_checkpointer is not None:
            return super().__init__(config, fsdp_checkpointer)
        else:
            return super().__init__(config)

    def build_model(self) -> torch.nn.ModuleDict:
        log.info("Start creating base model")
        base_model = super().build_model()
        # initialize base model
        config = self.config
        self.load_base_model(base_model)
        log.info("Done creating base model")

        log.info("Start creating ctrlnet model")
        net = lazy_instantiate(self.config.net_ctrl)
        conditioner = base_model.conditioner
        logvar = base_model.logvar
        # initialize controlnet encoder
        model = torch.nn.ModuleDict({"net": net, "conditioner": conditioner, "logvar": logvar})

        model.load_state_dict(base_model.state_dict(), strict=False)
        model.net.net_obj.load_state_dict_from_base_model(base_model.state_dict())

        model.base_model = base_model
        if not config.finetune_base_model:
            model.base_model.requires_grad_(False)
            log.critical("Only training ctrlnet model and keeping base model frozen")
        else:
            log.critical("Also training base model")
        log.info("Done creating ctrlnet model")

        self.hint_key = self.config.hint_key["hint_key"]
        return model

    @property
    def base_net(self):
        return self.model.base_model.net

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        super().on_train_start(memory_format)
        # self.base_model = self.base_model.to(memory_format=memory_format, **self.tensor_kwargs)
        self.model = self.model.to(memory_format=memory_format, **self.tensor_kwargs)
        if parallel_state.is_initialized() and parallel_state.get_tensor_model_parallel_world_size() > 1:
            if parallel_state.sequence_parallel:
                self.base_net.enable_sequence_parallel()
        if hasattr(self.config, "use_torch_compile") and self.config.use_torch_compile:  # compatible with old config
            # not tested yet
            if torch.__version__ < "2.3":
                log.warning(
                    "torch.compile in Pytorch version older than 2.3 doesn't work well with activation checkpointing.\n"
                    "It's very likely there will be no significant speedup from torch.compile.\n"
                    "Please use at least 24.04 Pytorch container, or imaginaire4:v7 container."
                )
            self.base_net = torch.compile(self.base_net, dynamic=False, disable=not self.config.use_torch_compile)

    def load_base_model(self, base_model) -> None:
        config = self.config
        if config.base_load_from is not None:
            checkpoint_path = config.base_load_from["load_path"]
        else:
            checkpoint_path = ""

        if "*" in checkpoint_path:
            # there might be better ways to decide if it's a converted tp checkpoint
            mp_rank = parallel_state.get_model_parallel_group().rank()
            checkpoint_path = checkpoint_path.replace("*", f"{mp_rank}")

        if checkpoint_path:
            log.info(f"Loading base model checkpoint (local): {checkpoint_path}", False)
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            log.success(f"Complete loading base model checkpoint (local): {checkpoint_path}", False)

            if state_dict.get("ema") is not None:
                # Copy the base model weights from ema model.
                log.info("Copying ema to base model", False)
                base_state_dict = {k.replace("-", "."): v for k, v in state_dict["ema"].items()}
            elif "model" in state_dict:
                # Copy the base model weights from reg model.
                log.warning("Using non-EMA base model", False)
                base_state_dict = state_dict["model"]
            else:
                log.info("Loading from an EMA only model", False)
                base_state_dict = state_dict
            try:
                base_model.load_state_dict(base_state_dict, strict=False)
            except Exception:
                log.critical("load model in non-strict mode", False)
                log.critical(non_strict_load_model(base_model, base_state_dict), rank0_only=False)
        log.info("Done loading the base model checkpoint.", False)

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor], **kwargs
    ) -> Tuple[Tensor, ViewConditionedVideoConditionerWithCtrl]:
        # process the control input
        hint_key = self.config.hint_key["hint_key"]
        is_image_batch = self.is_image_batch(data_batch)
        _data = {hint_key: data_batch[hint_key]}
        if IS_PREPROCESSED_KEY in data_batch:
            _data[IS_PREPROCESSED_KEY] = data_batch[IS_PREPROCESSED_KEY]
        if not is_image_batch:
            self._normalize_video_databatch_inplace(_data, input_key=hint_key)
        # if it is an image batch, the control input is also image
        if self.input_image_key in data_batch:
            self._augment_image_dim_inplace(_data, input_key=hint_key)
        data_batch[hint_key] = _data[hint_key]
        data_batch["hint_key"] = hint_key

        # process control_input_object
        hint_key_object = "control_input_object"
        is_image_batch = self.is_image_batch(data_batch)
        _data = {hint_key_object: data_batch[hint_key_object]}
        if IS_PREPROCESSED_KEY in data_batch:
            _data[IS_PREPROCESSED_KEY] = data_batch[IS_PREPROCESSED_KEY]
        if not is_image_batch:
            self._normalize_video_databatch_inplace(_data, input_key=hint_key_object)
        # if it is an image batch, the control input is also image
        if self.input_image_key in data_batch:
            self._augment_image_dim_inplace(_data, input_key=hint_key_object)
        data_batch[hint_key_object] = _data[hint_key_object]

        # process masked_video
        hint_key_masked_video = "control_input_masked_video"
        is_image_batch = self.is_image_batch(data_batch)
        _data = {hint_key_masked_video: data_batch[hint_key_masked_video]}
        if IS_PREPROCESSED_KEY in data_batch:
            _data[IS_PREPROCESSED_KEY] = data_batch[IS_PREPROCESSED_KEY]
        if not is_image_batch:
            self._normalize_video_databatch_inplace(_data, input_key=hint_key_masked_video)
        # if it is an image batch, the control input is also image
        if self.input_image_key in data_batch:
            self._augment_image_dim_inplace(_data, input_key=hint_key_masked_video)
        data_batch[hint_key_masked_video] = _data[hint_key_masked_video]

        raw_state, latent_state, condition = super(MultiVideoDiffusionModelWithCtrl, self).get_data_and_condition(
            data_batch, kwargs.get("num_condition_t", None)
        )
        # if not torch.is_grad_enabled() and all(self.config.hint_mask):
        use_multicontrol = (
            ("control_weight" in data_batch)
            and not isinstance(data_batch["control_weight"], float)
            and data_batch["control_weight"].shape[0] > 1
        )
        if use_multicontrol:  # encode individual conditions separately
            latent_hint = []
            num_conditions = data_batch[data_batch["hint_key"]].size(1) // 3
            for i in range(num_conditions):
                cond_mask = [False] * num_conditions
                cond_mask[i] = True
                latent_hint += [self.encode_latent(data_batch, cond_mask=cond_mask)]
            latent_hint = torch.cat(latent_hint)
        else:
            latent_hint = self.encode_latent(data_batch)
        latent_hint_object = self.encode_latent(data_batch, input_key=hint_key_object)
        latent_hint_masked_video = self.encode_latent(data_batch, input_key=hint_key_masked_video)

        # process mask
        comp_factor = self.vae.temporal_compression_factor
        mask = rearrange(data_batch["mask"], "B C (V T) H W -> B C V T H W", V=self.n_views)
        b, _, _, _, h, w = mask.shape
        mask_B_C_V_0 = mask[:, :, :, :1]
        mask_B_C_V_1T = mask[:, :, :, 1:-1:comp_factor]
        mask_B_C_V_T = torch.cat([mask_B_C_V_0, mask_B_C_V_1T], dim=3)
        mask = F.interpolate(
            rearrange(mask_B_C_V_T, "B C V T H W -> (B V T) C H W").float(),
            mode='nearest',
            size=(h // self.vae.spatial_compression_factor,
                  w // self.vae.spatial_compression_factor),
        )
        mask = rearrange(mask, '(B VT) C H W -> B C VT H W', B=b)
        condition.condition_video_input_mask *= (1. - mask)
        data_batch["mask"] = mask

        # copied from model.py
        is_image_batch = self.is_image_batch(data_batch)
        is_video_batch = not is_image_batch
        # VAE has randomness. CP/TP group should have the same encoded output.

        # TODO: (qsh 2024-08-23) the following may not be necessary!
        latent_hint = _broadcast(latent_hint, to_tp=True, to_cp=is_video_batch)
        latent_hint_object = _broadcast(latent_hint_object, to_tp=True, to_cp=is_video_batch)
        latent_hint_masked_video = _broadcast(latent_hint_masked_video, to_tp=True, to_cp=is_video_batch)
        data_batch["mask"] = _broadcast(data_batch["mask"], to_tp=True, to_cp=is_video_batch)
        condition = broadcast_condition(condition, to_tp=True, to_cp=is_video_batch)

        # add extra conditions
        data_batch["latent_hint"] = latent_hint
        data_batch["latent_hint_object"] = latent_hint_object
        data_batch["latent_hint_masked_video"] = latent_hint_masked_video
        setattr(condition, hint_key, latent_hint)
        setattr(condition, hint_key_object, latent_hint_object)
        setattr(condition, hint_key_masked_video, latent_hint_masked_video)
        setattr(condition, "base_model", self.model.base_model)
        return raw_state, latent_state, condition

    def encode_latent(self, data_batch: dict, cond_mask: list = [], input_key: str = None) -> torch.Tensor:
        input_key = data_batch["hint_key"] if input_key is None else input_key
        x = data_batch[input_key]
        if torch.is_grad_enabled() and self.pixel_corruptor is not None:
            x = self.pixel_corruptor(x)
        latent = []
        # control input goes through tokenizer, which always takes 3-input channels
        num_conditions = x.size(1) // 3  # input conditions were concatenated along channel dimension
        if num_conditions > 1 and self.config.hint_dropout_rate > 0:
            if torch.is_grad_enabled():  # during training, randomly dropout some conditions
                cond_mask = torch.rand(num_conditions) > self.config.hint_dropout_rate
                if not cond_mask.any():  # make sure at least one condition is present
                    cond_mask[torch.randint(num_conditions, (1,)).item()] = True
            elif not cond_mask:  # during inference, use hint_mask to indicate which conditions are used
                cond_mask = self.config.hint_mask
        else:
            cond_mask = [True] * num_conditions
        for idx in range(0, x.size(1), 3):
            x_rgb = x[:, idx : idx + 3]  # B C (V T) H W
            if self.config.hint_key["grayscale"]:
                x_rgb = x_rgb.mean(dim=1, keepdim=True).expand_as(x_rgb)
            # if idx == 0:
            #     x_max = x_rgb
            # else:
            #     x_max = torch.maximum(x_rgb, x_max)
            if not cond_mask[idx // 3]:  # if the condition is not selected, replace with a black image
                x_rgb = torch.zeros_like(x_rgb)
            latent.append(self.encode(x_rgb))
        # latent.append(self.encode(x_max))
        latent = torch.cat(latent, dim=1)
        return latent

    def compute_loss_with_epsilon_and_sigma(
        self,
        data_batch: dict[str, Tensor],
        x0_from_data_batch: Tensor,
        x0: Tensor,
        condition: CosmosCondition,
        epsilon: Tensor,
        sigma: Tensor,
    ):
        if self.is_image_batch(data_batch):
            # Turn off CP
            self.net.disable_context_parallel()
            self.base_net.disable_context_parallel()
        else:
            if parallel_state.is_initialized():
                if parallel_state.get_context_parallel_world_size() > 1:
                    # Turn on CP
                    cp_group = parallel_state.get_context_parallel_group()
                    self.net.enable_context_parallel(cp_group)
                    self.base_net.enable_context_parallel(cp_group)
                    log.debug("[CP] Split hint_input")
                    hint_key = self.config.hint_key["hint_key"]
                    x_hint_raw = getattr(condition, hint_key)
                    x_hint = split_inputs_cp(x=x_hint_raw, seq_dim=2, cp_group=self.net.cp_group)
                    setattr(condition, hint_key, x_hint)
        return super(MultiVideoDiffusionModelWithCtrl, self).compute_loss_with_epsilon_and_sigma(
            data_batch, x0_from_data_batch, x0, condition, epsilon, sigma
        )

    def get_x0_fn_from_batch_with_condition_latent(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        seed_inference: int = 1,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true
         condition_latent (torch.Tensor): latent tensor in shape B,C,T,H,W as condition to generate video.
        - num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        - condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        # data_batch should be the one processed by self.get_data_and_condition
        try:
            if "control_weight" not in data_batch and self.control_weight is not None:
                data_batch["control_weight"] = self.control_weight
                log.info(f"Setting control weight to {self.control_weight}")
            else:
                log.info(f"Control weight is {data_batch['control_weight']}")
        except Exception:
            pass

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        if condition.data_type == DataType.VIDEO and "view_indices" in data_batch:
            comp_factor = self.vae.temporal_compression_factor
            # n_frames = data_batch['num_frames']
            view_indices = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=self.n_views)
            view_indices_B_V_0 = view_indices[:, :, :1]
            view_indices_B_V_1T = view_indices[:, :, 1:-1:comp_factor]
            view_indices_B_V_T = torch.cat([view_indices_B_V_0, view_indices_B_V_1T], dim=-1)
            condition.view_indices_B_T = rearrange(view_indices_B_V_T, "B V T -> B (V T)", V=self.n_views)
            condition.data_n_views = self.n_views
            uncondition.view_indices_B_T = condition.view_indices_B_T
            uncondition.data_n_views = self.n_views

        if self.is_image_batch(data_batch):
            condition.data_type = DataType.IMAGE
            uncondition.data_type = DataType.IMAGE
        else:
            if condition_latent is None:
                condition_latent = torch.zeros(data_batch["latent_hint"].shape, **self.tensor_kwargs)
                num_condition_t = 0
                condition_video_augment_sigma_in_inference = 1000

            condition.video_cond_bool = True
            condition = self.add_condition_video_indicator_and_video_input_mask(
                condition_latent, condition, num_condition_t
            )
            if self.config.conditioner.video_cond_bool.add_pose_condition:
                condition = self.add_condition_pose(data_batch, condition)

            uncondition.video_cond_bool = True  # Not do cfg on condition frames
            uncondition = self.add_condition_video_indicator_and_video_input_mask(
                condition_latent, uncondition, num_condition_t
            )
            if self.config.conditioner.video_cond_bool.add_pose_condition:
                uncondition = self.add_condition_pose(data_batch, uncondition)

        # Add extra conditions for ctrlnet.
        latent_hint = data_batch["latent_hint"]
        hint_key = data_batch["hint_key"]
        setattr(condition, hint_key, latent_hint)
        if "use_none_hint" in data_batch and data_batch["use_none_hint"]:
            setattr(uncondition, hint_key, None)
        else:
            setattr(uncondition, hint_key, latent_hint)

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized() and not self.is_image_batch(data_batch):
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=True, to_cp=to_cp)
            cp_group = parallel_state.get_context_parallel_group()
            latent_hint = getattr(condition, hint_key)
            if latent_hint is not None:
                latent_hint = split_inputs_cp(latent_hint, seq_dim=2, cp_group=cp_group)
                setattr(condition, hint_key, latent_hint)
            if getattr(uncondition, hint_key) is not None:
                setattr(uncondition, hint_key, latent_hint)

        setattr(condition, "base_model", self.model.base_model)
        setattr(uncondition, "base_model", self.model.base_model)
        if hasattr(self, "hint_encoders"):
            self.model.net.hint_encoders = self.hint_encoders

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(
                noise_x,
                sigma,
                condition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            ).x0_pred_replaced
            uncond_x0 = self.denoise(
                noise_x,
                sigma,
                uncondition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            ).x0_pred_replaced
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Union[torch.Tensor, None] = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: float | None = None,
        add_input_frames_guidance: bool = False,
        return_noise: bool = False,
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Different from the base model, this function support condition latent as input, it will create a differnt x0_fn if condition latent is given.
        If this feature is stablized, we could consider to move this function to the base model.

        Args:
            condition_latent (Optional[torch.Tensor]): latent tensor in shape B,C,T,H,W as condition to generate video.
            num_condition_t (Optional[int]): number of condition latent T, if None, will use the whole first half
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        if is_image_batch:
            log.debug("image batch, call base model generate_samples_from_batch")
            return super(MultiVideoDiffusionModelWithCtrl, self).generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                seed=seed,
                state_shape=state_shape,
                n_sample=n_sample,
                is_negative_prompt=is_negative_prompt,
                num_steps=num_steps,
            )
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W
            else:
                log.debug(f"Default Video state shape is used. {self.state_shape}")
                state_shape = self.state_shape

        # assert condition_latent is not None, "condition_latent should be provided"

        # if self.net.is_context_parallel_enabled:
        #     data_batch["latent_hint"] = split_inputs_cp(x=data_batch["latent_hint"], seq_dim=2, cp_group=self.net.cp_group)

        x0_fn = self.get_x0_fn_from_batch_with_condition_latent(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            add_input_frames_guidance=add_input_frames_guidance,
            seed_inference=seed,
        )

        if sigma_max is None:
            sigma_max = self.sde.sigma_max

        if x_sigma_max is None:
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * sigma_max
            )

        if self.net.is_context_parallel_enabled:
            x_sigma_max = _broadcast(x_sigma_max, to_tp=True, to_cp=True)
            x_sigma_max = rearrange(x_sigma_max, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)
            x_sigma_max = rearrange(x_sigma_max, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        samples = self.sampler(
            x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=sigma_max, solver_option=solver_option
        )

        if self.net.is_context_parallel_enabled:
            samples = rearrange(samples, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
            samples = rearrange(samples, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        return samples

    @torch.no_grad()
    def validation_step(
        self, data: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        save generated videos
        """
        raw_data, x0, condition = self.get_data_and_condition(data)
        guidance = data["guidance"]
        sigma_max = data["sigma_max"]
        is_negative_prompt = data["is_negative_prompt"]
        data = misc.to(data, **self.tensor_kwargs)
        x_sigma_max = None
        if sigma_max is not None:
            x_sigma_max = self.get_x_from_clean(x0, sigma_max)
        sample = self.generate_samples_from_batch(
            data,
            guidance=guidance,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            x_sigma_max=x_sigma_max,
            sigma_max=sigma_max,
            is_negative_prompt=is_negative_prompt,
        )
        sample = self.decode(sample)
        gt = raw_data
        hint = data[data["hint_key"]][:, :3]
        result = torch.cat([hint, sample], dim=3)
        gt = torch.cat([hint, gt], dim=3)
        caption = data["ai_caption"]
        return {"gt": gt, "result": result, "caption": caption}, torch.tensor([0]).to(**self.tensor_kwargs)


@diffusion_fsdp_class_decorator
class FSDPMultiVideoDiffusionModelWithCtrl(MultiVideoDiffusionModelWithCtrl):
    pass
