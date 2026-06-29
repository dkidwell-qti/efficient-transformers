# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional, Type

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration

from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo

BS = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
FBS = constants.ONNX_EXPORT_EXAMPLE_FBS
DEFAULT_PREFILL_SEQ_LEN = 384
DEFAULT_CTX_LEN = 1024


def _to_legacy_cache_if_needed(past_key_values):
    if isinstance(past_key_values, Cache):
        if hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values.to_legacy_cache()
        if hasattr(past_key_values, "layers"):
            legacy_cache = ()
            for layer in past_key_values.layers:
                legacy_cache += ((getattr(layer, "keys", None), getattr(layer, "values", None)),)
            return legacy_cache
    return past_key_values


class QEffPaliGemmaEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.model.vision_model = self.model.vision_tower

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.vision_tower.vision_model.encoder.layers[0].__class__}

    def forward(self, pixel_values):
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        # Avoid FP16 overflow in projector output before language merge.
        return image_features.clamp(-60000.0, 60000.0)


class QEffPaliGemmaDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.model.language_model
        self.config = self.model.config
        self.lm_head = self.model.lm_head

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.language_model.model.layers[0].__class__}

    def forward(
        self,
        input_ids,
        vision_embeds,
        position_ids,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
        batch_index: Optional[torch.LongTensor] = None,
    ):
        image_token_id = getattr(self.config, "image_token_index", self.config.image_token_id)

        if image_token_id >= self.config.text_config.vocab_size:
            special_image_mask = input_ids == image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        inputs_embeds = self.model.get_input_embeddings()(llm_input_ids)
        vision_embeds = vision_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        selected = input_ids == image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.shape[0], device=selected.device).view(-1, 1)
        vision_embeds_expanded = vision_embeds[indices0, indices1.clamp(min=0)]
        merged_embeds = torch.where(selected.unsqueeze(-1), vision_embeds_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, merged_embeds)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )

        next_image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_image_idx, next_image_idx, image_idx)

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()

        present = _to_legacy_cache_if_needed(outputs.past_key_values)
        return logits, vision_embeds, image_idx, present


class QEffPaliGemmaForConditionalGeneration(PaliGemmaForConditionalGeneration):
    def __qeff_init__(self):
        self.language_model = self.model.language_model

    def get_qeff_vision_encoder(self):
        return QEffPaliGemmaEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffPaliGemmaDecoderWrapper(self)

    def forward(
        self,
        input_ids,
        position_ids,
        pixel_values,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
    ):
        image_token_id = getattr(self.config, "image_token_index", self.config.image_token_id)

        if image_token_id >= self.config.text_config.vocab_size:
            special_image_mask = input_ids == image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        inputs_embeds = self.get_input_embeddings()(llm_input_ids)
        image_features = self.get_image_features(pixel_values=pixel_values)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_features = image_features.clamp(-60000.0, 60000.0)

        selected = input_ids == image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.shape[0], device=selected.device).view(-1, 1)
        image_features_expanded = image_features[indices0, indices1.clamp(min=0)]
        merged_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, merged_embeds)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            use_cache=True,
        )

        next_image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_image_idx, next_image_idx, image_idx)

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()

        present = _to_legacy_cache_if_needed(outputs.past_key_values)
        return logits, pixel_values, image_idx, present

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        prefill_seq_len = int(kwargs.get("prefill_seq_len") or DEFAULT_PREFILL_SEQ_LEN)
        ctx_len = int(kwargs.get("ctx_len") or DEFAULT_CTX_LEN)

        num_layers = self.config.text_config.num_hidden_layers
        num_key_value_heads = self.config.text_config.num_key_value_heads
        head_dim = self.config.text_config.hidden_size // self.config.text_config.num_attention_heads

        img_size = getattr(self.config.vision_config, "image_size", 224)
        vision_size = getattr(self.config, "image_seq_length", (img_size // self.config.vision_config.patch_size) ** 2)

        vision_inputs = {
            "pixel_values": torch.zeros((BS, 3, img_size, img_size), dtype=self.config.torch_dtype),
        }
        lang_inputs = {
            "input_ids": torch.ones((BS, prefill_seq_len), dtype=torch.int64),
            "attention_mask": torch.ones((BS, prefill_seq_len), dtype=torch.int64),
            "vision_embeds": torch.ones(
                (BS, vision_size, self.config.text_config.hidden_size),
                dtype=self.config.torch_dtype,
            ),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
        }

        image_token_id = getattr(self.config, "image_token_index", self.config.image_token_id)
        lang_inputs["input_ids"][:, :vision_size] = image_token_id

        lang_inputs["position_ids"] = lang_inputs.pop("attention_mask").cumsum(1)
        lang_inputs["past_key_values"] = []
        for _ in range(num_layers):
            lang_inputs["past_key_values"].append(
                (
                    torch.zeros(
                        FBS if continuous_batching else BS,
                        num_key_value_heads,
                        ctx_len,
                        head_dim,
                        dtype=self.config.torch_dtype,
                    ),
                    torch.zeros(
                        FBS if continuous_batching else BS,
                        num_key_value_heads,
                        ctx_len,
                        head_dim,
                        dtype=self.config.torch_dtype,
                    ),
                )
            )

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)

        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(BS).view(BS, 1)

        if kv_offload:
            return {"vision": vision_inputs, "lang": lang_inputs}

        lang_inputs.pop("vision_embeds")
        return {**vision_inputs, **lang_inputs}

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        max_num_images = compiler_options.pop("max_num_images", 1)
        prefill_seq_len = prefill_seq_len if prefill_seq_len else DEFAULT_PREFILL_SEQ_LEN
        ctx_len = ctx_len if ctx_len else DEFAULT_CTX_LEN
        img_size = img_size if img_size else getattr(self.config.vision_config, "image_size", 224)

        user_vision_size = compiler_options.pop("vision_size", None)
        if user_vision_size:
            if user_vision_size >= ctx_len:
                raise ValueError("vision_size must be less than ctx_len")
            vision_size = user_vision_size
        else:
            vision_size = getattr(
                self.config,
                "image_seq_length",
                (img_size // self.config.vision_config.patch_size) ** 2,
            )

        vision = [{"batch_size": batch_size, "max_num_images": max_num_images, "img_size": img_size}]

        if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
            lang = []
            for comp_ctx_lengths in comp_ctx_lengths_prefill:
                lang_prefill = {
                    "batch_size": 1 if continuous_batching else batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "comp_ctx_lengths": comp_ctx_lengths,
                    "max_num_images": max_num_images,
                    "img_size": img_size,
                    "vision_size": vision_size,
                    "vision_batch_size": batch_size,
                }
                if continuous_batching:
                    lang_prefill["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_prefill["batch_size"] = kv_cache_batch_size
                if full_batch_size:
                    lang_prefill["full_batch_exec_size"] = full_batch_size
                lang.append(lang_prefill)

            for comp_ctx_lengths in comp_ctx_lengths_decode:
                lang_decode = {
                    "batch_size": full_batch_size if continuous_batching else batch_size,
                    "seq_len": "1",
                    "ctx_len": ctx_len,
                    "comp_ctx_lengths": comp_ctx_lengths,
                    "max_num_images": max_num_images,
                    "img_size": img_size,
                    "vision_size": vision_size,
                    "vision_batch_size": batch_size,
                }
                if continuous_batching:
                    lang_decode["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_decode["batch_size"] = kv_cache_batch_size
                lang.append(lang_decode)
        else:
            lang_prefill = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "max_num_images": max_num_images,
                "img_size": img_size,
                "vision_size": vision_size,
                "vision_batch_size": batch_size,
            }
            if continuous_batching:
                lang_prefill["full_batch_size"] = kv_cache_batch_size
            else:
                lang_prefill["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                lang_prefill["full_batch_exec_size"] = full_batch_size

            lang_decode = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "max_num_images": max_num_images,
                "img_size": img_size,
                "vision_size": vision_size,
                "vision_batch_size": batch_size,
            }
            if continuous_batching:
                lang_decode["full_batch_size"] = kv_cache_batch_size
            else:
                lang_decode["batch_size"] = kv_cache_batch_size
            lang = [lang_prefill, lang_decode]

        if kv_offload:
            return {"vision": vision, "lang": lang}, compiler_options

        return lang, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        num_layers = self.config.text_config.num_hidden_layers

        vision_dynamic_axes = {"pixel_values": {0: "batch_size", 2: "img_size", 3: "img_size"}}
        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
        }
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}

        pkv_axes = {0: "full_batch_size" if continuous_batching else "batch_size", 2: "ctx_len"}
        for i in range(num_layers):
            lang_dynamic_axes[f"past_key.{i}"] = pkv_axes
            lang_dynamic_axes[f"past_value.{i}"] = pkv_axes

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        if kv_offload:
            return {"vision": vision_dynamic_axes, "lang": lang_dynamic_axes}

        return {**vision_dynamic_axes, **lang_dynamic_axes}

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            return {"vision": vision_output_names, "lang": lang_output_names}

        lang_output_names.insert(1, "pixel_values_RetainedState")
        lang_output_names.insert(2, "image_idx_output")
        return lang_output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=self.config.torch_dtype,
                shape=("batch_size", 3, "img_size", "img_size"),
            ),
        ]
