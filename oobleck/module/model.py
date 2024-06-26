import random
from typing import Any, Optional, Type

import torch
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForPreTraining,
    PretrainedConfig,
    PreTrainedModel,
    TrainingArguments,
)

from oobleck.module.sharding import get_split_points, shard_model

RANDOM_SEED = 42

# Oobleck has been tested only with the following models.
lang_models = ["gpt2", "gpt2-xl", "t5", "bert", "bloom"]
image_models = ["vit", "resnet", "clip", "swin"]

automodel_dict = {
    "gpt2": AutoModelForPreTraining,
    "gpt2-xl": AutoModelForCausalLM, # add LM head
    "t5": AutoModelForPreTraining,
    "bert": AutoModelForCausalLM,
    "bloom": AutoModelForPreTraining,
    "vit": AutoModelForImageClassification,
    "resnet": AutoModelForImageClassification,
    "clip": AutoModelForImageClassification,
    "swin": AutoModelForImageClassification,
}


class OobleckModel:
    """
    A wrapper model class of Hugging Face model
    downloaded from Hugging Face Hub (https://huggingface.co/models).

    It runs huggingface.utils.fx.symbolic_trace to get GraphModule
    and shard it to multiple GraphModules for pipeline execution.

    Model initialization must be done before distributed initialization.
    """

    def __init__(
        self,
        model_name: str,
        sample_inputs: dict[str, Any],
        training_args: Optional[TrainingArguments] = None,
        model_tag: Optional[str] = None,
        config_args: Optional[dict[str, Any]] = None,
    ):
        # Initialize CPU seed
        random.seed(RANDOM_SEED)
        torch.default_generator.manual_seed(RANDOM_SEED)

        if config_args is None:
            config_args = {}
        config_args["use_cache"] = False
        config_args["remove_unused_columns"] = False
        # necessary to register backward hooks
        config_args["return_dict"] = False

        # Use training_args for fp16/bf16
        model_config: PretrainedConfig = AutoConfig.from_pretrained(
            model_name, **config_args
        )
        model: Optional[Type[PreTrainedModel]] = None
        with init_empty_weights():
            for key, automodel in automodel_dict.items():
                if key in model_name:
                    model = automodel.from_config(model_config)
                    break

        assert model, f"Given model {model_name} is not supported yet."

        self.sample_inputs = sample_inputs
        self.trace_input_names = list(sample_inputs.keys())

        split_points = get_split_points(model_config)
        self.layers = shard_model(model, self.trace_input_names, split_points)
        self.model_name = model_name
        self.model_tag = model_tag

        self.total_num_params = sum(
            sum(p.numel() for p in layer.parameters()) for layer in self.layers
        )
        self.training_args = training_args
        self.model_args = model_config
