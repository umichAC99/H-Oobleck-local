from __future__ import annotations

import copy

import torch
import torch.distributed
import torch.fx
from accelerate.utils.modeling import set_module_tensor_to_device
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleShardingStrategy


def is_checkpointable(layer: torch.fx.GraphModule) -> bool:
    if any(isinstance(m, torch.nn.Embedding) for _, m in layer.named_modules()):
        return False
    if any(isinstance(m, torch.nn.CrossEntropyLoss) for _, m in layer.named_modules()):
        return False
    if next(layer.parameters(), None) is None:
        return False
    return True


def init_tensors(layer: torch.fx.GraphModule, device: torch.device):
    """
    Initialize meta tensors and move it to GPU.
    TODO: must use checkpointed data
    """
    for param_name, param in layer.named_parameters():
        set_module_tensor_to_device(layer, param_name, device, torch.rand(param.shape))

    for buffer_name, buffer in layer.named_buffers():
        set_module_tensor_to_device(
            layer, buffer_name, device, torch.rand(buffer.shape)
        )


class Layer(torch.nn.Module):
    @classmethod
    def create_layer_from_layer(
        cls,
        existing_layer: Layer,
        process_group: torch.distributed.ProcessGroup,
    ) -> Layer:
        assert torch.distributed.get_rank(process_group) >= 0

        layer = cls.__new__(cls)
        layer._layer_id = existing_layer._layer_id
        layer._param_handle = existing_layer._param_handle
        return layer

    def remove_tensors(self):
        if self._param_handle.flat_param.grad is not None:
            self._param_handle.flat_param.grad.data = torch.tensor([])
        self._param_handle.flat_param.data = torch.tensor([])

    def __init__(
        self,
        layer_id: int,
        layer: torch.fx.GraphModule,
        process_group: torch.distributed.ProcessGroup,
    ):
        super().__init__()

        assert torch.distributed.get_rank(process_group) >= 0

        device = torch.device("cuda", torch.cuda.current_device())
        self._layer_id = layer_id
        layer = copy.deepcopy(layer)
        init_tensors(layer, device)
        if is_checkpointable(layer):
            layer = checkpoint_wrapper(layer)

        self._param_handle = FlatParamHandle(
            params=layer.parameters(),
            fully_sharded_module=layer,
            device=device,
            sharding_strategy=HandleShardingStrategy.NO_SHARD,
            offload_params=False,
            mp_param_dtype=torch.float32,  # TODO: change to bf16
            mp_reduce_dtype=torch.float32,
            keep_low_precision_grads=False,
            process_group=process_group,
            use_orig_params=False,
        )
        self._param_handle.shard()
        self._param_handle.init_flat_param_attributes()

    def forward(self, input: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        return self._param_handle._fully_sharded_module(*input)

    def backward(
        self,
        tensor: torch.Tensor
        | tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    ) -> None:
        if isinstance(tensor, torch.Tensor):
            loss = tensor
            loss.backward()
        else:
            output, gradients = tensor
            torch.autograd.backward(output, gradients)

    def reduce_gradients(self, process_groups: list[torch.distributed.ProcessGroup]):
        for process_group in process_groups:
            if torch.distributed.get_rank(process_group) < 0:
                continue
            torch.distributed.all_reduce(
                self._param_handle.flat_param.grad, group=process_group
            )