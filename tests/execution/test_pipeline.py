from __future__ import annotations

import pytest
import torch
import torch.distributed
from deepspeed.runtime.lr_schedules import WarmupLR
from torch.optim import AdamW

from tests.conftest import (
    TRAIN_BATCH_SIZE,
    OobleckDynamicClassFactory,
    OobleckMultiProcessTestCase,
    OobleckStaticClassFactory,
)


class TestOobleckSingleStagePipeline(OobleckMultiProcessTestCase):
    @staticmethod
    def _test_attributes_type(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
        num_gpus_per_node: int,
    ):
        model = factory.get_model()
        pipeline = dfactory.get_dummy_pipeline(
            num_stages=1, num_gpus_per_node=num_gpus_per_node
        )
        assert pipeline.communication.prev_rank is None
        assert pipeline.communication.next_rank is None

        # Because we only have one rank, it should execute all layers in the model
        assert len(pipeline.execution._layers) == len(model.model)

        assert isinstance(pipeline.execution._optimizer, AdamW)
        assert isinstance(pipeline.execution._lr_scheduler, WarmupLR)
        assert pipeline._global_step == 0

        for layer in pipeline.execution._layers:
            assert all(p.is_cuda for p in layer.parameters())

    @pytest.mark.parametrize(
        "num_gpus_per_node", [1, 2, 4], ids=["1gpu/node", "2gpus/node", "4gpus/node"]
    )
    def test_attributes_type(self, num_gpus_per_node: int):
        self.run_in_parallel(
            num_gpus_per_node,
            TestOobleckSingleStagePipeline._test_attributes_type,
            num_gpus_per_node,
        )

    @staticmethod
    def load_microbatch(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
        num_gpus_per_node: int,
    ):
        pipeline = dfactory.get_dummy_pipeline(
            num_stages=1, num_gpus_per_node=num_gpus_per_node
        )

        assert pipeline.pipe_buffers["inputs"][0] is None
        pipeline.execution.load_microbatch(buffer_id=0)
        assert isinstance(pipeline.pipe_buffers["inputs"][0], tuple)
        assert all(
            isinstance(tensor, torch.Tensor)
            for tensor in pipeline.pipe_buffers["inputs"][0]
        )
        # Check batch size is correct
        assert all(
            tensor.shape[0] == TRAIN_BATCH_SIZE
            for tensor in pipeline.pipe_buffers["inputs"][0]
        )

    @staticmethod
    def forward(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
        num_gpus_per_node: int,
    ):
        pipeline = dfactory.get_dummy_pipeline(
            num_stages=1, num_gpus_per_node=num_gpus_per_node
        )
        pipeline.execution.load_microbatch(buffer_id=0)

        assert pipeline.pipe_buffers["outputs"][0] is None
        assert pipeline.execution._loss is None
        assert pipeline.execution.total_loss is None
        pipeline.execution.forward_pass(buffer_id=0)
        # because it is the last stage, output should still be None
        # Instead, it should write loss and total_loss
        assert pipeline.pipe_buffers["outputs"][0] is None
        assert pipeline.execution._loss is not None
        assert pipeline.execution.total_loss is not None

    @staticmethod
    def backward(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
        num_gpus_per_node: int,
    ):
        pipeline = dfactory.get_dummy_pipeline(
            num_stages=1, num_gpus_per_node=num_gpus_per_node
        )
        pipeline.execution.load_microbatch(buffer_id=0)
        pipeline.execution.forward_pass(buffer_id=0)

        # backward_pass must clear outputs.
        # Inject a dummy value and check if it is cleared
        pipeline.pipe_buffers["outputs"][0] = torch.zeros(1)

        # before backward pass, check grad are none
        assert all(
            all(p.grad is None for p in layer.parameters())
            for layer in pipeline.execution._layers
        )

        pipeline.execution.backward_pass(buffer_id=0)

        # check if output is cleared by backward_pass
        assert pipeline.pipe_buffers["outputs"][0] is None

        # check gradients are generated by backward_pass
        assert all(
            all(
                p._grad is not None if p.requires_grad else None
                for p in layer.parameters()
            )
            for layer in pipeline.execution._layers
        )

    @staticmethod
    def optimizer_step(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
        num_gpus_per_node: int,
    ):
        pipeline = dfactory.get_dummy_pipeline(
            num_stages=1, num_gpus_per_node=num_gpus_per_node
        )
        pipeline.execution.load_microbatch(buffer_id=0)
        pipeline.execution.forward_pass(buffer_id=0)
        pipeline.execution.backward_pass(buffer_id=0)

        # optimizer must not have internal data for now
        for p in pipeline.execution._optimizer.param_groups[0]["params"]:
            assert len(pipeline.execution._optimizer.state[p]) == 0

        pipeline.execution.optimizer_step()

        # optimizer must have internal data for now
        for p in pipeline.execution._optimizer.param_groups[0]["params"]:
            assert len(pipeline.execution._optimizer.state[p]) > 0

    @pytest.mark.parametrize(
        "func_name",
        [
            "load_microbatch",
            "forward",
            "backward",
            "optimizer_step",
        ],
    )
    @pytest.mark.parametrize(
        "num_gpus_per_node", [1, 2, 4], ids=["1gpu/node", "2gpus/node", "4gpus/node"]
    )
    def test_execution(self, func_name: str, num_gpus_per_node: int):
        func = getattr(TestOobleckSingleStagePipeline, func_name)
        self.run_in_parallel(num_gpus_per_node, func, num_gpus_per_node)


class TestMultiStagePipeline(OobleckMultiProcessTestCase):
    @staticmethod
    def _four_stages(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
    ):
        model = factory.get_model()
        pipeline = dfactory.get_dummy_pipeline(4)
        assert pipeline.prev_rank == (
            None if dfactory._my_rank == 0 else dfactory._my_rank - 1
        )
        assert pipeline.next_rank == (
            None if dfactory._my_rank == 3 else dfactory._my_rank + 1
        )

        assert len(pipeline.execution._layers) < len(model.model)

        for layer in pipeline.execution._layers:
            assert all(p.is_cuda for p in layer.parameters())

        return (len(pipeline.execution._layers), len(model.model))

    def test_attributes_type(self):
        results = self.run_in_parallel(
            num_processes=4, func=TestMultiStagePipeline._four_stages
        )
        assert len(results) == 4
        # Check sum of number of stage layers equals total number of model layers
        assert sum(r[0] for r in results) == results[0][1]

    @staticmethod
    def send_recv_in_forward(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
    ):
        pipeline = dfactory.get_dummy_pipeline(4)
        rank = dfactory._my_rank

        assert pipeline.pipe_buffers["inputs"][0] is None
        assert pipeline.pipe_buffers["outputs"][0] is None

        assert pipeline.communication.sent_activation_meta is False
        assert pipeline.communication.activation_recv_buf is None
        assert pipeline.communication.grad_recv_buf is None

        if rank == 0:
            pipeline.execution.load_microbatch(buffer_id=0)
            assert pipeline.pipe_buffers["inputs"][0] is not None
            pipeline.execution.forward_pass(buffer_id=0)
            assert pipeline.pipe_buffers["outputs"][0] is not None
            pipeline.communication.send_activations(buffer_id=0)
        elif rank < 3:
            pipeline.communication.recv_activations(buffer_id=0)
            assert pipeline.pipe_buffers["inputs"][0] is not None
            pipeline.execution.forward_pass(buffer_id=0)
            assert pipeline.pipe_buffers["outputs"][0] is not None
            pipeline.communication.send_activations(buffer_id=0)
        elif rank == 3:
            pipeline.communication.recv_activations(buffer_id=0)
            assert pipeline.pipe_buffers["inputs"][0] is not None
            pipeline.execution.forward_pass(buffer_id=0)
            # The last stage: output should still be None
            # Instead, loss should be written
            assert pipeline.pipe_buffers["outputs"][0] is None
        else:
            raise RuntimeError("Invalid rank")

        if rank == 3:
            assert pipeline.execution.loss is not None
        else:
            assert pipeline.execution.loss is None
            assert pipeline.communication.sent_activation_meta is True

        if rank != 0:
            assert pipeline.communication.activation_recv_buf is not None

    @staticmethod
    def send_recv_in_backward(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
    ):
        pipeline = dfactory.get_dummy_pipeline(4)
        rank = dfactory._my_rank

        if rank == 0:
            pipeline.execution.load_microbatch(buffer_id=0)
        else:
            pipeline.communication.recv_activations(buffer_id=0)
        pipeline.execution.forward_pass(buffer_id=0)
        if rank < 3:
            pipeline.communication.send_activations(buffer_id=0)

        assert pipeline.communication.grad_recv_buf is None
        assert all(
            all(p.grad is None for p in layer.parameters())
            for layer in pipeline.execution._layers
        )

        # start backward
        if rank == 3:
            pipeline.execution.backward_pass(buffer_id=0)
            pipeline.communication.send_gradients(buffer_id=0)
        elif rank > 0:
            pipeline.communication.recv_gradients(buffer_id=0)
            assert pipeline.communication.grad_recv_buf is not None
            pipeline.execution.backward_pass(buffer_id=0)
            pipeline.communication.send_gradients(buffer_id=0)
        elif rank == 0:
            pipeline.communication.recv_gradients(buffer_id=0)
            assert pipeline.communication.grad_recv_buf is not None
            pipeline.execution.backward_pass(buffer_id=0)
        else:
            raise RuntimeError("Invalid rank")

        assert all(
            all(p.grad is not None for p in layer.parameters() if p.requires_grad)
            for layer in pipeline.execution._layers
        )

    @pytest.mark.parametrize(
        "func_name",
        [
            "send_recv_in_forward",
            "send_recv_in_backward",
        ],
    )
    def test_distributed_execution(self, func_name):
        func = getattr(TestMultiStagePipeline, func_name)
        self.run_in_parallel(num_processes=4, func=func)

    @pytest.mark.skip(reason="TODO")
    def test_pipeline_train(self):
        # TODO: test train() method with mock spy
        pass
