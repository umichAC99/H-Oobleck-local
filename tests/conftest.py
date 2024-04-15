from __future__ import annotations

import copy
import logging
import math
import multiprocessing as mp
import random
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import deepspeed.comm as dist
import pytest
import torch
import torch.distributed
from pytest_mock import MockerFixture
from transformers.training_args import TrainingArguments

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResult,
    LayerExecutionResults,
    PipelineTemplate,
    StageExecutionResult,
    HeteroNodeSpec,
    NodeConfig,
)
from oobleck.execution.dataloader import LoaderType, OobleckDataLoader
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEP = 4

logging.basicConfig(level=logging.INFO)


@dataclass
class Model:
    model_name: str
    dataset_path: str
    dataset_name: str | None = None


datasets: dict[str, tuple[str, (str | None)]] = {
    "gpt2": ("wikitext", "wikitext-2-raw-v1"),
    "microsoft/resnet-50": ("Maysee/tiny-imagenet", None),
}

models_to_test: dict[str, Model] = {
    "gpt2": Model("gpt2", "wikitext", "wikitext-2-raw-v1"),
    # "microsoft/resnet-50": Model("microsoft/resnet-50", "Maysee/tiny-imagenet"),
}

# Add model arguments here, if it is needed.
model_args: dict[str, dict[str, int] | None] = {
    "gpt2": {
        "num_hidden_layers": 32,
        "n_positions": 1024,
        "n_embd": 1024,
        "n_head": 16,
    },
    "microsoft/resnet-50": None,
}


@pytest.fixture(scope="session", params=list(models_to_test.keys()))
def model_name_fixture(request: pytest.FixtureRequest) -> str:
    return request.param


class OobleckStaticClassFactory:
    """
    Oobleck Class Factory that create classes for testing.
    "Static" here means that it is not relevant to Oobleck dynamic reconfiguration
    and fixed once a class object is created.
    """

    def __init__(self, model_name: str, test_directory: Path):
        self._model_data: Model = models_to_test[model_name]
        self._training_args = TrainingArguments(
            output_dir=test_directory,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEP,
        )

        self._dataset: OobleckDataset | None = None
        self._model: OobleckModel | None = None
        self._dataloader: OobleckDataLoader | None = None
        self._profile: LayerExecutionResults | None = None
        self._pipeline_templates: dict[int, PipelineTemplate] = {}

    def get_dataset(self) -> OobleckDataset:
        if not self._dataset:
            self._dataset = OobleckDataset(
                self._model_data.model_name,
                self._model_data.dataset_path,
                self._model_data.dataset_name,
            )
        return self._dataset

    def get_model(self) -> OobleckModel:
        self.get_dataset()

        if not self._model:
            self._model = OobleckModel(
                self._model_data.model_name,
                self._dataset.sample,
                self._training_args,
                "test",
                model_args.get(self._model_data.model_name, None),
            )

        return self._model
    
    # Takes in a list of LayerExecutionResults and a HeteroNodeSpec and returns a tuple of the number of nodes, number of gpus per node, and a list of scaling factors
    def dummy_node_folding(self, profiles: list[LayerExecutionResults], node_spec: HeteroNodeSpec) -> tuple[int, int, list[float]]:
        forward_cost_sums = [
            sum([layer._forward for layer in profile.get()]) for profile in profiles
        ]
        max_forward_cost = max(forward_cost_sums)
        scaling_factors = [max_forward_cost / forward_cost for forward_cost in forward_cost_sums]
        num_nodes = 0.0
        for i in range(len(node_spec._node_specs)):
            num_nodes += node_spec._node_specs[i]._num_nodes * scaling_factors[i]
        
        return (int(num_nodes), node_spec._node_specs[0]._num_gpus, scaling_factors)

    def get_dummy_profile(self) -> LayerExecutionResults:
        self.get_model()

        if not self._profile:
            num_layers = len(self._model.layers)

            results: list[LayerExecutionResult] = []
            for index in range(num_layers):
                results.append(
                    LayerExecutionResult(
                        layer_index=index,
                        forward=abs(random.random()) + 1.0,
                        backward=abs(random.random() * 3) + 1.0,
                        allreduce_in_node={i + 1: random.random() for i in range(8)},
                        allreduce_across_nodes={
                            i + 1: random.random() * 4 for i in range(64)
                        },
                        mem_required=(1024, 1024),
                    )
                )

            self._profile = LayerExecutionResults(results)

        return self._profile
    
    def get_dummy_hetero_profile(self) -> list[LayerExecutionResults]:
        self.get_model()

        results: list[LayerExecutionResults] = []
        for i in range(3):
            num_layers = len(self._model.layers)

            layer_results: list[LayerExecutionResult] = []
            for index in range(num_layers):
                layer_results.append(
                    LayerExecutionResult(
                        layer_index=index,
                        forward=abs(random.random())+1.0,
                        backward=abs(random.random() * 3)+1.0,
                        allreduce_in_node={i + 1: random.random() for i in range(8)},
                        allreduce_across_nodes={
                            i + 1: random.random() * 4 for i in range(64)
                        },
                        mem_required=(1024, 1024),
                    )
                )

            results.append(LayerExecutionResults(layer_results))

        return results
    
    def get_dummy_hetero_node_spec(self, is_random: bool=False, seed: int=0, num_nodes: int=5, num_types: int=3) -> HeteroNodeSpec:
        hetero_spec = None

        # can add more specs in the pool
        # navie reference (not considering mem, bandwidth):
        # https://lambdalabs.com/gpu-benchmarks
        spec_pool : dict[str, float] = {
            "gtx_1080ti": 0.6, # approx.
            "v_100_16gb": 1.0,
            "rtx_3090_24gb": 1.8,
            "rtx_4090_24gb": 2.94,
            "a_100_80gb_pcie": 4.41,
            "h_100_80gb_pcie": 5.45,
        }
        assert num_nodes > 0, "Must have at least 1 node"
        assert num_types <= len(list(spec_pool.keys())), "Cannot choose more types than we have"

        if is_random:
            random.seed(seed)
            # Randomly select device types from the spec pool
            chosed_type = random.sample(list(spec_pool.keys()), num_types)
            computer_power = [spec_pool[i] for i in chosed_type]

            num_hetero_nodes = [0]
            for _ in range(len(chosed_type)-1):
                num_hetero_nodes.append(random.randint(1, (num_nodes-1) - sum(num_hetero_nodes)))
            num_hetero_nodes.append(num_nodes - sum(num_hetero_nodes))
            num_hetero_nodes = num_hetero_nodes[1:]
            random.shuffle(num_hetero_nodes)

            assert sum(num_hetero_nodes) == num_nodes
            
            num_device_per_node = []
            # Randomly select the number of devices per node from [1, 2, 4, 8]
            for _ in range(len(num_hetero_nodes)):
                num_device_per_node.append(random.choice([1, 2, 4, 8]))
        else:
            chosed_type = ["gtx_1080ti", "v_100_16gb", "rtx_4090_24gb"]
            num_hetero_nodes = [1, 2, 2]
            num_device_per_node = [2, 2, 2]
            computer_power = [spec_pool[i] for i in chosed_type]

        assert len(num_device_per_node) == len(num_hetero_nodes) == len(chosed_type) == len(computer_power)
        hetero_spec = HeteroNodeSpec(
            [
                NodeConfig(*i) for i in zip(chosed_type, num_hetero_nodes, num_device_per_node, computer_power)
            ]
        )
        return hetero_spec
      
    def get_dummy_profile_by_scaling(self, node_spec: HeteroNodeSpec) -> list[LayerExecutionResults]:
        #Assume node_spec[0] is the weakest one
        result = []
        weakest_layer_results = self.get_dummy_profile().get()
        for i in range(0, len(weakest_layer_results)):
            weakest_layer_results[i] = LayerExecutionResult(
                layer_index=weakest_layer_results[i]._index,
                forward=weakest_layer_results[i]._forward,
                backward=weakest_layer_results[i]._backward,
                allreduce_in_node=weakest_layer_results[i]._allreduce_in_node,
                allreduce_across_nodes=weakest_layer_results[i]._allreduce_across_nodes,
                mem_required=weakest_layer_results[i]._mem_required
            )
        result.append(LayerExecutionResults(weakest_layer_results))
        
        # Iterate over other node types, append scaled layer results to result by computing power
        for i in range(1, len(node_spec._node_specs)):
            layer_results = [layer for layer in weakest_layer_results]
            for j in range(len(layer_results)):
                layer_results[j] = LayerExecutionResult(
                    layer_index=layer_results[j]._index,
                    forward=layer_results[j]._forward * 1.0/node_spec._node_specs[i]._compute_power,
                    backward=layer_results[j]._backward * 1.0/node_spec._node_specs[i]._compute_power,
                    allreduce_in_node=layer_results[j]._allreduce_in_node,
                    allreduce_across_nodes=layer_results[j]._allreduce_across_nodes,
                    mem_required=layer_results[j]._mem_required
                )
            result.append(LayerExecutionResults(layer_results))
        return result

    def get_dummy_pipeline_template(
        self,
        num_stages: int,
        num_gpus_per_node: int,
        num_nodes: int = 1,
    ) -> PipelineTemplate:
        self.get_dummy_profile()

        def slice_layers(lst: list[Any], num_chunks: int) -> list[tuple[int, int]]:
            if num_chunks > len(lst):
                raise ValueError(
                    f"Cannot slice {len(list)} layers into {num_chunks} chunks."
                )

            length_chunk = math.ceil(len(lst) / num_chunks)
            slicing_points: list[tuple[int, int]] = []
            for i in range(0, len(lst), length_chunk):
                end = i + length_chunk if i + length_chunk < len(lst) else len(lst)
                slicing_points.append((i, end))
            return slicing_points

        def split_gpus(num_nodes: int, num_gpus: int, num_stages: int) -> list[int]:
            # num_gpus_per_stage -> num_stages with the num_gpus_per_stage
            num_gpus_per_stage: dict[int, int] = defaultdict(int)
            num_gpus_per_stage[1] = num_nodes * num_gpus

            while sum(num_gpus_per_stage.values()) > num_stages:
                min_num_gpus_per_stage = min(
                    [n for n in num_gpus_per_stage.keys() if num_gpus_per_stage[n] >= 2]
                )
                num_gpus_per_stage[min_num_gpus_per_stage] -= 2
                num_gpus_per_stage[min_num_gpus_per_stage * 2] += 1

            assert (
                sum([k * v for k, v in num_gpus_per_stage.items()])
                == num_nodes * num_gpus
            )

            subnumbers: list[int] = []
            for num_gpus in sorted(num_gpus_per_stage.keys()):
                subnumbers.extend([num_gpus] * num_gpus_per_stage[num_gpus])

            return subnumbers

        assert num_stages >= num_nodes, "num_stages must be greater than num_nodes."
        assert (
            num_stages <= num_nodes * num_gpus_per_node
        ), "num_stages must be less than or equal to num_nodes * num_gpus_per_node."

        key = (num_stages, num_nodes, num_gpus_per_node)
        if key not in self._pipeline_templates:
            layer_indices = slice_layers(self._profile.get(), num_stages)
            num_gpus_per_stage = split_gpus(num_nodes, num_gpus_per_node, num_stages)

            assert len(layer_indices) == len(num_gpus_per_stage)

            stages = [
                StageExecutionResult(self._profile, indices, num_gpus)
                for indices, num_gpus in zip(layer_indices, num_gpus_per_stage)
            ]

            self._pipeline_templates[key] = PipelineTemplate(
                stages,
                0.1,
                self._profile.size,
                num_nodes,
                num_gpus_per_node,
            )

        return self._pipeline_templates[key]


class OobleckDynamicClassFactory:
    """
    Oobleck Class Factory that create classes for testing.
    "Dynamic" here means that the internal states are changed during training.
    Thus the class object should be created every time a new state is needed.
    """

    def __init__(
        self, static_factory: OobleckStaticClassFactory, my_rank: int, ranks: list[int]
    ):
        assert dist.is_initialized()
        assert torch.distributed.is_initialized()

        self._static_factory = static_factory
        self._my_rank = my_rank
        self._ranks = ranks

    def get_dataloader(
        self,
        pipeline_index: int,
        num_microbatches: list[int],
        num_iterations: int = 0,
    ) -> OobleckDataLoader:
        dataset = self._static_factory.get_dataset()
        training_args = self._static_factory._training_args

        return OobleckDataLoader(
            args=training_args,
            datasets=dataset,
            dataloader_type=LoaderType.Training,
            pipeline_index=pipeline_index,
            num_microbatches=num_microbatches,
            num_iterations_done=num_iterations,
            epoch=0,
            shuffle=False,
        )

    def get_dummy_pipeline(
        self,
        num_stages: int,
        num_nodes: int = 1,
        num_gpus_per_node: int = 1,
    ) -> OobleckPipeline:
        model = copy.deepcopy(self._static_factory.get_model())
        # TODO: make this more flexible
        template = self._static_factory.get_dummy_pipeline_template(
            num_stages=num_stages,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
        )
        training_args = self._static_factory._training_args
        dataloader = self.get_dataloader(0, [training_args.gradient_accumulation_steps])

        pipeline = OobleckPipeline(
            pipeline_id=0,
            pipeline_template=template,
            ranks=self._ranks,
            dataloader=dataloader,
            step=0,
            training_args=training_args,
        )

        pipeline.initialize_distributed_fsdp()
        pipeline.initialize_distributed_pipeline()
        pipeline.initialize_execution(model)

        return pipeline


@pytest.fixture(scope="session")
def factory(
    model_name_fixture: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> OobleckStaticClassFactory:
    directory = tmp_path_factory.mktemp(
        f"single_process_{model_name_fixture.replace('/', '-')}"
    )
    return OobleckStaticClassFactory(model_name_fixture, directory)


class OobleckSingleProcessTestCase:
    """
    A base class for Oobleck test cases that run in a single process.
    Test cases for functionalities of static classes will inherit this class.
    """

    factory: OobleckStaticClassFactory

    @pytest.fixture(scope="function", autouse=False)
    def distributed(self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
        assert not dist.is_initialized() and not torch.distributed.is_initialized()

        # envs required by deepspeed.comm
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")

        # Initialize a single process torch.distributed group.
        store = torch.distributed.HashStore()
        torch.distributed.init_process_group(
            backend="nccl", store=store, rank=0, world_size=1
        )
        dist.init_distributed(dist_backend="nccl", dist_init_required=False)
        assert torch.distributed.is_initialized()
        assert dist.is_initialized()

        yield

        dist.destroy_process_group()
        dist.cdb = None
        assert not torch.distributed.is_initialized()
        assert not dist.is_initialized()

    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(
        cls,
        model_name_fixture: str,
        tmp_path_factory: pytest.TempPathFactory,
        class_mocker: MockerFixture,
        request: pytest.FixtureRequest,
    ):
        with pytest.MonkeyPatch().context() as monkeypatch:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
            class_mocker.patch("torch.cuda.device_count", return_value=1)
            directory = tmp_path_factory.getbasetemp()
            request.cls.factory = OobleckStaticClassFactory(
                model_name_fixture, directory
            )
            yield


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires 4 GPUs")
class OobleckMultiProcessTestCase:
    """
    A base class for Oobleck test cases that run in multiple processes in parallel.
    Test cases for functionalities of dynamic classes will inherit this class.
    """

    @staticmethod
    def _worker_init(
        queue: mp.Queue,
        rank: int,
        world_size: int,
        model_name: str,
        directory: Path,
        test: callable,
        *args,
    ):
        # Very careful initialization dependency due to too many third-party libraries.
        # As we use torch.distributed.FileStore for distributed initialization, it doesn't require
        # os envs (MASTER_ADDR, MASTER_PORT), while deepspeed and HuggingFace by default use them.
        # Thus, initialize StaticClassFactory (which relies on HF) first without the envs.
        # Then, initialize distributed and deepspeed.
        # After that, create dynamic class factory since it requires distributed configuration.
        try:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(rank))
            monkeypatch.delenv("RANK", raising=False)
            monkeypatch.delenv("WORLD_SIZE", raising=False)
            monkeypatch.delenv("MASTER_ADDR", raising=False)
            monkeypatch.delenv("MASTER_PORT", raising=False)

            patcher = patch("torch.cuda.device_count", return_value=1)
            patcher.start()

            factory = OobleckStaticClassFactory(model_name, directory)

            monkeypatch.setenv("RANK", str(rank))
            monkeypatch.setenv("WORLD_SIZE", str(world_size))
            torch.cuda.set_device(0)

            store = torch.distributed.FileStore(
                str(directory.joinpath("store")), world_size
            )
            torch.distributed.init_process_group(
                backend="nccl", store=store, rank=rank, world_size=world_size
            )
            dist.init_distributed(dist_backend="nccl", dist_init_required=False)

            dynamic_factory = OobleckDynamicClassFactory(
                factory, rank, list(range(world_size))
            )

            result = test(factory, dynamic_factory, *args)

            queue.put(
                {
                    "success": (result if result is not None else ""),
                    "rank": rank,
                }
            )
        except Exception as e:
            queue.put({"error": str(e) + "\n" + traceback.format_exc()})

    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(
        cls,
        model_name_fixture: str,
        tmp_path_factory: pytest.TempPathFactory,
        request: pytest.FixtureRequest,
    ):
        request.cls.model_name = model_name_fixture
        request.cls.tmp_path_factory = tmp_path_factory

    model_name: str
    tmp_path_directory: pytest.TempPathFactory

    def run_in_parallel(
        self, num_processes: int, func: callable, *args
    ) -> list[str | None]:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        tmp_directory = self.tmp_path_factory.mktemp(func.__qualname__, numbered=True)
        logging.info(f"Using directory {tmp_directory}")

        processes: list[mp.Process] = []
        for rank in range(num_processes):
            p = ctx.Process(
                target=self._worker_init,
                args=(
                    queue,
                    rank,
                    num_processes,
                    self.model_name,
                    tmp_directory,
                    func,
                    *args,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)

        results: list[Any] = [None] * len(processes)

        try:
            for _ in range(len(processes)):
                result = queue.get(timeout=120)
                # result = queue.get()

                if "error" in result:
                    # If any process get an error,
                    # immediately abort the test.
                    raise RuntimeError(result["error"])
                else:
                    results[result["rank"]] = result["success"]

            # Here, all processes are successfully finished.
            for process in processes:
                process.join()
        except Exception as e:
            for process in processes:
                process.terminate()
                process.join()
            raise e

        return results
