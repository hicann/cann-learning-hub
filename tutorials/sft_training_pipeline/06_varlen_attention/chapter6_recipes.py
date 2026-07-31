"""Fixed, tutorial-local recipes for the Chapter 6 packing comparison."""

from dataclasses import dataclass, replace

from torchtitan_npu.config.configs import ChatDataLoaderConfig
from torchtitan_npu.models.qwen3.config_registry import (
    sft_qwen3_1_7b_wordle,
    sft_qwen3_1_7b_wordle_block_causal_sdpa,
)


@dataclass(kw_only=True, slots=True)
class FixedNonGreedyChatDataLoaderConfig(ChatDataLoaderConfig):
    """Build a chat loader whose evaluation containers contain one sample each."""

    def build(self, **kwargs):
        # dataclass(slots=True) replaces the class object, so zero-argument
        # super() is unsafe in methods defined on the pre-decoration class.
        loader = ChatDataLoaderConfig.build(self, **kwargs)
        loader.dataset._greedy_packing = False
        return loader


def _comparison_config(base, *, checkpoint_folder: str):
    train_loader = replace(
        base.dataloader,
        dataset_path="./assets/data/wordle",
        load_dataset_kwargs={"split": "train[:900]"},
        dataset_split=None,
    )
    validation_loader = FixedNonGreedyChatDataLoaderConfig(
        dataset_path="./assets/data/wordle",
        load_dataset_kwargs={"split": "train[900:]"},
        chat_processor=(
            "torchtitan_npu.hf_datasets.chat_processors.process_wordle_sample"
        ),
    )

    return replace(
        base,
        debug=replace(base.debug, seed=42, print_config=False),
        training=replace(
            base.training,
            steps=2,
            local_batch_size=2,
            global_batch_size=64,
            seq_len=4096,
        ),
        dataloader=train_loader,
        checkpoint=replace(
            base.checkpoint,
            enable=True,
            load_only=True,
            folder=checkpoint_folder,
            initial_load_in_hf=True,
            initial_load_path="./assets/hf/Qwen3-1.7B",
        ),
        validator=replace(
            base.validator,
            enable=True,
            freq=1,
            steps=-1,
            dataloader=validation_loader,
        ),
        profiling=replace(base.profiling, enable_profiling=False),
    )


def wordle_non_greedy_gbs64():
    """Causal SDPA and one Wordle sample per padded container."""

    return _comparison_config(
        sft_qwen3_1_7b_wordle(),
        checkpoint_folder="checkpoints/ch6_non_greedy_gbs64",
    )


def wordle_greedy_block_gbs64():
    """Greedy packing with the dense block-causal SDPA correctness route."""

    return _comparison_config(
        sft_qwen3_1_7b_wordle_block_causal_sdpa(),
        checkpoint_folder="checkpoints/ch6_greedy_block_gbs64",
    )
