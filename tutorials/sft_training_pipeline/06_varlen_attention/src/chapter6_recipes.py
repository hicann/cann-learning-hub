"""Fixed, tutorial-local recipes for the Chapter 6 packing comparison."""

from dataclasses import replace

from torchtitan_npu.hf_datasets.wordle import (
    WordleChatDataLoader,
    process_wordle_sample,
)
from torchtitan_npu.models.qwen3.config_registry import (
    sft_qwen3_1_7b_wordle,
    sft_qwen3_1_7b_wordle_block_causal_sdpa,
)


def _comparison_config(base, *, checkpoint_folder: str):
    train_loader = replace(
        base.dataloader,
        dataset_path="./assets/data/wordle",
        load_dataset_kwargs={"split": "train[:900]"},
    )
    validation_loader = WordleChatDataLoader.Config(
        dataset_path="./assets/data/wordle",
        load_dataset_kwargs={"split": "train[900:]"},
        sample_processor=process_wordle_sample,
        infinite=False,
        greedy_packing=False,
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
            initial_load_path=None,
        ),
        validator=replace(
            base.validator,
            enable=True,
            freq=1,
            steps=-1,
            dataloader=validation_loader,
        ),
        profiler=replace(base.profiler, enable_profiling=False),
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
