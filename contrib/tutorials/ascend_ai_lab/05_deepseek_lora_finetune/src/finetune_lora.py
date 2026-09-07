"""DeepSeek-7B LoRA 微调训练参考脚本。"""
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

MODEL_PATH = "./model/deepseek-ai/deepseek-llm-7b-chat"
DATA_PATH = "./data/medical_multi_data.json"


def load_model_and_tokenizer():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{MODEL_PATH}/tokenizer.json",
        pad_token="</s>", eos_token="</s>", bos_token="<s>")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="npu:0",
        attn_implementation="sdpa")
    return model, tokenizer


def build_lora_model(model):
    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    return model


def train(model, tokenizer, train_data, eval_data):
    args = SFTConfig(
        output_dir="./sft_output",
        max_steps=500,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        bf16=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=50,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )
    trainer.train()
