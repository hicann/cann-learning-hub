#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen1.5-0.5B-Chat 模型在昇腾310B(香橙派)上的推理脚本
=====================================================
功能：
  1. 使用ACL(Ascend Computing Language)加载ATC转换后的.om模型
  2. 使用Transformers分词器对输入文本进行编解码
  3. 自回归生成实现对话推理
  4. 通过Gradio提供Web聊天交互界面 (http://127.0.0.1:7860)

运行环境：
  - 硬件：昇腾香橙派 (Ascend 310B)
  - 系统：Ubuntu 22.04
  - 软件：CANN 9.0、Python 3.8+、acl、transformers、gradio、numpy

使用方法：
  python3 qwen1.5-0.5b-chat.py
  然后在浏览器中打开 http://127.0.0.1:7860
"""

import os
import sys
import time
import numpy as np

import acl

from transformers import AutoTokenizer

import gradio as gr


# ==================== 配置参数 ====================

OM_MODEL_PATH = "./om_export/qwen_merged.om"

TOKENIZER_PATH = "./Qwen1.5-0.5B-Chat"

SEQ_LEN = 32

MAX_NEW_TOKENS = 28

DEVICE_ID = 0

PORT = 7860

INPUT_DTYPE = np.int64

OUTPUT_DTYPE = np.float32


# ==================== ACL模型推理类 ====================

class ACLInference:
    """
    基于pyACL的模型推理封装类
    负责ACL环境初始化、模型加载、数据集创建、推理执行和资源释放
    """

    def __init__(self, model_path, device_id=0):
        self.device_id = device_id
        self.model_path = model_path
        self.model_id = None
        self.model_desc = None
        self.context = None
        self._init_acl()
        self._load_model()

    def _init_acl(self):
        """初始化ACL运行环境"""
        ret = acl.init()
        if ret != 0:
            raise RuntimeError("acl.init 失败, ret={}".format(ret))

        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            raise RuntimeError("acl.rt.set_device 失败, ret={}".format(ret))

        self.context, ret = acl.rt.create_context(self.device_id)
        if ret != 0:
            raise RuntimeError("acl.rt.create_context 失败, ret={}".format(ret))

        print("[ACL] 环境初始化成功, device_id={}".format(self.device_id))

    def _load_model(self):
        """加载.om离线模型并获取模型描述信息"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("模型文件不存在: {}".format(self.model_path))

        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError("acl.mdl.load_from_file 失败, ret={}".format(ret))

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            raise RuntimeError("acl.mdl.get_desc 失败, ret={}".format(ret))

        self.input_num = acl.mdl.get_num_inputs(self.model_desc)
        self.output_num = acl.mdl.get_num_outputs(self.model_desc)

        self.input_sizes = []
        for i in range(self.input_num):
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            self.input_sizes.append(size)

        self.output_sizes = []
        for i in range(self.output_num):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            self.output_sizes.append(size)

        print("[ACL] 模型加载成功: {}".format(self.model_path))
        print("[ACL] 输入数量={}, 输出数量={}".format(self.input_num, self.output_num))
        for i in range(self.input_num):
            print("[ACL]   input[{}] size={} bytes".format(i, self.input_sizes[i]))
        for i in range(self.output_num):
            print("[ACL]   output[{}] size={} bytes".format(i, self.output_sizes[i]))

    def _create_dataset(self, buffers_info):
        """
        创建ACL数据集
        buffers_info: [(data_bytes, size), ...] 或 [(None, size), ...] (仅分配不拷贝)
        返回: (dataset, device_ptrs)
        """
        dataset = acl.mdl.create_dataset()
        device_ptrs = []
        for data_bytes, size in buffers_info:
            device_ptr, ret = acl.rt.malloc(size, 0)
            if ret != 0:
                raise RuntimeError("acl.rt.malloc 失败, ret={}".format(ret))
            device_ptrs.append(device_ptr)

            if data_bytes is not None:
                host_ptr = acl.util.numpy_to_ptr(data_bytes)
                ret = acl.rt.memcpy(device_ptr, size, host_ptr, size, 1)
                if ret != 0:
                    raise RuntimeError("acl.rt.memcpy H2D 失败, ret={}".format(ret))

            data_buf = acl.create_data_buffer(device_ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(dataset, data_buf)
            if ret != 0:
                raise RuntimeError("acl.mdl.add_dataset_buffer 失败, ret={}".format(ret))

        return dataset, device_ptrs

    def _destroy_dataset(self, dataset):
        """销毁ACL数据集并释放设备内存"""
        num = acl.mdl.get_dataset_num_buffers(dataset)
        for i in range(num):
            buf = acl.mdl.get_dataset_buffer(dataset, i)
            device_ptr = acl.get_data_buffer_addr(buf)
            if device_ptr:
                acl.rt.free(device_ptr)
            acl.destroy_data_buffer(buf)
        acl.mdl.destroy_dataset(dataset)

    def infer(self, input_arrays):
        """
        执行模型推理
        input_arrays: [np.ndarray, ...] 输入numpy数组列表
        返回: [np.ndarray, ...] 输出numpy数组列表(原始字节)
        """
        acl.rt.set_context(self.context)

        input_info = []
        for arr in input_arrays:
            input_info.append((arr, arr.nbytes))

        input_dataset, _ = self._create_dataset(input_info)

        output_info = [(None, size) for size in self.output_sizes]
        output_dataset, output_ptrs = self._create_dataset(output_info)

        ret = acl.mdl.execute(self.model_id, input_dataset, output_dataset)
        if ret != 0:
            raise RuntimeError("acl.mdl.execute 失败, ret={}".format(ret))

        outputs = []
        for i in range(self.output_num):
            buf = acl.mdl.get_dataset_buffer(output_dataset, i)
            data_ptr = acl.get_data_buffer_addr(buf)
            data_size = acl.get_data_buffer_size_v2(buf)

            output_bytes = np.zeros(data_size, dtype=np.uint8)
            host_ptr = acl.util.numpy_to_ptr(output_bytes)
            ret = acl.rt.memcpy(host_ptr, data_size, data_ptr, data_size, 2)
            if ret != 0:
                raise RuntimeError("acl.rt.memcpy D2H 失败, ret={}".format(ret))
            outputs.append(output_bytes)

        self._destroy_dataset(input_dataset)
        self._destroy_dataset(output_dataset)

        return outputs

    def get_output_array(self, output_bytes, dtype=np.float32):
        """将原始输出字节转换为numpy数组"""
        return np.frombuffer(output_bytes.tobytes(), dtype=dtype)

    def __del__(self):
        """释放ACL资源"""
        try:
            if self.model_id is not None:
                acl.mdl.unload(self.model_id)
            if self.model_desc is not None:
                acl.mdl.destroy_desc(self.model_desc)
            if self.context is not None:
                acl.rt.destroy_context(self.context)
            acl.rt.reset_device(self.device_id)
            acl.finalize()
            print("[ACL] 资源已释放")
        except Exception:
            pass


# ==================== Qwen对话生成类 ====================

class QwenGenerator:
    """
    Qwen1.5-0.5B-Chat 对话生成器
    负责构建对话prompt、分词、自回归生成、解码
    """

    def __init__(self, model, tokenizer, seq_len, max_new_tokens):
        self.model = model
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_new_tokens = max_new_tokens

        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        self.needs_position_ids = (model.input_num >= 3)

        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

        print("[Qwen] eos_token_id={}, pad_token_id={}".format(
            self.eos_token_id, self.pad_token_id))
        print("[Qwen] 模型输入数={}, 需要position_ids={}".format(
            model.input_num, self.needs_position_ids))

    def _build_chat_prompt(self, message, history):
        """
        构建ChatML格式的对话prompt
        history: [(user_msg, assistant_msg), ...] 历史对话
        message: 当前用户输入
        """
        messages = []
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def _prepare_inputs(self, token_ids):
        """
        将token列表转换为模型输入张量(左填充至seq_len)
        返回: [input_ids, attention_mask] 或 [input_ids, attention_mask, position_ids]
        """
        tokens = token_ids[-self.seq_len:]
        actual_len = len(tokens)
        pad_len = self.seq_len - actual_len

        padded_ids = np.array(
            [self.pad_token_id] * pad_len + tokens,
            dtype=INPUT_DTYPE
        ).reshape(1, self.seq_len)

        attention_mask = np.array(
            [0] * pad_len + [1] * actual_len,
            dtype=INPUT_DTYPE
        ).reshape(1, self.seq_len)

        inputs = [padded_ids, attention_mask]

        if self.needs_position_ids:
            position_ids = np.array(
                [0] * pad_len + list(range(actual_len)),
                dtype=INPUT_DTYPE
            ).reshape(1, self.seq_len)
            inputs.append(position_ids)

        return inputs

    def generate(self, message, history):
        """
        自回归生成对话回复(生成器函数，支持流式输出)
        message: 用户输入文本
        history: 历史对话列表 [(user, assistant), ...]
        yield: 逐步生成的回复文本
        """
        prompt = self._build_chat_prompt(message, history)
        prompt_ids = self.tokenizer.encode(prompt)

        print("[Qwen] prompt token数={}, 将生成最多{}个token".format(
            len(prompt_ids), self.max_new_tokens))

        all_tokens = list(prompt_ids)
        generated_tokens = []
        response_text = ""

        for step in range(self.max_new_tokens):
            inputs = self._prepare_inputs(all_tokens)

            outputs = self.model.infer(inputs)

            logits = self.model.get_output_array(outputs[0], dtype=OUTPUT_DTYPE)

            output_element_count = logits.size
            vocab_size = output_element_count // self.seq_len
            logits = logits.reshape(self.seq_len, vocab_size)

            next_token = int(np.argmax(logits[-1]))

            if next_token == self.eos_token_id:
                print("[Qwen] 生成结束(EOS), 共生成{}个token".format(len(generated_tokens)))
                break

            generated_tokens.append(next_token)
            all_tokens.append(next_token)

            new_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            if new_text != response_text:
                response_text = new_text
                yield response_text

        if not response_text:
            response_text = "(模型未生成有效回复，请尝试缩短输入或重试)"
            yield response_text

        print("[Qwen] 最终回复: {}".format(response_text[:100]))


# ==================== Gradio Web界面 ====================

def create_interface(generator):
    """
    创建Gradio聊天界面
    包含: 聊天框、消息输入框、Examples示例、Submit/Retry/Undo/Clear按钮
    """
    examples = [
        "你好，请介绍一下你自己",
        "什么是人工智能？",
        "请写一首关于春天的短诗",
        "1+1等于几？",
        "请用一句话解释什么是深度学习",
    ]

    custom_css = """
    .gradio-container {
        max-width: 800px !important;
    }
    #component-0 {
        border-radius: 12px;
    }
    """

    demo = gr.ChatInterface(
        fn=generator.generate,
        title="Qwen1.5-0.5B-Chat 昇腾310B推理",
        description=(
            "基于昇腾香橙派 Ascend 310B 嵌入式平台推理 | "
            "模型: Qwen1.5-0.5B-Chat | "
            "框架: CANN ACL + OM离线模型"
        ),
        examples=examples,
        theme=gr.themes.Soft(),
        css=custom_css,
        retry_btn="retry",
        undo_btn="undo",
        clear_btn="clear",
        textbox=gr.Textbox(
            placeholder="Type a message...",
            scale=7,
            lines=2,
        ),
        submit_btn="Submit",
    )

    return demo


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("Qwen1.5-0.5B-Chat 昇腾310B(香橙派) 推理服务")
    print("=" * 60)

    print("\n[1/3] 正在初始化ACL并加载.om模型...")
    print("      模型路径: {}".format(OM_MODEL_PATH))
    model = ACLInference(OM_MODEL_PATH, device_id=DEVICE_ID)

    print("\n[2/3] 正在加载Qwen分词器...")
    print("      分词器路径: {}".format(TOKENIZER_PATH))
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True
    )

    generator = QwenGenerator(
        model=model,
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        max_new_tokens=MAX_NEW_TOKENS
    )

    print("\n[3/3] 正在启动Gradio Web服务...")
    demo = create_interface(generator)

    print("\n" + "=" * 60)
    print("服务已启动!")
    print("请在浏览器中打开: http://127.0.0.1:{}".format(PORT))
    print("按 Ctrl+C 停止服务")
    print("=" * 60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
