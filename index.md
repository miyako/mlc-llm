---
layout: default
---

![version](https://img.shields.io/badge/version-20%2B-E23089)
![platform](https://img.shields.io/static/v1?label=platform&message=mac-intel%20|%20mac-arm%20|%20win-64&color=blue)
[![license](https://img.shields.io/github/license/miyako/mlc_llm)](LICENSE)
![downloads](https://img.shields.io/github/downloads/miyako/mlc_llm/total)

# Use MLC_LLM from 4D

#### Abstract

[**MLC_LLM**](https://github.com/mlc-ai/mlc-llm) is an inference engine that can run compiled LLMs. Other engines are effectively interpreters, whether they work with quantised GGUF files or native safetensors. MLC is unique in that the models are precompiled for a specific platform (Metal, Vulkan, ROCm, or CUDA).

This package includes the runtime only. You need to [install the TVM compiler](https://llm.mlc.ai/docs/install/tvm.html) and [install the MLC LLM python package](https://llm.mlc.ai/docs/install/mlc_llm.html) in order to [compile models](https://llm.mlc.ai/docs/compilation/compile_models.html).

#### AI Kit compatibility

The API is compatibile with [Open AI](https://platform.openai.com/docs/api-reference/embeddings). 

|Class|API|Availability|
|-|-|:-:|
|Models|`/v1/models`|✅|
|Chat|`/v1/chat/completions`|✅|
|Images|`/v1/images/generations`||
|Moderations|`/v1/moderations`||
|Embeddings|`/v1/embeddings`|✅|
|Files|`/v1/files`||
