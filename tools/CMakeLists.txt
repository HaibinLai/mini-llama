# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(main)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    # if (LLAMA_BUILD_SERVER)
    #     add_subdirectory(server)
    # endif()
    add_subdirectory(run)
    add_subdirectory(tokenize)
    # add_subdirectory(tts)
    # add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (NOT GGML_BACKEND_DL)
        # these examples use the backends directly and cannot be built with dynamic loading
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
endif()


# | 子目录名                | 功能简介                                                                   |
# | ------------------- | ---------------------------------------------------------------------- |
# | `batched-bench`     | 用于测试“批量推理”（batched inference）的性能。例如多个 prompt 一起推理时的吞吐对比。               |
# | `gguf-split`        | 将 GGUF（新格式的模型文件）拆分成多个小的 GGUF 文件，用于模型裁剪、分布式加载等实验。                       |
# | `imatrix`           | 一个示例程序，展示如何使用 llama.cpp 生成 embedding，类似文本转向量的功能。                       |
# | `llama-bench`       | 性能 benchmark 工具，对模型推理做精确计时（比如 token latency、吞吐量等）。                     |
# | `main`              | 主 CLI 推理入口，例如 `./main -m model.gguf -p "Hello"`，这是 llama.cpp 的核心 demo。 |
# | `perplexity`        | 计算模型在某段文本上的 perplexity（困惑度）指标，衡量语言模型的好坏。                               |
# | `quantize`          | 用于对模型做量化处理的工具（fp32 → int8/int4 等），减少模型体积，提高推理速度。                       |
# | `run`               | 另一个 CLI 工具，类似 `main`，但通常用于 pipeline 式评测（可自动循环输入文件等）。                   |
# | `tokenize`          | 提供 tokenizer 的 demo，用于测试 prompt 分词效果或生成 token 序列。                      |
# | `tts`               | **Text-to-Speech 示例**，使用语音模型将文本转换为音频（通常不是 llama 本身，而是集成的 demo）。        |
# | `mtmd`              | 和多模态模型或 metadata 处理有关的实验目录，目前主要用于试验性功能。                                |
# | `rpc`               | 远程调用服务（Remote Procedure Call），启动一个推理服务，可通过网络远程调用模型。前提是 `GGML_RPC=ON`。  |
# | `cvector-generator` | 用于生成向量的工具（可能用于文档嵌入、向量搜索等）。不能在动态链接后端构建时使用。                              |
# | `export-lora`       | 从 LLaMA 模型中导出 LoRA 权重的工具（Low Rank Adapter，微调技术）。也需要静态后端。               |
