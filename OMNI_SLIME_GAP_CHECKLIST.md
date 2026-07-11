# Relax vs slime-main

第一层：types.py / arguments.py
MultimodalTypes（IMAGE / VIDEO / AUDIO）
两边定义一致，占位符都是 <image>、<video>、<audio>，这一层 不必再从 Relax「搬类型枚举」。

CLI / 训练参数
差异主要在 arguments.py：

Relax：在 --multimodal-keys 之后有一整块 Omni/VL 处理参数，其中包括 --use-audio-in-video、--audio-sample-rate、--mm-processor-pool-size，以及 --image-*、--video-*、--frame-factor 等（约 Relax/relax/utils/arguments.py 774–845 行一带）。
slime-main：仅有 --multimodal-keys，接着就是 --metadata-key，没有上述音频、视频内音轨、processor 进程池等参数。
因此：Omni 必备差别是 slime-main 缺 Relax 里那套多模态处理相关的 argparse（不仅是三个 flag）。

第二层：加载与 process_vision_info
Relax：relax/utils/multimodal/（process.py、audio_utils.py、video_utils.py、image_utils.py、config.py）实现 process_multimodal_info：从对话里抽取 image / video / audio，支持 use_audio_in_video（视频带音轨时并入 audio 列表）。relax/utils/data/processing_utils.py 里 process_vision_info(..., use_audio_in_video, config) 返回 {"images", "videos", "audio"}（键名是 audio）。
slime-main：slime/utils/processing_utils.py 的 process_vision_info 基本是 qwen_vl_utils.process_vision_info → 只有 images + videos，没有 audio；也没有 Relax 里 encode_video_tensor_for_rollout_engine / encode_audio_for_rollout_engine 及异步封装（slime-main 只有 encode_image_for_rollout_engine）。
结论：Omni 路径上，Relax 是「完整三模态 + 可选视频中音频」；slime-main 仍是 偏 Qwen-VL 的 vision 管线，缺音频与完整 rollout 编码工具。

第三层：data.py（占位符 → message content）
_build_messages 里占位符展开逻辑（<audio> / <video> → {"type": "audio", "audio": ...} 等）
两边实现 是同一路思路，字符串里的占位符会拆成 content list。

差别在 Dataset 是否把 Omni 选项往下传：

Relax Dataset：__init__ 接收 use_audio_in_video、multimodal_config，并交给基类（用于后续处理链）。
slime-main Dataset：__init__ 没有 use_audio_in_video / multimodal_config（约 slime-main/slime/utils/data.py 195–211 行），因此即使 JSON 里占位符对了，训练侧仍缺与 Relax 对齐的配置入口。
另外，slime-main 的 filter_long_prompt 对「纯文本 vs 有多模态输入」做了分支；这和 Omni 是否完备无关，只是实现差异。

第四层：sglang_rollout.py（请求 payload / token）
Relax：对 multimodal 做 _encode_multimodal_inputs：并行编码 image / video tensor / audio，填入 image_data、video_data、audio_data；生成路径里还带 use_audio_in_video；并对 audio_token_id、video_token_id 出现在输出里做了替换/告警逻辑。
slime-main：generate 里仅在存在 images 时设置 image_data + text（约 182–189 行），未见 video_data / audio_data，也 未见 输出侧对 audio/video special token 的处理。
这是 Omni rollout 与 slime-main 的核心缺口之一。

第五层：Megatron ↔ HF（权重 / Bridge）
Relax：relax/backends/megatron/weight_conversion/qwen3_omni_moe.py，且在 weight_conversion/__init__.py 里对 qwen3omni 名字做了 convert_qwen3omni_to_hf 分支。
slime-main：slime/backends/megatron_utils/megatron_to_hf/__init__.py 的 _convert_to_hf_core 没有 qwen3 omni 分支；目录里也 没有 qwen3_omni_moe.py（在工作区 slime-main 内 grep/glob 为 0）。
你提到的 slime/slime_plugins/megatron_bridge/qwen3_omni_moe.py 存在于工作区的 slime\ 目录，不在 slime-main 树里——那是 Megatron-Bridge 建模侧插件；而 Relax 的 weight_conversion 对应 Megatron → HF 导出/对齐。两条线互补：一个管训练侧 model provider，一个管权重映射。

第六层：训练脚本
未逐行 diff 两个仓库的全部 shell；与你贴的 GPT 建议一致：Relax 有面向 Omni 的多卡脚本；my_omni 示例里的 --rotary-base、--megatron-to-hf-mode bridge、--multimodal-keys、去掉 --untie-embeddings-and-output-weights 等属于「跑通 Qwen3-Omni」的经验参数，需在 slime 侧脚本里 对照合并，而不是假设 upstream slime-main 已内置。第一层：types.py / arguments.py
MultimodalTypes（IMAGE / VIDEO / AUDIO）
两边定义一致，占位符都是 <image>、<video>、<audio>，这一层 不必再从 Relax「搬类型枚举」。

CLI / 训练参数
差异主要在 arguments.py：

Relax：在 --multimodal-keys 之后有一整块 Omni/VL 处理参数，其中包括 --use-audio-in-video、--audio-sample-rate、--mm-processor-pool-size，以及 --image-*、--video-*、--frame-factor 等（约 Relax/relax/utils/arguments.py 774–845 行一带）。
slime-main：仅有 --multimodal-keys，接着就是 --metadata-key，没有上述音频、视频内音轨、processor 进程池等参数。
因此：Omni 必备差别是 slime-main 缺 Relax 里那套多模态处理相关的 argparse（不仅是三个 flag）。

第二层：加载与 process_vision_info
Relax：relax/utils/multimodal/（process.py、audio_utils.py、video_utils.py、image_utils.py、config.py）实现 process_multimodal_info：从对话里抽取 image / video / audio，支持 use_audio_in_video（视频带音轨时并入 audio 列表）。relax/utils/data/processing_utils.py 里 process_vision_info(..., use_audio_in_video, config) 返回 {"images", "videos", "audio"}（键名是 audio）。
slime-main：slime/utils/processing_utils.py 的 process_vision_info 基本是 qwen_vl_utils.process_vision_info → 只有 images + videos，没有 audio；也没有 Relax 里 encode_video_tensor_for_rollout_engine / encode_audio_for_rollout_engine 及异步封装（slime-main 只有 encode_image_for_rollout_engine）。
结论：Omni 路径上，Relax 是「完整三模态 + 可选视频中音频」；slime-main 仍是 偏 Qwen-VL 的 vision 管线，缺音频与完整 rollout 编码工具。

第三层：data.py（占位符 → message content）
_build_messages 里占位符展开逻辑（<audio> / <video> → {"type": "audio", "audio": ...} 等）
两边实现 是同一路思路，字符串里的占位符会拆成 content list。

差别在 Dataset 是否把 Omni 选项往下传：

Relax Dataset：__init__ 接收 use_audio_in_video、multimodal_config，并交给基类（用于后续处理链）。
slime-main Dataset：__init__ 没有 use_audio_in_video / multimodal_config（约 slime-main/slime/utils/data.py 195–211 行），因此即使 JSON 里占位符对了，训练侧仍缺与 Relax 对齐的配置入口。
另外，slime-main 的 filter_long_prompt 对「纯文本 vs 有多模态输入」做了分支；这和 Omni 是否完备无关，只是实现差异。

第四层：sglang_rollout.py（请求 payload / token）
Relax：对 multimodal 做 _encode_multimodal_inputs：并行编码 image / video tensor / audio，填入 image_data、video_data、audio_data；生成路径里还带 use_audio_in_video；并对 audio_token_id、video_token_id 出现在输出里做了替换/告警逻辑。
slime-main：generate 里仅在存在 images 时设置 image_data + text（约 182–189 行），未见 video_data / audio_data，也 未见 输出侧对 audio/video special token 的处理。
这是 Omni rollout 与 slime-main 的核心缺口之一。

第五层：Megatron ↔ HF（权重 / Bridge）
Relax：relax/backends/megatron/weight_conversion/qwen3_omni_moe.py，且在 weight_conversion/__init__.py 里对 qwen3omni 名字做了 convert_qwen3omni_to_hf 分支。
slime-main：slime/backends/megatron_utils/megatron_to_hf/__init__.py 的 _convert_to_hf_core 没有 qwen3 omni 分支；目录里也 没有 qwen3_omni_moe.py（在工作区 slime-main 内 grep/glob 为 0）。
你提到的 slime/slime_plugins/megatron_bridge/qwen3_omni_moe.py 存在于工作区的 slime\ 目录，不在 slime-main 树里——那是 Megatron-Bridge 建模侧插件；而 Relax 的 weight_conversion 对应 Megatron → HF 导出/对齐。两条线互补：一个管训练侧 model provider，一个管权重映射。

第六层：训练脚本
未逐行 diff 两个仓库的全部 shell；与你贴的 GPT 建议一致：Relax 有面向 Omni 的多卡脚本；my_omni 示例里的 --rotary-base、--megatron-to-hf-mode bridge、--multimodal-keys、去掉 --untie-embeddings-and-output-weights 等属于「跑通 Qwen3-Omni」的经验参数，需在 slime 侧脚本里 对照合并，而不是假设 upstream slime-main 已内置。