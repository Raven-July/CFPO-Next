# 202步训练与后续诊断代码审阅

本目录对应截至2026年11月16日已完成的四轮实验。当前仓库中的训练工具包含后续本地修订；历史运行代码在各轮`source_overlay`中单独冻结，不能将当前工作树当成历史进程加载版本。

## 代码入口

| 实验 | 代码与配置 | 运行状态 |
|---|---|---|
| 旧奖励202步GRPO | [启动源码增量](legacy202/source_overlay/)、[原始配置](legacy202/resolved-original.yaml)、[恢复配置](legacy202/resolved-resume.yaml)、[恢复与检查工具](legacy202/cluster_tools/) | 222487在117步后中断；223620从100步恢复至202步，完成 |
| 新奖励原始模型30步 | [运行源码增量](newreward30/source_overlay/)、[完整配置](newreward30/resolved-config.yaml)、[配置变化](newreward30/config-differences.json)、[准备与启动工具](newreward30/cluster_tools/) | 224397完成；原始Instruct重新开始，无旧checkpoint恢复 |
| 32题三提示推理 | [fix2成功版本](formatprobe32/) | 224596完成，480条；无模型更新 |
| 4题HF与vLLM及提示对照 | [诊断代码](diagnostic12/) | 224673完成，12条；无模型更新 |

`source_overlay`是基于commit `0c6bd1855ccf31f711924e66a70a1d25387425b0`的源码增量，不是完整仓库。旧202步取自实际启动快照；新30步用原始payload和日志补丁重建，25份文件逐一通过实际运行receipt的SHA256。来源与hash见[provenance.json](provenance.json)。当前主目录默认新奖励，旧202步的旧奖励位于其冻结源码中；`legacy_base.py`也单独保留。

## 结果摘要

| 实验 | 主要观测 | 已试方案状态 |
|---|---|---|
| 旧202步 | 五集加权验证正确率37.39%至44.59%；验证格式分持续为0，仍有较多长度触顶 | 完成训练；新规则离线回放未解决旧输出缺标签 |
| 新30步 | 验证37.54%至43.22%；训练57600条仅15条格式正例，验证全0；末轮验证触顶19.28% | 奖励部署与评分通过；格式生成未解决 |
| 32题对照 | 原提示、预填开标签、占位符示例合格式数分别0、1、0，各160条 | 预填与占位符示例未稳定解决；出现误读及重复 |
| 4题对照 | HF也复现部分误读、重复；简短提示4/4语义正确且正常结束，但无boxed，固定奖励accuracy仍0/4 | 提示影响得到小样本证据；格式仍未解决 |

上述训练验证为五集temperature=0、n=1，推理诊断使用选定样本。[机器可读汇总](results-summary.json)保留各轮分母。正式十集temperature=1、Avg@8尚未执行。新旧30步存在初始输出和运行条件差异，不以单次分数差作严格因果结论。

## 重点审查项

1. [当前奖励](../../reward_function/base.py)接受闭think标签后boxed，实际新30步已部署。tokenizer保留标签，但模型很少生成它；11505/11520个五回答组格式全0。增加权重不能为全0组创造正例。
2. [输入system](../../format_prompt/reference_base.jinja)与completion结构共同检查。预填和占位符示例的真实实现见`formatprobe32/core.py`，简短system见`diagnostic12/protocol.py`及`run_probe.py`。
3. 图像token打包修复见`formatprobe32/core.py`及`diagnostic12/image_tokens.py`。成功32题运行96次、12条运行8次引擎展开输入检查通过；请求侧一致不等于vLLM内部最终像素tensor均已观测。
4. [参考差异](reference-differences.json)比较公开Qwen2.5 GRPO参考与实际配置，不是Qwen3成功运行快照。微批归一化数学差异已确认，训练影响仍待对照；它不参与原始模型零更新推理。

后续候选为简短提示加独立完整格式演示、单变量2048/4096对照及微批归一化对照，均未执行。没有更换Thinking模型、改写历史评分或启动新训练。

## 使用条件

本次发布仅包含代码、配置、hash与小型统计。原图、题目与答案、选样计划、完整生成、集群私有manifest/receipt和权重不在Git中。各诊断prepare会严格检查这些输入，不能在缺失时直接提交；这里的代码目录不是可独立解包运行的完整历史传输包。new30的历史`prepare.py`从原私有启动快照恢复源码，原完整包的非代码校验文件也保留在本地；本目录不改变该准备流程或绕过校验。

历史配置保留集群路径与环境变量用于定位实际条件，其他环境需显式适配。不要用旧冻结快照覆盖当前工作树或在共享环境热更新。

AI辅助整理与验证；未创建新PR，未自动发送消息。
