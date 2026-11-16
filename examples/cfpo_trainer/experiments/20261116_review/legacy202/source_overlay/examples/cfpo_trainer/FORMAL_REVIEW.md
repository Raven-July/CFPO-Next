# 正式GRPO候选配置：训练前审阅

这套入口准备202步、双卡、完整训练manifest，训练前及每5步验证，并记录training accuracy/reward与评测生成。旧pilot独立保留，不续训其checkpoint。

## 向学长提交的内容

- `run_formal.sh`、`formal_job.sbatch`、`formal_metrics.py`，及ray_trainer中仅由`trainer.cfpo_component_metrics=True`开启的指标汇总与验证输出ID/来源/长度记录。
- 远端预检查生成的`resolved-config.yaml`、`length-audit.json`、`data-audit.json`及三个manifest。
- 本文的差异和原生看板预检查结果。AI辅助编写；Bao需审阅实现和配置。

## 已对齐的主要参数

Qwen3-VL-2B-Instruct原始模型；prompt/response上限4096/2048；rollout batch384；n=5；actor mini batch128、micro batch4；lr=1e-6、warmup ratio0.05；top_p0.99；训练数据seed1；2epochs、2GPUs；图像面积200704–1003520；训练前与每5步验证、30条网页生成展示、完整生成落盘；每50步保存，保留5份actor checkpoint。

## 需明确接受的迁移差异

- 2026-11-09已迁移原版`AnyPrecisionAdamW`类，类正文保持不变，仅适配导入；来源哈希见`reference-optimizer-provenance.json`。CPU测试核对类AST、连续更新和优化器状态一致性；还须在容器PyTorch版本及双卡短检查中验证集成。
- 已显式对齐actor/ref模型参数BF16、PPO裁剪上下限0.2/0.3、dual clip 3.0和token-mean损失归约，避免沿用新框架不同的默认值。
- 正式实验单独使用原版`reference_base.jinja`，包含`in \\boxed{}`前的空格；旧pilot提示词与历史哈希保留。实际提示词及奖励适配文件哈希在正式配置检查中核对。
- 当前使用eager、chunked prefill、rollout显存比例0.35、最大并发64及模型长度6144；与旧config的性能设置不同，尚未测量双卡吞吐与峰值显存。
- checkpoint已对齐最多保留5份；运行前核对个人磁盘额度。Slurm时限由实际队列上限与学长授权决定，不从旧20步耗时线性外推。
- 新框架保留现有KL loss配置0.01；新旧框架训练轨迹不保证逐步相同。
- 验证使用原设计五个val来源的现有实际路径，修正旧manifest中`COCO_val2014_imagess`拼写；最终测试单独使用十集test manifest。预处理检查不证明消除了所有训练/验证重合，需要审阅来源与重复审计。
- 202步以过滤后dataloader长度核对；不一致时脚本停止，不能只强制改总步数。短检查限定1step、16条验证样本，不能作为完整评测。
- 最终评测保持原版评测prompt和评分逻辑，包装层检查十集齐全、图像可读、未跳题、两卡导出、完整输出次数及结束状态；n=1或8由学长确认。n=8采用逐题8次正确比例的平均（Avg@8），不是Pass@8。

## 已做与待做的验证

本地Python语法、Bash语法、CPU指标/数据/汇总测试已检查。尚未进行远端配置解析、真实数据过滤、SwanLab local兼容性验证或双卡GPU运行。依次执行操作指南的预检查与短运行，通过后才进入长训练。

Job221948已完成依赖阶段，随后读取`MathVerse_images/images_version_6/image_745.png`遇到PermissionError并停止，未训练。只读定位已确认该图属于MathVerse验证集第505条（source_index504，ID MathVerse_3725），权限为600、所有者u2021213615。训练及周期验证所需图片必须全部可读。独立最终test可单独审计，十集清单保留；不可删题或冒称完整评测已通过。
