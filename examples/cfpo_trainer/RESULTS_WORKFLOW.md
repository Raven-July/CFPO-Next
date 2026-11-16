# 实验执行与结果提交工作流

生效：2026-11-04。适用于 Bao 的 CFPO-Next GRPO/DAPO 实验。依据：Bao 明确要求采用学长本次反馈作为以后提交标准。此文件为该流程正文；README 为入口，WEEKLY_REVIEW 保留历史审查，不复制一套平行规则。

2026-11-11补充：交付脚本、提交实验及收尾前按项目[脚本审查清单](../../../审查/脚本审查清单.md)保留证据记录；[本次会议记录](../../../审查/会议记录/2026-11-11-GRPO训练复盘.md)集中保存逐字稿时间戳、学长奖励参考片段及图片参数。特别区分训练间隙验证的贪心提速设置与最终temperature=1.0、n=8的正式推理评测；不得把前者覆盖带入后者。会议参考分数不自动成为已验证达标线，当前作业是否加载新奖励须独立核验。

## 验收标准

每次正式结果提交必须具备以下三项；缺项只能报阶段进度，不能称完整交付：

1. **SwanLab 原生本地网页看板**：从实验日志目录启动离线看板，核对正确 run/step 范围；可查看 training accuracy 与 reward，并分别标明 format、总 reward 和验证指标。截图与静态 HTML 仅为补充。
2. **训练中各评测轮的生成内容**：能按 validation step/checkpoint 查看题目/样本 ID、输入、完整输出、标准答案、各奖励分量和可获得的结束原因/长度。必须保留实际数据文件；不能用训练 rollout 或训练后独立评测替代。
3. **全部约定数据集的完整评测表**：所有 manifest 数据集必须出现，给出训练前后结果、变化、样本数、split、grader、解码设置与状态。完整清单不等于已完成评测；未测/失败/部分完成必须显式标记，不能填 0、漏列或进入完整平均分。

不要求效果必须为正；有退步先给详细结果和逐题证据，再讨论原因。已测少量集合可作为 pilot 附表，不能代替全套表。

## 1. 训练前冻结与预审

### 执行工具的验收

- 参数迁移先逐项对照基线，区分任务要求的改变、框架兼容调整和无意遗漏。优化器、精度、裁剪、prompt、训练预算和评测解码不能静默变化。
- 交付远端命令前，实际执行Bash包装链与Hydra解析/真实YAML组合，至少覆盖正式202步和短训练1步、检查和训练两种入口。语法检查、手工构造的配置对象、测试数量均不能代替该路径。
- 按目标解释器检查：登录节点恢复工具Python3.6，容器训练工具Python3.12；测试中未覆盖的框架导入、CUDA与模型更新明确列为远端待验。
- 恢复作业先查询当前用户队列，再用sacct核对历史作业及退出码。不将旧作业ID失效视为脚本异常，不将查询失败视为作业已结束。状态未知则保留现场并说明缺少哪项证据。
- 申请GPU前完成可在CPU上完成的文件访问、路径、包哈希和配置检查。GPU作业再验证CUDA张量运算、模型更新、生成、验证与checkpoint。准备作业不能当训练进度。
- 同时核对Slurm分配与Ray登记的CPU/GPU，以及worker placement group需求；有GPU分配不代表Ray可以调度。双卡正式配置Ray登记8CPU，覆盖6CPU worker资源池和控制进程；不能沿用单卡pilot的4CPU上限。配置解析成功不等于worker调度通过。
- 资源验收须区分16条短验证与2848条完整验证。Job222374在128GiB限制下出现OUT_OF_MEMORY，不能用短验证通过证明正式峰值可承受。正式提交入口显式覆盖--mem=256G，并保存submission-resources.txt；256GiB为扩容后的待实测配置，不预先宣称足够。OOM恢复先查Slurm记账和checkpoint，保留初始验证与失败目录；有训练产物时先审查续跑，不默认从头启动。
- 重试保留已完成依赖、图片修复和旧日志；不重装、不重新初始化，不自动重复提交。恢复工具本身也要测试正常结束、已失败、仍活动、查询异常和状态未知。
- 发布修复时列明改动、真实通过的检查、远端未验证项和下一条命令。代码包及使用说明保持一致；不以本机测试通过承诺集群全链成功。

2026-11-04 补充设计约束：学长明确完整设计为 **202 steps，保守估计 2 卡**。原 GRPO 脚本的全量 manifest、rollout batch 384、2 epochs、2 GPUs 已隐含完整预算。不能仅因任务正文未单列步数就把预算当作未定义，也不能以 pilot 代替完整基线。正式配置预审须核对过滤后 N、dataloader 长度、有效 global batch、每题 rollout 数、epochs 与总步数的推导；结果与 202 不一致时先解释框架/过滤差异。202 不是任意小 batch 下都等价的硬编码步数。

双卡方案须同时核对调度分配、trainer GPU 数、world size、global/micro batch 和 rollout 并发。单 A800 的小规模试跑通过不验证原配置单卡可行；学长的资源估计不等于本轮已实测双卡显存/耗时。pilot 与正式基线分开记录，不把不同 batch 的步数直接相加或按比例计算完成率。

- 将当前脚本、解析后的实际配置、代码 commit/patch、模型/tokenizer/模板、奖励实现与权重提交给学长预审，执行发生在预审之后。已有同对象/范围/条件授权沿用，不重复请求。
- 固定 GRPO/DAPO 对照中的数据、模型、prompt、奖励、长度、图像设置、batch、每题采样数、优化器、学习率/warmup、预算和评测口径；不可等效迁移的旧参数注明原因。保存 requested 与 resolved config，不只看 launcher。
- 冻结训练及验证 manifest，检查来源、缺图、标签、重复与训练/验证重合。不能把构造数据集的占位路径当作正式验证选择。
- 确认覆盖所有约定数据集，另确认每个数据集全量还是抽样、n=1 还是多次采样及聚合方式。学长此次明确“所有数据集出现”，不据此擅自扩张每集样本数和 GPU 预算。
- 在个人隔离查看环境核对 SwanLab、dashboard 扩展、日志模式、媒体表支持；用非 GPU 的合成记录验收时明确为合成，不能混入真实实验。

2026-11-09恢复流程：Job221948因MathVerse验证图`images_version_6/image_745.png`读取权限600失败。通过`audit_validation_access.py`先收集五个验证源的完整不可读清单；权限由数据所有者或管理员处理，不能删题、补灰图或更改验证范围。用`recover_session.sh sync`建立新代码执行树并核对传输哈希，保留旧失败目录、复用已验证依赖；`prepare`通过sbatch进行后台检查。正式配置入口必须通过`formal_contract.py`的解析配置校验及原版提示词/优化器来源核对。训练集和周期验证集必须通过检查；十个最终test集保留完整清单，独立评测前严格审计，未核验不能标为完成。

## 2. 原生 SwanLab 本地目录与网页路径

官方流程：https://docs.swanlab.cn/guide_cloud/offline-board.html

模式区分：用于旧版原生离线看板的记录模式为 `local`；`offline` 二进制归档不能未经验证就视为可供 watch 读取的 local 目录。SwanLab 0.10.3 的 dashboard extra 对应 `swanboard==0.1.10b2`；后续升级需重新核对兼容性，不在共享训练环境安装依赖。

标准访问路径：**run → SwanLab 日志目录 → 独立个人查看环境中的 swanlab watch → 本地 URL/SSH 转发 → 浏览器核验**。

Remote-SSH / Bash 查看模板（仅适用于已确认可读取的日志和已准备好的查看环境）：

```bash
# 在查看机器的个人环境执行。VIEW_LOG_DIR 是实际日志目录，不是模型目录。
test -d "$VIEW_LOG_DIR"
python -m pip show swanlab swanboard
swanlab watch --help
swanlab watch "$VIEW_LOG_DIR"
# 默认地址以命令实际输出为准；官方示例为 http://127.0.0.1:5092。
```

若查看服务在远程机器，用 VS Code Ports 转发实际端口，或在本机另开终端：

```bash
ssh -N -L 5092:127.0.0.1:5092 <viewing-host>
```

不暴露公共监听、不做云同步。记录实际主机、绝对日志路径、run ID、SDK/dashboard 版本、启动命令、URL、可见指标与生成页截图。服务能启动不等于目标 run/数据可用。

旧日志打不开时，保留源文件与哈希，先核查格式和支持路线；必要时在新目录导入/重放并标记“历史重放”。不能改名冒充转换，不能伪造缺失指标或评测轮。原始归档可读取、静态曲线可画，与原生网页验收是不同状态。

## 3. 训练指标与评测轮记录

- training accuracy：逐 step 的真实训练生成 accuracy 分量均值及分母；training format：独立格式分量；training reward：混合总分。建议清晰命名为 `train/accuracy`、`train/format`、`train/reward`，实际 key 需记录，不能只改显示标题。
- 区分 accuracy（0–1）与百分比，区分总 reward、KL 后 reward、验证准确率，保留原始定义。历史推导的曲线必须标记派生及假设，不伪称当时原生记录。
- 启用训练前验证和约定频率的周期验证。当前接口为 `trainer.val_before_train`、`trainer.test_freq`、`trainer.log_val_generations`、`trainer.validation_data_dir`；其值与资源成本须进入预审。旧 pilot 的 False/-1 不作为正式提交默认。
- 将评测轮生成持久化到 validation_data_dir，并在 SwanLab 中展示。当前原生 logger 的入口为 `ValidationGenerationsLogger.log_generations_to_swanlab`，指标 `val/generations`；核查当前 SDK 的 echarts.Table 与 local 看板是否支持，不能只检查配置项存在。
- 网页可展示固定、可解释的样本子集，磁盘保留评测轮完整输出；训练 rollout 另存到 rollout_data_dir，按 train/validation 区分。
- 每轮核对 step、实际评测题数、输出行数、失败数、样本 ID、奖励分量及 checkpoint 对应关系。若某字段上游不提供，应记录缺失，不能推造。

### 格式奖励与生成展示的真实验收（2026-11-15补充）

- 首次真实生成时同时检查模型精确变体、实际prompt末尾、解码后的completion、format规则和各分量。若format整批为0，先区分缺开标签、缺双标签、boxed缺失、截断与特殊token处理；配置中写了system指令不等于输出已满足格式。
- 用手工正例确认奖励函数确实能给1分，再用同一启动链的真实回答检查可达性。仅把`fullmatch`换成`search`不能修复缺少全部think标签的输出。未查清整批零格式分原因时，不以“程序未报错”认定格式适配通过；若继续旧奖励实验，保留明确的旧奖励口径及异常记录。
- 第1步后及每十几步联查accuracy、format、总分、回答长度、触顶比例、KL、entropy和完整生成。触发异常先报告与排查，是否停训按当前授权处理。变更prompt、奖励或模型是新实验，不热改旧运行或回写历史成绩。
- 本机SwanLab0.10.3的`LocalCallbacker`仅将类型1～4映射为line/image/audio/text，ECharts为类型6，会被`_create_column`跳过。`media/echarts`文件存在不代表离线图表已登记。生成验收必须包括实际chart/tag和内容接口，以及最终网页检查。
- 历史ECharts表可在独立查看副本转换为受支持的text，注明来源与展示子集、不修改原表或重新生成答案。文本图在单实验面板查看，项目多实验面板的默认支持列表不含text。完整逐轮JSONL仍须另外保留。
- 2026-11-15学长进一步要求按11月11日会议片段改成新奖励：本地`reward_function/base.py`采用闭标签后boxed的search规则，旧实现冻结于`legacy_base.py`。此变更是新实验契约，未部署到历史202步运行；当前规则/hash见`reference-inputs.json`。先核实远端实际加载和首轮真实生成，不凭修改正则认定长度触顶已解决。
- Bao随后明确：从原始Qwen3-VL-2B-Instruct独立跑30步，今后默认保持该会议奖励。该次验证保持回答上限2048与batch384/n5，不恢复旧202步checkpoint；30步profile显式warmup10步以保留原202步前30步学习率轨迹。任务包位于仓库外`cfpo-next-delivery/validation30-pack-20261115`；是否已提交及运行状态必须查实际job回执，不据本地文件推断。

## 4. 全部数据集表

正式执行前固定清单版本/hash。当前暂以 `Counterfactual-Eval/datasets_config_math.json` 为源，共十项：geometry3k、WeMath、MMK12、MathVerse_fixed、LogicVista、MMMUPro-Vision、POPE、MARS_Bench、C-VQA-Real、TextVQA。与学长确认后更新清单；Geo3k、We-Math、MMMU-Pro(V) 等展示别名须保留映射。

每项必须有：数据版本/split、预期与实际 N、before/after 正确数及准确率、变化（百分点）、格式/截断率、评测状态、样本哈希、checkpoint、grader、n/temperature/长度与聚合方式。缺图、异常跳过或灰图替代必须被计数审计，不能静默减少分母。

MathVerse val 与 MathVerse_fixed test 不可直接合并；五集 val 抽样与十集 test 不直接比较。全套平均分仅在所有必测集按约定完成后计算，并注明宏平均/微平均；部分均值只能明确标为已测子集平均。

## 5. 退步审查

先验证两侧样本/配置/评分一致，再列全部正确性变化：对→错、错→对、均对、均错。对退步集逐题提供完整前后输出、答案、长度/结束原因及可用源图。

将“重复/截断”“正常结束但答案变化”“答案抽取/评分异常”“图像或条件解释差异”分开描述。输出文本不足以确认读图对错，必须回看源图；小样本单次变化不能自动推出稳定退化。先给详细证据，再提出待验证假设；改变长度、prompt、奖励或训练配置属于下一实验，不修改历史结果。

## 6. 提交包与收尾

- 提交说明：已完成、结果结论、异常、未完成与下一步。
- 原生看板访问说明及核验记录；training accuracy/reward 曲线；评测轮生成索引与实际产物路径。
- 全部数据集表（可编辑正文/机器可读汇总）及微信用图片；配置差异与复现记录；异常集逐题审查。
- PR 只纳入已授权的小型代码、汇总和图片；模型、数据、原始题目/输出和大日志继续按现有边界保存，分享需对应授权。图片是否真正上传须查远端 PR，不能以本地生成代替。
- 正式提交前逐项验收三项标准；缺项写为缺项，不以重排表格或静态网页追认完成。保留唯一源记录，链接引用，不自动联系学长或追加 GPU 作业。

## 当前历史运行的适用状态

Job 220236：20 步 pilot；未执行训练中验证，未原生记录独立 accuracy，日志为 offline 归档。Job 220325：五集各 64 题的独立对照。上述历史缺项不能靠修改当前 launcher 补成已完成。2026-11-04 审查材料位于仓库外 `cfpo-next-delivery/review-20261104/`，含 Geo3k 64 题本地输出审查及十集覆盖清单；不自动纳入 Git。
# 2026-11-09数据访问修复与评测口径

MathVerse验证样本MathVerse_3725的共享图像权限为600。使用学长公开数据源RavenInJuly/CFPO_Datasets固定revision 088ea718c4586fce8a3fed7eef6a76c28b869866的原图，在个人运行目录建立图像视图：仅该图片使用已校验公开字节，其余图片链接原共享路径。原标注不改、600条验证样本不删，记录来源、SHA-256与修复路径。远端完整访问和图像解码检查通过才进入短训练。

最终十集评测固定temperature=1.0、每题8次生成，指标为Avg@8；正式MathVerse使用MathVerse-closed_test.json，区别于训练监控验证集和MathVerse_test.json。不得用验证集监控结果代替最终评测表。
