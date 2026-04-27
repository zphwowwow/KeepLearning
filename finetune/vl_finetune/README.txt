code文件夹：存放项目相关的代码及输出文件
	- data_preprocess.py：数据预处理
	- finetune.py：基于Qwen3-VL-4B-Instruct进行LoRA微调
	- evaluate.py：模型效果评估
	- requirements_web_demo.txt：前端交互界面相关依赖库
	- web_demo_mm.py：前端交互界面代码
	- Qwen3-VL-main.zip：官方代码库(可参照PPT文件夹中的使用方法.pdf进行使用）
	- 基于官方代码简化的图文场景LoRA微调代码.py：如命名所言，可辅助理解官方代码（建议先看懂这份，再展开看Qwen3-VL-main中finetune模块的代码
	- requirements.txt：使用Qwen3-VL官方微调代码时，环境相关依赖库
	- flash_attn-2.6.3+cu123torch2.4cxx11abiTRUE-cp312-cp312-linux_x86_64.whl：flash-attention2的二进制安装包，官方代码微调环境准备需要使用
	
code/output文件夹：模型微调训练输出的Checkpoint

data文件夹：存放项目相关的数据文件
	- timetravel.parquet：原始数据集
	- EN_to_CN_prompt-v2.txt：【英译中+问答对生成】的提示词
	- data_ch_v2.jsonl：大模型辅助处理好的中文问答对数据
	- train_data_v2.jsonl：训练集
	- val_data_v2.jsonl：验证集
	- test_data_v2.jsonl ：测试集
	- test_QNA_v2.jsonl：基座模型、微调模型对测试集的推理结果，【问题+标准答案+基座模型回答+微调模型回答】
	- evaluate_prompt.txt：将推理结果交由高阶模型进行评估的提示词
	- judge_result.jsonl：高阶模型评估结果
