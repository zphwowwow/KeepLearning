data文件夹：项目使用数据
	- 【修订版】XX公司员工手册.docx ：原始知识文件
	- train：训练集
	- val：验证集
	- test：测试集

code/output文件夹：模型训练输出的Checkpoint

code代码文件说明：
	- requirements-gpu.txt：项目环境所需依赖
	- deepseek_api.py：通过API调用厂商提供的Deepseek大模型
	- qwen3-1.7B_infer.py：调用Qwen3-1.7B大模型进行推理
	- qwen3-1.7B-lora_train.py：基于Qwen3-1.7B大模型进行LoRA微调
	- qwen3-4B-qlora_train.py：基于Qwen3-4B大模型进行QLoRA微调（量化+LoRA）
	- qwen3-0.6B-fft_train.py：基于Qwen3-0.6B大模型进行全量微调
	- train_loss_plt.py：绘制训练过程的损失值变化曲线
	- qwen3-1.7B-lora_save.py：合并微调得到的权重与原模型权重，并保存为新的模型
	- qwen3-1.7B-lora_verify.py：针对进行LoRA微调后的Qwen3-1.7B模型进行评估
	- qwen3-1.7B-lora_verify_mult.py：多线程调用多个高阶模型对微调后的大模型进行综合评估