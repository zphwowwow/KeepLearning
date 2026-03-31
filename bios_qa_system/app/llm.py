import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from app.config import Config

class LLMInference:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_PATH, trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            Config.BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, Config.MODEL_PATH)
        self.model.eval()
    
    def generate(self, query, context_list):
        """根据检索到的上下文生成答案"""
        # 构建prompt
        context_str = "\n\n".join([f"[来源: {c['source']} 第{c['page']}页]\n{c['content']}" for c in context_list])
        prompt = f"""<|im_start|>system
你是一个专业的BIOS/UEFI技术专家。请根据以下参考资料回答用户的问题。如果参考资料不足，请说明。
<|im_end|>
<|im_start|>user
参考资料：
{context_str}

问题：{query}
<|im_end|>
<|im_start|>assistant
"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                do_sample=True,
                top_p=0.9,
            )
        answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return answer