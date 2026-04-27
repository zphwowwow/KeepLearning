import pandas as pd
from PIL import Image
import io
import os

# 1. 导入原始数据集
data = pd.read_parquet("data/timetravel.parquet") 
data.iloc[:2,:]  #查看前两行内容  


# 2.数据探索
# 目的/明确问题：英译中采用何种方式效率更高更合理、是否有影响训练结果的重复数据需要去除
print("空值情况：")
data.info() 

print("数据集整体去重后记录数：",len(data.drop_duplicates()))

def col_duplicate(df,col):
    return len(set(df[col]))

for i in data.columns:
    print(i,"列去重后记录数：",col_duplicate(data,i))
    
# 观察可发现重复图像一般是因为多件文物共用一张图，因此不做去重处理。
data[data['Image'].duplicated(keep=False)]
# 除Image、id列外其余列都需要英译中


# 3.图像文件整理
# 将二进制图像数据转化为图像文件保存到指定目录（images/id.jpg）
dir = "data/images"  # 指定图像文件保存路径
if not os.path.exists(dir):
    os.mkdir(dir)
# 遍历原始数据中的图像字段，将二进制图像数据转存为指定路径下的图像文件
for i in range(len(data)):
    id = str(data["id"][i])
    Image.open(io.BytesIO(data['Image'][i])).convert("RGB").save(dir+"/"+id+".jpg",format="JPEG",quality=95)


# 4.英译中 + 问答对生成
# 异步调用API，加快模型处理速度
import asyncio
import json
import time
import pandas as pd
import openai
from openai import AsyncOpenAI,RateLimitError
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# 导入原始数据集
data = pd.read_parquet("data/timetravel.parquet")

PROMPT = Path("data/EN_to_CN_prompt-v2.txt").read_text(encoding="utf-8")
print(PROMPT)

# 初始化客户端
MODEL = "Pro/zai-org/GLM-4.7"
KEY = Path("data/silicon-key.txt").read_text()
CLIENT = AsyncOpenAI(api_key=KEY,base_url="https://api.siliconflow.cn/v1")

# 针对除了Image、id列外的其余列进行翻译与问题生成
en_data = data.iloc[:,1:].to_dict(orient="records")

# ============ 配置区 ============
@dataclass
class RateLimitConfig:
    rpm: int = 500                    # 厂商提供的 RPM 限制
    safety_margin: float = 0.8        # 安全余量（只用 80% 配额）
    batch_size: int = 100             # 每批处理数量（RPM 高可增大）
    max_retries: int = 5              # 单条最大重试次数
    base_retry_interval: float = 1.0  # 基础重试间隔（秒）
    
    @property
    def max_concurrent(self) -> int:
        """计算建议并发数：(RPM * 安全余量) / 60 * 平均请求耗时系数"""
        # 假设平均请求耗时 2-3 秒，并发数 ≈ 8-10
        return int((self.rpm * self.safety_margin) / 60 * 1.2) + 1

CONFIG = RateLimitConfig(rpm=500)
# semaphore = asyncio.Semaphore(CONFIG.max_concurrent)  # 约 8-10 并发
semaphore = asyncio.Semaphore(90)
# 实际观察因为每个请求可能要15秒左右才响应完成，此时RPM只有30，远远达不到平台500RPM的限制，所以手动调高了这里的并发数
# 按当前的配置生成问答对大概耗时6H左右

# ============ 核心请求函数 ============
async def call_model(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    data_row: Dict[str, Any],
    config: RateLimitConfig = CONFIG
) -> Dict[str, Any]:
    """
    调用大模型，针对 500 RPM 优化重试策略
    """
    row_id = data_row.get("id", "unknown")
    user_content = json.dumps(data_row, ensure_ascii=False)
    
    for attempt in range(1, config.max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                stream=False,
                response_format={"type": "json_object"},
                timeout=60  # 若实际调试确认服务响应快，可适当缩短超时
            )
            
            content = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "id": row_id,
                "content": content,
                "original_data": data_row,
                "usage": response.usage.dict() if response.usage else None,
                "attempts": attempt,
                "timestamp": time.time()
            }
            
        except RateLimitError as e:
            # 遇到 429 错误，使用指数退避 + 抖动
            wait = min(2 ** attempt, 60) + (hash(row_id) % 10) / 10  # 抖动避免羊群效应
            print(f"[{row_id}] 限流(429)，等待 {wait:.1f}s 后重试 ({attempt}/{config.max_retries})")
            await asyncio.sleep(wait)
            
        except asyncio.TimeoutError:
            print(f"[{row_id}] 超时，{attempt}/{config.max_retries}")
            if attempt < config.max_retries:
                await asyncio.sleep(config.base_retry_interval * attempt)
                
        except Exception as e:
            print(f"[{row_id}] 错误 {type(e).__name__}: {e}")
            if attempt < config.max_retries:
                await asyncio.sleep(config.base_retry_interval * attempt)
    
    return {
        "success": False,
        "id": row_id,
        "error": f"{config.max_retries} 次重试后失败",
        "original_data": data_row,
        "attempts": config.max_retries
    }


# ============ 并发控制 ============
async def call_model_limited(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    data_row: Dict[str, Any]
) -> Dict[str, Any]:
    async with semaphore:
        return await call_model(client, model_name, system_prompt, data_row)


# ============ 主处理流程（带动态速率监控）===========
async def process_dataset(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    data_list: List[Dict[str, Any]],
    output_path: str = "data/data_ch_v2.jsonl",
    config: RateLimitConfig = CONFIG
) -> List[Dict[str, Any]]:
    
    results = []
    total = len(data_list)
    start_time = time.time()
    request_count = 0
    
    print(f"配置: RPM={config.rpm}, 安全并发={config.max_concurrent}, 批大小={config.batch_size}")
    
    for batch_start in range(0, total, config.batch_size):
        batch = data_list[batch_start : batch_start + config.batch_size]
        batch_num = batch_start // config.batch_size + 1
        
        elapsed = time.time() - start_time
        current_rpm = (request_count / elapsed) * 60 if elapsed > 0 else 0
        eta = (total - batch_start) / (request_count / elapsed) / 60 if request_count > 0 else 0
        
        print(f"\n[{batch_num}] 进度 {batch_start}/{total} | 当前RPM: {current_rpm:.0f} | 预计剩余: {eta:.1f}min")
        
        tasks = [
            call_model_limited(client, model_name, system_prompt, row)
            for row in batch
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        request_count += len(batch)
        
        # 处理结果
        processed = []
        success_contents = []  # 用于落盘的成功内容
        
        for idx, res in enumerate(batch_results):
            row_id = batch[idx].get("id", f"idx{batch_start+idx}")
            
            if isinstance(res, Exception):
                processed.append({
                    "success": False,
                    "id": row_id,
                    "error": str(res),
                    "original_data": batch[idx]
                })
            else:
                processed.append(res)
                # 只收集成功的 answer 用于落盘
                try:
                    if res.get("success"):
                        success_contents.append(json.loads(res["content"]))
                except:
                    print(res["content"])
        
        # 只落盘成功的 answer（简化格式）
        if success_contents:
            with open(output_path, "a", encoding="utf-8") as f:
                for item in success_contents:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        success = sum(1 for r in processed if r.get("success"))
        print(f"  结果: {success}/{len(batch)} 成功 (落盘 {len(success_contents)} 条)")
        
        results.extend(processed)
        
        if success < len(batch) * 0.8:
            print("  [警告] 成功率低于 80%，建议检查是否触发限流")
    
    total_time = time.time() - start_time
    actual_rpm = (total / total_time) * 60
    print(f"\n{'='*50}")
    print(f"完成: {sum(1 for r in results if r.get('success'))}/{total}")
    print(f"总耗时: {total_time/60:.1f}min | 平均 RPM: {actual_rpm:.0f}")
    
    return results
    
# 开始异步处理
async def main():  
    results = await process_dataset(
        client=CLIENT,
        model_name=MODEL,
        system_prompt=PROMPT,
        data_list=tmp_data,
        output_path="data/data_ch_v2.jsonl"
    )
    
    return results


results = await main()

# 5.检查模型处理结果（主要针对格式、英译中进行检测）
###### 检查内容格式
import json
import re
from typing import List, Dict, Any, Tuple

def _is_chinese_content(text: str, prefix_len: int = 20) -> Tuple[bool, str]:
    """检查文本是否以中文开头（允许内容中包含外语单词）"""
    if not text or not text.strip():
        return False, "内容为空"
    
    cleaned = text.lstrip()
    check_part = cleaned[:prefix_len]
    
    if not re.search(r'[\u4e00-\u9fff]', check_part):
        return False, f"开头 {prefix_len} 个字符未检测到中文"
    
    return True, ""


def check_jsonl_file(filepath: str, verbose: bool = False) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    检查整个 JSONL 文件的格式是否符合规范（含中文内容校验）
    
    参数:
        filepath: JSONL 文件路径
        verbose: 是否输出详细错误报告（默认 False）
        
    返回:
        Tuple[List[Dict], List[int]]:
            - error_report: 未通过检查的行的详细报告
            - failed_line_indices: 未通过检查的行号列表（从 1 开始）
    """
    error_report = []
    failed_line_indices = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):  #行号从1开始记，方便排错定位
                line = line.strip()
                if not line:
                    continue

                errors = []
                current_id = None

                # JSON 语法检查
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    error_report.append({
                        "line": line_num,
                        "id": None,
                        "errors": [f"JSON格式错误: {str(e)}"]
                    })
                    failed_line_indices.append(line_num)
                    continue

                # 根对象类型检查
                if not isinstance(data, dict):
                    errors.append(f"根对象必须是字典，实际为 {type(data).__name__}")
                    error_report.append({"line": line_num, "id": None, "errors": errors})
                    failed_line_indices.append(line_num)
                    continue

                current_id = data.get('id', None)

                # 核心字段检查
                if 'id' not in data: 
                    errors.append("缺少必需字段 'id'")
                elif not isinstance(data['id'], (int, str)):
                    errors.append("字段 'id' 类型错误")

                if 'questions' not in data: 
                    errors.append("缺少必需字段 'questions'")
                elif not isinstance(data['questions'], list):
                    errors.append("字段 'questions' 类型错误")
                else:
                    for idx, qa in enumerate(data['questions']):
                        qa_tag = f"questions[{idx}]"
                        
                        if not isinstance(qa, dict):
                            errors.append(f"{qa_tag}: 元素类型错误")
                            continue

                        # question 字段检查
                        if 'question' not in qa:
                            errors.append(f"{qa_tag}: 缺少 'question'")
                        elif not isinstance(qa['question'], str) or not qa['question'].strip():
                            errors.append(f"{qa_tag}: 'question' 必须为非空字符串")
                        else:
                            is_zh, zh_err = _is_chinese_content(qa['question'])
                            if not is_zh:
                                errors.append(f"{qa_tag}: question-{zh_err}")

                        # answer 字段检查
                        if 'answer' not in qa:
                            errors.append(f"{qa_tag}: 缺少 'answer'")
                        elif not isinstance(qa['answer'], str) or not qa['answer'].strip():
                            errors.append(f"{qa_tag}: 'answer' 必须为非空字符串")
                        else:
                            is_zh, zh_err = _is_chinese_content(qa['answer'])
                            if not is_zh:
                                errors.append(f"{qa_tag}: answer-{zh_err}")

                if errors:
                    error_report.append({
                        "line": line_num,
                        "id": current_id,
                        "errors": errors
                    })
                    failed_line_indices.append(line_num)

    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {filepath}")
    except Exception as e:
        raise RuntimeError(f"读取文件时发生错误: {str(e)}")

    # 默认输出：仅统计 + 行号列表
    if failed_line_indices:
        print(f"无效数据: {len(failed_line_indices)} 条")
        print(f"异常行号列表: {failed_line_indices}")
    else:
        print("检查通过: 所有数据格式正常")

    # 详细模式：输出完整错误报告
    if verbose and error_report:
        print("\n详细错误报告:")
        for item in error_report:
            print(f"行 {item['line']} | 原始ID: {item['id']}")
            for err in item['errors']:
                print(f"  {err}")

    return error_report, failed_line_indices


FILE_PATH = "data/data_ch_v2.jsonl"

# 默认模式：只输出无效数据量和行号列表
# report, line_indices = check_jsonl_file(FILE_PATH)

# 如需查看详细错误，开启 verbose 模式：
report, line_indices = check_jsonl_file(FILE_PATH, verbose=True)

#对有问题的记录进行重新生成，如：
failed_data = [en_data[i] for i in range(len(en_data)) if i+1 in failed_line_indices]
#重新将该list传入第4步的异步处理中进行重新生成，注意修改输出文件名
#生成后再将有效数据覆盖到此前完成预处理的文件中
    
    
# 6. 数据划分与格式处理
from sklearn.model_selection import train_test_split
import random

def save_jsonl(data,path):
    with open(path,"w",encoding="utf-8") as f:
        for i in range(len(data)):
            f.write(json.dumps(data[i], ensure_ascii=False))
            f.write("\n")

ids = [json.loads(i).get("id") for i in ch_data]  # 获取数据id
random.seed(42)
random.shuffle(ids)  #打乱数据集
test_ids = ids[:250]  #共10250条数据，抽取250条作为测试集剩余部分按9:1的比例划分为训练集、验证集
train_ids = ids[250:9250]
val_ids = ids[9250:]

train_data = []
val_data = []
test_data = []
for i in ch_data:
    i = json.loads(i)
    s_id = i.get("id")
    if (s_id in train_ids):
        train_data.append(i)
    elif (s_id in val_ids):
        val_data.append(i)
    else:
        test_data.append(i)

save_jsonl(train_data,"./train_data_v2.jsonl")
save_jsonl(val_data,"./val_data_v2.jsonl")
save_jsonl(test_data,"./test_data_v2.jsonl")
