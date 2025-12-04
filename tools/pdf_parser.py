import os
import time
import json
import requests
import zipfile
import io
import shutil
from typing import List, Tuple, Dict, Any
from pathlib import Path
from omegaconf import DictConfig

# 引入你的 schema
from schema import TextChunk, FigureData

class PDFParser:
    def __init__(self, config: DictConfig):
        self.config = config
        self.api_key = config.api.mineru_token
        self.base_url = config.api.mineru_base_url
        
        # MinerU 批量接口 URL
        self.batch_url = f"{self.base_url}/file-urls/batch"
        self.result_url_tpl = f"{self.base_url}/extract-results/batch/{{}}"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.img_output_dir = Path(config.pdf_parser.image_output_dir)
        self.img_output_dir.mkdir(parents=True, exist_ok=True)

    def parse_pdf(self, pdf_path: str) -> Tuple[List[TextChunk], List[FigureData]]:
        """
        主流程：获取上传链接 -> 上传文件 -> 轮询结果 -> 下载并解析 ZIP -> 返回结构化数据
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
            
        print(f"[*] [MinerU] 开始解析 PDF: {pdf_path}")
        file_name = os.path.basename(pdf_path)
        
        try:
            # 1. 获取预签名上传 URL
            batch_id, upload_url = self._get_upload_url(file_name)
            print(f"[*] [MinerU] Batch ID: {batch_id}")
            
            # 2. 上传文件 (PUT)
            self._upload_file(upload_url, pdf_path)
            print(f"[*] [MinerU] 文件上传成功，等待服务端处理...")
            
            # 3. 轮询任务状态
            task_result = self._poll_batch_task(batch_id, file_name)
            
            # 4. 下载并清洗数据
            return self._process_zip_result(task_result['full_zip_url'], pdf_path)
            
        except Exception as e:
            print(f"[!] [MinerU] 解析流程发生错误: {str(e)}")
            raise e

    def _get_upload_url(self, file_name: str) -> Tuple[str, str]:
        """第一步：请求上传链接 (Batch API)"""
        data = {
            "files": [
                {"name": file_name}
            ],
            "model_version": "vlm" # 使用 VLM 版本通常对图片处理更好，也可选 pipeline
        }
        
        resp = requests.post(self.batch_url, headers=self.headers, json=data)
        
        if resp.status_code != 200:
            raise Exception(f"请求API失败 Status: {resp.status_code}, Msg: {resp.text}")
            
        resp_json = resp.json()
        if resp_json.get('code') != 0:
            raise Exception(f"获取上传链接失败 Code: {resp_json.get('code')}, Msg: {resp_json.get('msg')}")
            
        batch_id = resp_json['data']['batch_id']
        file_urls = resp_json['data']['file_urls']
        
        if not file_urls:
            raise Exception("API返回的 file_urls 为空")
            
        return batch_id, file_urls[0]

    def _upload_file(self, upload_url: str, file_path: str):
        """第二步：PUT 文件内容 (注意：不设置 Content-Type)"""
        with open(file_path, 'rb') as f:
            # 根据文档：上传文件时，无须设置 Content-Type 请求头
            resp = requests.put(upload_url, data=f)
            if resp.status_code != 200:
                raise Exception(f"文件上传到OSS失败, Code: {resp.status_code}")

    def _poll_batch_task(self, batch_id: str, target_file_name: str) -> Dict[str, Any]:
        """第三步：轮询 Batch 状态"""
        url = self.result_url_tpl.format(batch_id)
        
        start_time = time.time()
        timeout = self.config.api.timeout  # 从配置读取超时时间
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("解析任务超时")

            try:
                resp = requests.get(url, headers=self.headers)
                if resp.status_code != 200:
                    print(f"[!] 查询状态接口异常: {resp.status_code}")
                    time.sleep(5)
                    continue
                
                resp_json = resp.json()
                if resp_json.get('code') != 0:
                    raise Exception(f"查询任务失败: {resp_json.get('msg')}")

                # 提取对应文件的结果
                extract_results = resp_json['data'].get('extract_result', [])
                target_task = next((t for t in extract_results if t['file_name'] == target_file_name), None)
                
                if not target_task:
                    print("[*] 任务尚未进入队列，等待中...")
                    time.sleep(3)
                    continue

                state = target_task.get('state')
                
                # 处理各种状态
                if state == 'done':
                    print(f"[*] 解析完成! 用时: {time.time() - start_time:.1f}s")
                    return target_task
                
                elif state == 'failed':
                    err_msg = target_task.get('err_msg', '未知错误')
                    raise Exception(f"服务端解析失败: {err_msg}")
                
                elif state in ['running', 'converting', 'waiting-file', 'pending']:
                    progress_info = ""
                    if state == 'running' and 'extract_progress' in target_task:
                        prog = target_task['extract_progress']
                        curr = prog.get('extracted_pages', 0)
                        total = prog.get('total_pages', '?')
                        progress_info = f" ({curr}/{total} 页)"
                    
                    print(f"[*] 状态: {state}{progress_info}...")
                    time.sleep(5)
                
                else:
                    print(f"[?] 未知状态: {state}, 等待中...")
                    time.sleep(5)

            except json.JSONDecodeError:
                time.sleep(3)
            except Exception as e:
                print(f"[!] 轮询异常: {e}")
                # 如果是致命错误可以在这里 break，否则继续重试
                if "服务端解析失败" in str(e):
                    raise e
                time.sleep(5)

    def _process_zip_result(self, zip_url: str, original_pdf_path: str) -> Tuple[List[TextChunk], List[FigureData]]:
        """第四步：下载 ZIP 并转换为 schema 对象"""
        print(f"[*] 正在下载结果: {zip_url}")
        r = requests.get(zip_url)
        if r.status_code != 200:
            raise Exception("下载解析结果ZIP失败")
            
        text_chunks: List[TextChunk] = []
        figure_data_list: List[FigureData] = []
        pdf_stem = Path(original_pdf_path).stem

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            file_list = z.namelist()
            
            # 1. 优先寻找 content_list.json (包含详细结构化信息)
            json_files = [n for n in file_list if n.endswith('content_list.json')]
            md_files = [n for n in file_list if n.endswith('.md')]
            image_files = [n for n in file_list if n.startswith('images/') and not n.endswith('/')]
            
            # --- 处理图片 ---
            for img_file in image_files:
                img_name = Path(img_file).name
                # 为了防止不同PDF图片重名，加上PDF前缀
                unique_img_name = f"{pdf_stem}_{img_name}"
                save_path = self.img_output_dir / unique_img_name
                
                with open(save_path, 'wb') as f:
                    f.write(z.read(img_file))
                
                # 创建 FigureData
                # 注意：MinerU 目前不直接在 images 目录给 caption，
                # caption 通常在 json/md 的引用上下文中。这里先留空，后续 Agent 可能会补充。
                figure_data_list.append(FigureData(
                    figure_id=unique_img_name,
                    page_number=0, # 暂时无法仅从文件名确知页码，需从json解析关联，此处设默认
                    image_path=str(save_path.absolute()), # 使用绝对路径方便后续读取
                    caption=None
                ))

            # --- 处理文本 ---
            if json_files:
                # 推荐：使用 JSON 格式解析，包含页码信息
                content_data = json.loads(z.read(json_files[0]))
                if isinstance(content_data, list):
                    for item in content_data:
                        # 仅提取正文文本，type 'text' 或 'table_caption' 等
                        # MinerU json 结构: {"type": "text", "text": "...", "page_idx": 0}
                        if item.get('type') in ['text', 'title', 'section_header'] and item.get('text'):
                            text_chunks.append(TextChunk(
                                page_number=item.get('page_idx', 0) + 1, # 转为从1开始
                                content=item.get('text').strip()
                            ))
                        
                        # 尝试修正图片的页码 (如果有对应的 image type)
                        if item.get('type') == 'image' and item.get('img_path'):
                            img_filename = Path(item.get('img_path')).name
                            target_id = f"{pdf_stem}_{img_filename}"
                            # 在 figure_list 中找到对应对象更新页码
                            for fig in figure_data_list:
                                if fig.figure_id == target_id:
                                    fig.page_number = item.get('page_idx', 0) + 1
                                    # 如果 JSON 里有 caption 字段也可以赋值
                                    if item.get('caption'):
                                        fig.caption = item.get('caption')

            elif md_files:
                # 备选：使用 Markdown 解析 (页码信息可能丢失或不准)
                print("[!] 未找到 content_list.json，使用 Markdown 解析，页码默认为 1")
                content = z.read(md_files[0]).decode('utf-8')
                # 简单按段落分割，实际生产中可能需要按 Markdown 标题分割
                paragraphs = content.split('\n\n')
                for p in paragraphs:
                    if p.strip():
                        text_chunks.append(TextChunk(
                            page_number=1,
                            content=p.strip()
                        ))
            
            else:
                print("[!] 压缩包中未找到有效的文本数据文件")

        print(f"[*] 解析完成: 提取文本段 {len(text_chunks)} 个, 图片 {len(figure_data_list)} 张")
        return text_chunks, figure_data_list