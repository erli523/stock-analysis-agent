#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识库向量索引构建脚本
首次使用或知识库文件更新后运行此脚本，约需 1~3 分钟（取决于文档数量和网络速度）。

用法：
    conda activate stock-agent
    python build_rag_index.py            # 仅在索引为空时构建
    python build_rag_index.py --rebuild  # 强制重建（适用于更新了知识库文件后）
"""
import argparse
import json
import os
import sys
import time

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(".env")

from config.config import get_data_embeddings_path, get_embedding_config
from hengline.logger import info, error, warning
from hengline.tools.llama_index_tool import create_index_from_directory, _index_has_content


def get_kb_stats(kb_dir: str):
    """统计知识库文件信息。"""
    total, files = 0, []
    for root, _, names in os.walk(kb_dir):
        for name in names:
            if name.endswith((".txt", ".md", ".pdf")):
                path = os.path.join(root, name)
                files.append(os.path.relpath(path, kb_dir))
                total += 1
    return total, files


def main():
    parser = argparse.ArgumentParser(description="构建/重建 RAG 知识库向量索引")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引（即使已存在）")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(root, "knowledge_base")
    embed_dir = get_data_embeddings_path()

    print("=" * 60)
    print("  Stock Analysis Agent -- RAG Knowledge Base Index Builder")
    print("=" * 60)

    # 统计知识库文件
    doc_count, doc_files = get_kb_stats(kb_dir)
    print(f"\n[DIR] 知识库目录: {kb_dir}")
    print(f"[DOC] 找到文档数量: {doc_count} 个文件")
    for f in doc_files:
        print(f"   - {f}")

    # 检查 embedding 配置
    emb_cfg = get_embedding_config()
    print(f"\n[CFG] Embedding 配置:")
    print(f"   provider : {emb_cfg.get('provider')}")
    print(f"   model    : {emb_cfg.get('model_name')}")
    print(f"   base_url : {emb_cfg.get('base_url', '(default)')}")
    api_key = emb_cfg.get("api_key", "")
    print(f"   api_key  : {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")

    # 检查现有索引状态
    has_content = _index_has_content(embed_dir)
    print(f"\n[IDX] 向量索引目录: {embed_dir}")
    print(f"   当前状态: {'[OK] 已有内容' if has_content else '[EMPTY] 空索引（未构建）'}")

    if has_content and not args.rebuild:
        print("\n索引已存在且非空，无需重建。如需强制重建请加 --rebuild 参数。")
        print("=" * 60)
        return

    if args.rebuild:
        print("\n[WARN] 将强制重建索引...")
    else:
        print("\n[START] 开始构建向量索引...")

    print("（调用 DashScope text-embedding-v4 向量化文档，请稍候...）\n")

    t0 = time.time()
    try:
        index = create_index_from_directory(
            directory_path=kb_dir,
            index_name="stock_knowledge_base",
            storage_dir=embed_dir,
            recursive=True,
            rebuild=True,   # 本脚本始终强制重建
        )
        elapsed = time.time() - t0

        # 验证结果
        vs_file = os.path.join(embed_dir, "default__vector_store.json")
        if os.path.exists(vs_file):
            with open(vs_file, "r", encoding="utf-8") as f:
                vs_data = json.load(f)
            vec_count = len(vs_data.get("embedding_dict", {}))
        else:
            vec_count = 0

        print(f"\n[OK] 索引构建完成！")
        print(f"   耗时       : {elapsed:.1f} 秒")
        print(f"   向量块数量 : {vec_count} 个")
        print(f"   保存位置   : {embed_dir}")
        print("\n现在可以启动应用，Agent 将自动使用知识库进行 RAG 增强分析。")

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[ERROR] 索引构建失败（{elapsed:.1f}s）: {e}")
        print("\n常见原因及解决方案:")
        print("  1. API Key 无效 -> 检查 .env 中 EMBEDDING_API_KEY 是否正确")
        print("  2. 网络问题     -> 检查网络连接是否可以访问 DashScope API")
        print("  3. 缺少依赖     -> pip install llama-index-embeddings-openai")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
