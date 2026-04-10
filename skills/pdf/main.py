"""PDF Reader Skill - 读取和解析 PDF 文件"""

from typing import Any

from utils.file_loader import FileLoader
from utils.pdf_preprocessor import PDFPreprocessor
from utils.paper_parser import PaperParser, PaperMeta


def run(
    file_path: str,
    action: str = "read",
    max_pages: int = 0,
    **kwargs
) -> dict[str, Any]:
    """读取和解析 PDF 文件

    Args:
        file_path: PDF 文件路径
        action: 操作类型
            - read: 读取全文
            - sections: 提取论文章节
            - summarize: 生成摘要
        max_pages: 最大读取页数（0 表示不限制）
        **kwargs: 额外参数

    Returns:
        结果字典
    """
    if not file_path or not file_path.strip():
        return {
            "success": False,
            "error": "文件路径不能为空",
            "data": None,
        }

    file_path = file_path.strip()

    # 检查文件是否存在
    from pathlib import Path
    if not Path(file_path).exists():
        return {
            "success": False,
            "error": f"文件不存在: {file_path}",
            "data": None,
        }

    # 检查文件类型
    if not file_path.lower().endswith('.pdf'):
        return {
            "success": False,
            "error": "只支持 PDF 文件",
            "data": None,
        }

    try:
        if action == "read":
            return _read_full(file_path, max_pages)
        elif action == "sections":
            return _extract_sections(file_path)
        elif action == "summarize":
            return _summarize(file_path)
        else:
            return {
                "success": False,
                "error": f"未知操作: {action}。支持: read, sections, summarize",
                "data": None,
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"处理失败: {str(e)}",
            "data": None,
        }


def _read_full(file_path: str, max_pages: int) -> dict[str, Any]:
    """读取 PDF 全文"""
    # 加载 PDF
    docs = FileLoader.load_documents(file_path)

    # 限制页数
    if max_pages > 0:
        docs = docs[:max_pages]

    # 提取文本
    text_parts = []
    for i, doc in enumerate(docs):
        page_num = i + 1
        text_parts.append(f"=== 第 {page_num} 页 ===\n{doc.page_content}")

    raw_text = "\n\n".join(text_parts)

    # 预处理
    preprocessor = PDFPreprocessor()
    clean_text = preprocessor.preprocess(raw_text)

    # 统计信息
    total_pages = len(docs)
    total_chars = len(clean_text)

    return {
        "success": True,
        "action": "read",
        "data": {
            "file_path": file_path,
            "total_pages": total_pages,
            "total_chars": total_chars,
            "content": clean_text,
        },
    }


def _extract_sections(file_path: str) -> dict[str, Any]:
    """提取论文章节（摘要、引言、结论）"""
    parser = PaperParser()
    sections = parser.parse_pdf(file_path)

    if not sections:
        return {
            "success": True,
            "action": "sections",
            "data": {
                "file_path": file_path,
                "message": "未能识别到标准论文章节（摘要、引言、结论）",
                "sections": [],
            },
        }

    # 格式化结果
    formatted_sections = []
    for s in sections:
        formatted_sections.append({
            "type": s.section_type,
            "content": s.content,
            "length": len(s.content),
        })

    return {
        "success": True,
        "action": "sections",
        "data": {
            "file_path": file_path,
            "sections_found": len(formatted_sections),
            "sections": formatted_sections,
        },
    }


def _summarize(file_path: str) -> dict[str, Any]:
    """生成文档摘要"""
    # 先提取章节
    parser = PaperParser()
    sections = parser.parse_pdf(file_path)

    if not sections:
        # 没有识别到章节，读取全文生成摘要
        raw_text = FileLoader.load(file_path)
        preprocessor = PDFPreprocessor()
        clean_text = preprocessor.preprocess(raw_text)

        # 截取前 5000 字符生成摘要
        text_to_summarize = clean_text[:5000]
        summary = _generate_summary(text_to_summarize, "文档")

        return {
            "success": True,
            "action": "summarize",
            "data": {
                "file_path": file_path,
                "summary": summary,
                "note": "未能识别标准论文章节，基于全文生成摘要",
            },
        }

    # 基于提取的章节生成摘要
    sections_text = []
    for s in sections:
        sections_text.append(f"【{s.section_type.upper()}】\n{s.content}")

    combined_text = "\n\n".join(sections_text)
    summary = _generate_summary(combined_text, "论文")

    return {
        "success": True,
        "action": "summarize",
        "data": {
            "file_path": file_path,
            "summary": summary,
            "sections_used": [s.section_type for s in sections],
        },
    }


def _generate_summary(text: str, doc_type: str = "文档") -> str:
    """使用 LLM 生成摘要"""
    from utils.llm import get_llm_model

    llm = get_llm_model()

    prompt = f"""请为以下{doc_type}内容生成一份简洁的摘要。

要求：
1. 概括主要内容（200字以内）
2. 列出 3-5 个关键要点
3. 使用中文

内容：
{text[:3000]}

请按以下格式输出：

## 摘要
[内容概要]

## 关键要点
1. ...
2. ...
3. ...
"""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"摘要生成失败: {str(e)}"
