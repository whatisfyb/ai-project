"""论文解析器

从 PDF 中提取关键内容（摘要、引言、结论）并入库到向量数据库。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from utils.document.file_loader import FileLoader
from utils.document.pdf_preprocessor import PDFPreprocessor
from utils.chunking.structure_chunker import StructureChunker
from utils.retrieval.vector_store import VectorStore


@dataclass
class PaperMeta:
    """论文元数据"""
    paper_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    year: Optional[int] = None
    source: str = ""  # arxiv, local, etc.
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    categories: list[str] = field(default_factory=list)


@dataclass
class PaperSection:
    """论文章节"""
    section_type: str  # abstract, introduction, conclusion
    content: str
    paper_id: str
    metadata: dict = field(default_factory=dict)


# 要提取的章节
TARGET_SECTIONS = {
    "abstract": ["abstract", "摘要", "摘　要"],
    "introduction": ["introduction", "引言"],
    "conclusion": ["conclusion", "conclusions", "结论", "结束语"],
}


# 内联章节模式（摘要、关键词等，格式为 "摘要：内容"）
INLINE_SECTION_PATTERNS = [
    # 摘要：内容 或 摘　要：内容（匹配到关键词或引言之前）
    re.compile(
        r"^(摘\s*要)[：:]\s*(.+?)(?=^关键词|^Key\s*words|^引言|^Introduction|\Z)",
        re.M | re.S
    ),
    # Abstract: content（匹配到 Keywords 或 Introduction 之前）
    re.compile(
        r"^(Abstract)[：:]\s*(.+?)(?=^Key\s*words|^Introduction|^引言|\Z)",
        re.I | re.M | re.S
    ),
]


class PaperParser:
    """论文解析器

    流程：
    1. 加载 PDF 文本
    2. 预处理（清理、合并段落）
    3. 结构解析（识别章节）
    4. 提取目标章节（摘要、引言、结论）
    5. 入库向量数据库
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.preprocessor = PDFPreprocessor()
        self.chunker = StructureChunker(mode="paper")
        self.vector_store = vector_store

    def parse_pdf(
        self,
        pdf_path: str,
        meta: Optional[PaperMeta] = None,
    ) -> list[PaperSection]:
        """解析 PDF 文件

        Args:
            pdf_path: PDF 文件路径
            meta: 论文元数据（可选，会尝试自动提取）

        Returns:
            提取的章节列表
        """
        # 1. 加载 PDF
        raw_text = FileLoader.load(pdf_path)

        # 2. 预处理
        clean_text = self.preprocessor.preprocess(raw_text)

        # 3. 提取元数据（如果未提供）
        if meta is None:
            meta = self._extract_metadata(clean_text, pdf_path)

        # 4. 结构解析
        sections = self._extract_sections(clean_text, meta.paper_id)

        # 5. 添加元数据到章节
        # 注意：ChromaDB 不接受空列表或 None，需要处理
        for section in sections:
            section.metadata = {
                "paper_id": meta.paper_id,
                "title": meta.title or "",
                "authors": ",".join(meta.authors) if meta.authors else "",
                "keywords": ",".join(meta.keywords) if meta.keywords else "",
                "year": meta.year if meta.year is not None else 0,
                "source": meta.source or "",
                "section": section.section_type,
            }

        return sections

    def parse_and_store(
        self,
        pdf_path: str,
        meta: Optional[PaperMeta] = None,
        collection_name: str = "papers",
    ) -> dict:
        """解析 PDF 并存入向量数据库

        Args:
            pdf_path: PDF 文件路径
            meta: 论文元数据
            collection_name: 向量库 collection 名称

        Returns:
            入库结果
        """
        if self.vector_store is None:
            self.vector_store = VectorStore(collection_name=collection_name)

        # 解析
        sections = self.parse_pdf(pdf_path, meta)

        # 转换为 Document
        documents = []
        ids = []

        for section in sections:
            doc = Document(
                page_content=section.content,
                metadata=section.metadata,
            )
            documents.append(doc)
            ids.append(f"{section.metadata['paper_id']}_{section.section_type}")

        # 入库
        if documents:
            self.vector_store.add_documents(documents, ids=ids)

        return {
            "paper_id": sections[0].metadata.get("paper_id") if sections else None,
            "sections_stored": len(documents),
            "ids": ids,
        }

    def _extract_metadata(self, text: str, pdf_path: str) -> PaperMeta:
        """从文本中提取元数据

        Args:
            text: 预处理后的文本
            pdf_path: PDF 路径

        Returns:
            论文元数据
        """
        # 生成默认 paper_id
        paper_id = Path(pdf_path).stem

        meta = PaperMeta(paper_id=paper_id)

        lines = [l.strip() for l in text.split("\n") if l.strip()]

        # 1. 尝试从 arXiv 模式提取（优先级最高）
        arxiv_match = re.search(r"arXiv[:\s]*(\d{4}\.\d{4,5})", text, re.I)
        if arxiv_match:
            meta.arxiv_id = arxiv_match.group(1)
            meta.source = "arxiv"
            meta.paper_id = f"arxiv_{meta.arxiv_id}"

        # 2. 提取标题
        # 策略：找第一个较长的、不像引用格式的行
        for i, line in enumerate(lines[:15]):
            # 跳过引用格式行
            if any(kw in line for kw in ["引用格式", "［Ｊ］", "[J]", "doi:", "DOI:", "et al"]):
                continue
            # 跳过机构/单位行（括号开头）
            if line.startswith("（") or line.startswith("("):
                continue
            # 跳过作者行（包含多个逗号分隔的姓名）
            if line.count("，") >= 2 or line.count(",") >= 2:
                # 可能是作者行，也提取作者
                if re.search(r"[\u4e00-\u9fa5]{2,4}[，,][\u4e00-\u9fa5]{2,4}", line):
                    authors = re.findall(r"[\u4e00-\u9fa5]{2,4}", line)
                    if len(authors) >= 2 and not meta.authors:
                        meta.authors = authors[:10]
                continue
            # 跳过太短或太长的行
            if len(line) < 10 or len(line) > 200:
                continue
            # 跳过以标点结尾的行（通常是正文）
            if line.endswith(("。", "，", ".", ",", "：", ":")):
                continue
            # 这行可能是标题
            meta.title = line
            break

        # 3. 如果没提取到作者，再尝试一次
        if not meta.authors:
            for i, line in enumerate(lines[:20]):
                # 跳过引用格式行
                if "引用格式" in line or "［Ｊ］" in line or "[J]" in line:
                    continue

                # 中文作者格式：姓名，姓名，姓名（支持全角空格）
                if "，" in line or "," in line:
                    # 提取所有中文姓名（2-4个字）
                    authors = re.findall(r"[\u4e00-\u9fa5]{2,4}", line)
                    # 过滤掉太短的，且这行不能只有一两个词
                    authors = [a for a in authors if len(a) >= 2]
                    if len(authors) >= 2:
                        meta.authors = authors[:10]
                        break

                # 英文作者格式
                if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+[,;]", line):
                    authors = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", line)
                    if len(authors) >= 1:
                        meta.authors = authors[:10]
                        break

        # 4. 提取关键词
        keywords_match = re.search(
            r"(?:关键词|Keywords|Key\s*words)[：:]\s*([^\n]+)",
            text, re.I
        )
        if keywords_match:
            keywords_str = keywords_match.group(1)
            # 分割关键词（支持全角/半角符号）
            keywords = re.split(r"[、，,;；　]", keywords_str)
            meta.keywords = [k.strip() for k in keywords if k.strip() and len(k.strip()) > 1][:10]

        # 5. 提取年份
        # 支持全角和半角数字
        # 匹配 19xx 或 20xx
        year_match = re.search(r"[１２][９０][０-９]{2}|[12][0-9]{2}", text)
        if year_match:
            year_str = year_match.group()
            # 全角转半角
            year_str = year_str.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
            try:
                meta.year = int(year_str)
            except ValueError:
                pass

        return meta

    def _extract_sections(self, text: str, paper_id: str) -> list[PaperSection]:
        """从文本中提取目标章节

        Args:
            text: 预处理后的文本
            paper_id: 论文 ID

        Returns:
            提取的章节列表
        """
        sections = []

        # 1. 先提取内联章节（摘要）
        text, inline_sections = self._extract_inline_sections(text)
        for section_type, content in inline_sections.items():
            if content:
                sections.append(PaperSection(
                    section_type=section_type,
                    content=content.strip(),
                    paper_id=paper_id,
                ))

        # 2. 按独立标题分割
        section_map = self._split_by_sections(text)

        # 提取目标章节（排除已提取的 abstract）
        for section_type, keywords in TARGET_SECTIONS.items():
            # 跳过已通过内联方式提取的
            if section_type == "abstract" and any(s.section_type == "abstract" for s in sections):
                continue

            content = None
            for keyword in keywords:
                if keyword.lower() in section_map:
                    content = section_map[keyword.lower()]
                    break

            if content:
                sections.append(PaperSection(
                    section_type=section_type,
                    content=content.strip(),
                    paper_id=paper_id,
                ))

        return sections

    def _extract_inline_sections(self, text: str) -> tuple[str, dict[str, str]]:
        """提取内联章节（如 摘要：内容）

        Args:
            text: 文本

        Returns:
            (处理后的文本, {section_type: content})
        """
        sections = {}

        for pattern in INLINE_SECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                section_title = match.group(1).lower()
                content = match.group(2).strip()

                # 映射到标准 section_type
                # 摘要可能包含各种空格：摘要、摘　要、摘 要
                if re.match(r"^摘\s*要$", section_title) or section_title == "abstract":
                    sections["abstract"] = content

                # 从文本中移除已提取的部分
                text = text[:match.start()] + text[match.end():]

        return text, sections

    def _split_by_sections(self, text: str) -> dict[str, str]:
        """按章节标题分割文本

        Args:
            text: 文本

        Returns:
            {section_name: section_content} 映射
        """
        lines = text.split("\n")
        section_map = {}
        current_section = None
        current_content = []

        for line in lines:
            stripped = line.strip()

            # 检查是否是章节标题
            section_name = self._match_section_name(stripped)

            if section_name:
                # 保存上一个章节
                if current_section and current_content:
                    section_map[current_section] = "\n".join(current_content)

                current_section = section_name
                current_content = []
            else:
                if current_section:
                    current_content.append(line)

        # 保存最后一个章节
        if current_section and current_content:
            section_map[current_section] = "\n".join(current_content)

        return section_map

    def _match_section_name(self, line: str) -> Optional[str]:
        """匹配章节名称

        Args:
            line: 一行文本

        Returns:
            标准化的章节名称，不匹配返回 None
        """
        line_stripped = line.strip()

        # 移除编号前缀:
        # - 阿拉伯数字: "1 Introduction", "1.1 Method", "４　结束语"
        # - 中文数字: "一、引言"
        line_clean = re.sub(r"^[\d一二三四五六七八九十]+[\.\s、　]+", "", line_stripped)
        line_lower = line_clean.lower()

        for section_type, keywords in TARGET_SECTIONS.items():
            for keyword in keywords:
                if keyword.lower() == line_lower:
                    return section_type

        return None


def parse_paper(
    pdf_path: str,
    meta: Optional[PaperMeta] = None,
    vector_store: Optional[VectorStore] = None,
) -> list[PaperSection]:
    """解析论文的便捷函数

    Args:
        pdf_path: PDF 文件路径
        meta: 论文元数据
        vector_store: 向量存储（用于入库）

    Returns:
        提取的章节列表
    """
    parser = PaperParser(vector_store)
    return parser.parse_pdf(pdf_path, meta)


@dataclass
class PaperValidationResult:
    """论文格式验证结果"""
    is_valid: bool
    missing_sections: list[str]
    sections_found: list[str]
    page_count: int
    message: str


def validate_paper_format(pdf_path: str) -> PaperValidationResult:
    """验证 PDF 是否符合论文格式

    要求：
    - 必须包含摘要（Abstract/摘要）

    可选章节（会提取但非必需）：
    - 引言（Introduction/引言）
    - 结论（Conclusion/结论）

    Args:
        pdf_path: PDF 文件路径

    Returns:
        验证结果
    """
    # 加载 PDF
    try:
        docs = FileLoader.load_documents(pdf_path)
    except Exception as e:
        return PaperValidationResult(
            is_valid=False,
            missing_sections=["abstract"],
            sections_found=[],
            page_count=0,
            message=f"无法加载 PDF: {str(e)}",
        )

    page_count = len(docs)

    # 预处理
    raw_text = "\n".join(doc.page_content for doc in docs)
    preprocessor = PDFPreprocessor()
    clean_text = preprocessor.preprocess(raw_text)

    # 检查是否包含章节
    sections_found = []
    missing_sections = []

    # 使用 paper_parser 的逻辑检查章节
    parser = PaperParser()
    inline_text, inline_sections = parser._extract_inline_sections(clean_text)

    # 检查摘要（可能是内联的）
    if "abstract" in inline_sections:
        sections_found.append("abstract")
    else:
        # 检查独立章节
        section_map = parser._split_by_sections(inline_text)
        if "abstract" in section_map:
            sections_found.append("abstract")

    # 检查引言和结论（可选）
    section_map = parser._split_by_sections(inline_text)
    for section_type in ["introduction", "conclusion"]:
        keywords = TARGET_SECTIONS.get(section_type, [])
        found = False
        for keyword in keywords:
            if keyword.lower() in section_map:
                found = True
                break
        if found:
            sections_found.append(section_type)
        else:
            missing_sections.append(section_type)

    # 验证：只需要有摘要
    is_valid = "abstract" in sections_found

    if is_valid:
        message = f"论文格式验证通过，共 {page_count} 页，包含: {', '.join(sections_found)}"
    else:
        message = f"论文格式验证失败，缺少摘要"

    return PaperValidationResult(
        is_valid=is_valid,
        missing_sections=missing_sections if not is_valid else [],
        sections_found=sections_found,
        page_count=page_count,
        message=message,
    )


def validate_and_parse(
    pdf_path: str,
    meta: Optional[PaperMeta] = None,
    vector_store: Optional[VectorStore] = None,
) -> tuple[bool, PaperValidationResult, Optional[list[PaperSection]]]:
    """验证论文格式并解析

    Args:
        pdf_path: PDF 文件路径
        meta: 论文元数据
        vector_store: 向量存储

    Returns:
        (是否成功, 验证结果, 章节列表)
    """
    # 1. 验证格式
    validation = validate_paper_format(pdf_path)

    if not validation.is_valid:
        return False, validation, None

    # 2. 解析论文
    parser = PaperParser(vector_store)
    sections = parser.parse_pdf(pdf_path, meta)

    return True, validation, sections
