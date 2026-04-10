"""PDF 文本预处理

解决 PDF 提取后的常见问题：
- 单词换行打断 (Intro-\nduction)
- 页眉页脚干扰
- 多余换行
- 特殊字符
"""

import re
from collections import Counter
from typing import Optional


class PDFPreprocessor:
    """PDF 文本预处理器"""

    # 页码模式
    PAGE_NUMBER_PATTERNS = [
        # - 1 -, - 10 -
        re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$"),
        # Page 1, Page 10
        re.compile(r"^\s*Page\s+\d+\s*$", re.I),
        # [1], [10]
        re.compile(r"^\s*\[\d+\]\s*$"),
        # 纯数字行（单独）
        re.compile(r"^\s*\d+\s*$"),
        # -1-, -10-
        re.compile(r"^\s*-\d+-\s*$"),
    ]

    # 连字符打断模式
    HYPHENATED_WORD_PATTERN = re.compile(r"(\w+)-\s*\n\s*(\w+)")

    # 页眉页脚阈值：出现在超过此比例的"页"中则视为页眉页脚
    HEADER_FOOTER_THRESHOLD = 0.5

    # 最小行长：短于此长度的行不参与页眉页脚检测
    MIN_LINE_LENGTH = 10

    def preprocess(self, text: str) -> str:
        """预处理 PDF 文本

        Args:
            text: PDF 提取的原始文本

        Returns:
            预处理后的干净文本
        """
        if not text or not text.strip():
            return ""

        # 1. 合并打断的单词
        text = self._merge_hyphenated_words(text)

        # 2. 移除页码模式
        text = self._remove_page_numbers(text)

        # 3. 移除页眉页脚
        text = self._remove_headers_footers(text)

        # 4. 合并段落
        text = self._merge_paragraphs(text)

        # 5. 标准化空白
        text = self._normalize_whitespace(text)

        return text.strip()

    def _merge_hyphenated_words(self, text: str) -> str:
        """合并打断的单词

        Example:
            "Intro-\nduction" -> "Introduction"
            "well-\nknown" -> "well-known"
        """
        def replace_hyphenated(match):
            prefix = match.group(1)
            suffix = match.group(2)
            # 如果后缀首字母小写，说明是同一个单词被打断
            if suffix[0].islower():
                return prefix + suffix
            # 否则保留连字符（可能是复合词）
            return prefix + "-" + suffix

        return self.HYPHENATED_WORD_PATTERN.sub(replace_hyphenated, text)

    def _remove_page_numbers(self, text: str) -> str:
        """移除页码行"""
        lines = text.split("\n")
        filtered = []

        for line in lines:
            stripped = line.strip()
            # 检查是否匹配页码模式
            is_page_number = any(p.match(stripped) for p in self.PAGE_NUMBER_PATTERNS)
            if not is_page_number:
                filtered.append(line)

        return "\n".join(filtered)

    def _remove_headers_footers(self, text: str) -> str:
        """移除页眉页脚

        基于频率统计：出现在多"页"中的相同行视为页眉页脚。
        由于 PDF 解析后没有明确的页边界，我们用空行或特定模式来分割"页"。
        """
        lines = text.split("\n")

        # 尝试识别"页"边界（连续多个空行或分页符）
        pages = self._split_into_pages(lines)

        if len(pages) < 2:
            # 少于2页，无法统计，直接返回
            return text

        # 统计每行出现的页数
        line_page_count = Counter()
        for page_lines in pages:
            # 每页中的唯一行（去重，避免同页多次出现）
            unique_lines = set()
            for line in page_lines:
                stripped = line.strip()
                if len(stripped) >= self.MIN_LINE_LENGTH:
                    unique_lines.add(stripped)
            for line in unique_lines:
                line_page_count[line] += 1

        # 识别页眉页脚
        threshold = max(2, len(pages) * self.HEADER_FOOTER_THRESHOLD)
        header_footer_lines = {
            line for line, count in line_page_count.items()
            if count >= threshold
        }

        # 过滤掉页眉页脚
        filtered = []
        for line in lines:
            stripped = line.strip()
            if stripped not in header_footer_lines:
                filtered.append(line)

        return "\n".join(filtered)

    def _split_into_pages(self, lines: list[str]) -> list[list[str]]:
        """将文本行分割成"页"

        使用连续空行或分页符作为边界。
        """
        pages = []
        current_page = []
        empty_count = 0

        for line in lines:
            if line.strip() == "":
                empty_count += 1
                # 连续3个或以上空行视为分页
                if empty_count >= 3 and current_page:
                    pages.append(current_page)
                    current_page = []
                else:
                    current_page.append(line)
            else:
                empty_count = 0
                current_page.append(line)

        if current_page:
            pages.append(current_page)

        return pages

    def _merge_paragraphs(self, text: str) -> str:
        """合并段落

        将被错误分割的段落行合并。
        规则：
        - 如果一行不以句号、问号、感叹号结尾，且下一行以字母开头，则合并
        """
        lines = text.split("\n")
        if not lines:
            return ""

        merged = []
        current_para = lines[0]

        for i in range(1, len(lines)):
            prev_line = lines[i - 1].strip()
            curr_line = lines[i].strip()

            # 空行：段落分隔
            if not curr_line:
                merged.append(current_para)
                merged.append("")
                current_para = ""
                continue

            # 如果没有累积内容，直接开始新段落
            if not current_para:
                current_para = lines[i]
                continue

            # 判断是否需要合并
            should_merge = self._should_merge_lines(prev_line, curr_line)

            if should_merge:
                # 合并，加一个空格
                current_para = current_para.rstrip() + " " + lines[i].lstrip()
            else:
                # 新段落
                merged.append(current_para)
                current_para = lines[i]

        if current_para:
            merged.append(current_para)

        return "\n".join(merged)

    def _should_merge_lines(self, prev_line: str, curr_line: str) -> bool:
        """判断两行是否应该合并为同一段落

        Args:
            prev_line: 上一行（已 strip）
            curr_line: 当前行（已 strip）

        Returns:
            是否应该合并
        """
        if not prev_line or not curr_line:
            return False

        # 上一行以句子结束符结尾：不合并
        if prev_line.endswith((".", "。", "!", "！", "?", "？", ":", "：", ";", "；")):
            return False

        # 上一行是标题模式（短且不以标点结尾）：不合并
        if len(prev_line) < 50 and not any(c in prev_line for c in "。！？.!?:;"):
            # 可能是标题
            return False

        # 当前行以小写字母开头：合并
        if curr_line[0].islower():
            return True

        # 当前行以大写字母开头，但上一行明显未结束：合并
        # （英文段落中间可能有大写，如专有名词）
        # 这个规则比较激进，暂时不用
        # if curr_line[0].isupper() and not prev_line.endswith((".", "。", "!", "！", "?", "？")):
        #     return True

        return False

    def _normalize_whitespace(self, text: str) -> str:
        """标准化空白字符"""
        # 多个连续空格变为单个
        text = re.sub(r"[ \t]+", " ", text)
        # 多个连续空行变为最多两个
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 移除行首行尾空白
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines)


def preprocess_pdf_text(text: str) -> str:
    """预处理 PDF 文本的便捷函数

    Args:
        text: PDF 提取的原始文本

    Returns:
        预处理后的干净文本
    """
    preprocessor = PDFPreprocessor()
    return preprocessor.preprocess(text)
