"""
从HTML格式中提取文本

文本提取函数：
    resiliparse.extract.html2text.extract_plain_text
    由于该函数接受的是字符串（string）而不是字节串（bytes），
    因此你需要首先将输入的字节串解码（decode）为 Unicode 字符串。
    需要注意的是：
        输入字节串未必采用 UTF-8 编码，
        因此当 UTF-8 解码失败时，函数应该能够自动检测编码格式
        Resiliparse 提供了如下工具：
            resiliparse.parse.encoding.detect_encoding()

"""

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding)

    return extract_plain_text(html_str)