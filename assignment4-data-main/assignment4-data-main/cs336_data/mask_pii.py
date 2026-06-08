"""
对个人信息：邮箱、手机、IP 进行掩码
"""

import re
def mask_emails(text: str) -> tuple[str, int]:
    # 正则表达式匹配邮箱格式
    email_pattern = r"[\w.+-]+@[\w-]+\.[\w.-]+"

    """ re.findall() 找到所有匹配 -> 列表长度就是匹配数量 """
    count = len(re.findall(email_pattern, text))

    """ re.sub 把所有匹配替换掉 """
    masked_text = re.sub(email_pattern, "|||EMAIL_ADDRESS|||", text)

    return (masked_text, count)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    # 覆盖测试中的四种格式：
    # 2831823829 | (283)-182-3829 | (283) 182 3829 | 283-182-3829
    phone_pattern = (
        r"\b\d{10}\b"                          # 2831823829
        r"|\(\d{3}\)[ -]?\d{3}[ -]?\d{4}"     # (283)-182-3829 或 (283) 182 3829
        r"|\b\d{3}-\d{3}-\d{4}\b"              # 283-182-3829
    )
    count = len(re.findall(phone_pattern, text))
    masked_text = re.sub(phone_pattern, "|||PHONE_NUMBER|||", text)
    return (masked_text, count)

def mask_ips(text: str) -> tuple[str, int]:
    ip_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    count = len(re.findall(ip_pattern, text))
    masked_text = re.sub(ip_pattern, "|||IP_ADDRESS|||", text)
    return (masked_text, count)

