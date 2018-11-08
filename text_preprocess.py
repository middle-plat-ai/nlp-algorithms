# coding:utf-8

import re


def clean_text(text):
    """
    清理数据,正则方式,去除标点符号等
    :param text:
    :return:
    """
    text = re.sub(r'["\' ?!【】\[\]./%：:&()=，,<>+_；;\-*]+', " ", text)
    return text
