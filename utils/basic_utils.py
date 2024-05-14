import streamlit as st
from streamlit import cache_resource


def split_list_by_key_value(dict_list, key, value):
    result = []
    temp_list = []
    count = 0

    for d in dict_list:
        # 检查字典是否有指定的key，并且该key的值是否等于指定的value
        if d.get(key) == value:
            count += 1
            temp_list.append(d)
            # 如果指定值的出现次数为2，则分割列表
            if count == 2:
                result.append(temp_list)
                temp_list = []
                count = 0
        else:
            # 如果当前字典的key的值不是指定的value，则直接添加到当前轮次的列表
            temp_list.append(d)

    # 将剩余的临时列表（如果有）添加到结果列表
    if temp_list:
        result.append(temp_list)

    return result


class Meta(type):
    def __new__(cls, name, bases, attrs):
        for name, value in attrs.items():
            if callable(value) and not name.startswith('__') and not name.startswith('_'):  # 跳过特殊方法和私有方法
                attrs[name] = cache_resource(value)
        return super().__new__(cls, name, bases, attrs)