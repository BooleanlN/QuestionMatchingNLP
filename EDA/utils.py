import json


def read_file_by_line(file_path, line_num=None,
                      skip_empty_line=True, strip=True,
                      auto_loads_json=True):
    """ 读取一个文件的前 N 行，按列表返回，
    文件中按行组织，要求 utf-8 格式编码的自然语言文本。
    若每行元素为 json 格式可自动加载。

    Args:
        file_path(str): 文件路径
        line_num(int): 读取文件中的行数，若不指定则全部按行读出
        skip_empty_line(boolean): 是否跳过空行
        strip: 将每一行的内容字符串做 strip() 操作
        auto_loads_json(bool): 是否自动将每行使用 json 加载，默认是

    Returns:
        list: line_num 行的内容列表

    Examples:
        >>> file_path = '/path/to/stopwords.txt'
        >>> print(jio.read_file_by_line(file_path, line_num=3))

        # ['在', '然后', '还有']

    """
    content_list = list()
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while True:
            if line == '':  # 整行全空，说明到文件底
                break
            if line_num is not None:
                if count >= line_num:
                    break

            if line.strip() == '':
                if skip_empty_line:
                    count += 1
                    line = f.readline()
                else:
                    try:
                        if auto_loads_json:
                            cur_obj = json.loads(line.strip())
                            content_list.append(cur_obj)
                        else:
                            if strip:
                                content_list.append(line.strip())
                            else:
                                content_list.append(line)
                    except:
                        if strip:
                            content_list.append(line.strip())
                        else:
                            content_list.append(line)

                    count += 1
                    line = f.readline()
                    continue
            else:
                try:
                    if auto_loads_json:
                        cur_obj = json.loads(line.strip())
                        content_list.append(cur_obj)
                    else:
                        if strip:
                            content_list.append(line.strip())
                        else:
                            content_list.append(line)
                except:
                    if strip:
                        content_list.append(line.strip())
                    else:
                        content_list.append(line)

                count += 1
                line = f.readline()
                continue

    return content_list