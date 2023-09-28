import json


def read_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    return data


def save_json(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, save_path, ensure_ascii=False, indent=1)
        f.close()
