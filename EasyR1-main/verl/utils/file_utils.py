import json
import os

def read_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            items.append(json.loads(line))
    return items

def write_jsonl(items, path, mode='w'):
    with open(path, mode, encoding='utf8') as fw:
        for item in items:
            fw.write(json.dumps(item, ensure_ascii=False) + '\n')

def read_text(path):
    items = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            items.append(line.rstrip())
    return items


def get_signal(default_local_dir):
    os.makedirs(default_local_dir, exist_ok=True)
    signals = []
    try:
        signals_path = os.path.join(default_local_dir, 'signal.txt')
        if not os.path.exists(signals_path):
            open(signals_path, 'w', encoding='utf8')
        signals = read_text(signals_path)
    except:
        signals = []
    return signals