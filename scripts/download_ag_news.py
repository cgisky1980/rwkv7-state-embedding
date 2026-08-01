#!/usr/bin/env python3
"""下载 AG News 数据集并转为 jsonl 格式

AG News: 4类新闻 (World=0, Sports=1, Business=2, Tech=3)
  train: 120,000 samples
  test:  7,600 samples

输出:
  data/ag_news/train.jsonl
  data/ag_news/test.jsonl

用法: uv run --project ../../scripts python download_ag_news.py
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ag_news"


def main():
    import csv
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    label_names = ["World", "Sports", "Business", "Tech"]
    urls = {
        "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
        "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
    }

    for split, url in urls.items():
        out = DATA_DIR / f"{split}.jsonl"
        if out.exists():
            print(f"  {split}: 已存在，跳过 ({out})")
            continue
        print(f"下载 {split}: {url}", flush=True)
        tmp = DATA_DIR / f"{split}.csv"
        urllib.request.urlretrieve(url, tmp)
        n = 0
        with open(tmp, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
            reader = csv.reader(fin)
            for row in reader:
                # CSV 格式: class_idx (1-4), title, description
                cls = int(row[0]) - 1  # 转为 0-3
                title = row[1]
                desc = row[2]
                text = f"{title}. {desc}".strip()
                fout.write(json.dumps({
                    "text": text,
                    "label": cls,
                    "label_name": label_names[cls],
                }, ensure_ascii=False) + "\n")
                n += 1
        tmp.unlink()
        print(f"  保存 {n} 条 -> {out}")

    print(f"\n完成。数据位于: {DATA_DIR}")


if __name__ == "__main__":
    main()
