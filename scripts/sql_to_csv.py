import re
import pandas as pd
from tqdm import tqdm
import logging

# تنظیم لاگر
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# الگوی regex برای گرفتن داده از SQL
pattern = re.compile(r"\((\d+), '([^']*)', (\d+), '([^']*)'\)")

def convert(sql_file, output_file):
    rows = []

    logging.info(f"شروع پردازش فایل SQL: {sql_file}")

    # خواندن فایل خط به خط با tqdm
    with open(sql_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading SQL lines", unit="line"):
            matches = pattern.findall(line)
            for m in matches:
                msg_id, message, reply, date = m
                rows.append((int(msg_id), message, int(reply), date))

    logging.info(f"تعداد رکورد استخراج‌شده: {len(rows)}")

    df = pd.DataFrame(rows, columns=["id", "message", "reply", "date"])

    pairs = []
    logging.info("شروع ساخت دیالوگ‌ها...")

    for i in tqdm(range(len(df)-1), desc="Building dialogues", unit="row"):
        if df.iloc[i]["reply"] == 0 and df.iloc[i+1]["reply"] == 1:
            pairs.append((df.iloc[i]["message"], df.iloc[i+1]["message"]))

    dialogues = pd.DataFrame(pairs, columns=["user", "bot"])
    dialogues.to_csv(output_file, index=False, encoding="utf-8-sig")

    logging.info(f"✅ ذخیره شد: {len(dialogues)} دیالوگ در {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input file argument
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="path to input file"
    )

    # output file argument
    parser.add_argument(
        "--output",
        required=True,
        help="path to output file"
    )

    args = parser.parse_args()

    convert(args.file, args.output)