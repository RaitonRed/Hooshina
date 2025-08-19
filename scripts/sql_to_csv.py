import re
import pandas as pd
import argparse

pattern = re.compile(r"\((\d+), '([^']*)', (\d+), '([^']*)'\)")

def convert(sql_file, output_file):

    rows = []
    with open(sql_file, "r", encoding="utf-8") as f:
        for line in f:
            matches = pattern.findall(line)
            for m in matches:
                msg_id, message, reply, date = m
                rows.append((int(msg_id), message, int(reply), date))

    df = pd.DataFrame(rows, columns=["id", "message", "reply", "date"])

    pairs = []
    for i in range(len(df)-1):
        if df.iloc[i]["reply"] == 0 and df.iloc[i+1]["reply"] == 1:
            pairs.append((df.iloc[i]["message"], df.iloc[i+1]["message"]))

    dialogues = pd.DataFrame(pairs, columns=["user", "bot"])
    dialogues.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved", len(dialogues), "dialogues to telegram_dialogues.csv")

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