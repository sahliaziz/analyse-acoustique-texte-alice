from pathlib import Path
import pandas as pd
import tgt


def tier_to_df(textgrid_path : Path, tier_index : int) -> pd.DataFrame:
    tg = tgt.io.read_textgrid(textgrid_path)

    tier = tg.tiers[tier_index]

    if isinstance(tier, tgt.IntervalTier):
        rows = [
            {
                "tmin": round(interval.start_time, 6),
                "text": interval.text,
                "tmax": round(interval.end_time, 6),
            }
            for interval in tier
        ]
    else:
        raise ValueError(f"Unknown TextGrid tier type: {type(tier)}")

    return pd.DataFrame(rows)


def extract_consonants(df : pd.DataFrame) -> pd.DataFrame:
    consonants = ("p", "t", "k", "b", "d", "g", "f", "s", "S", "v", "z", "Z")
    consonants_df = pd.DataFrame(columns=["tmin", "text", "tmax"])
    length = len(df) - 2
    if length >= 658:
        for i in range(2, length):
            tmin, text, tmax = df.iloc[i].values
            text_before = df.iloc[i - 1]["text"]
            text_bebefore = df.iloc[i - 2]["text"]
            text_after = df.iloc[i + 1]["text"]
            text_afterter = df.iloc[i + 2]["text"]
            if text in consonants:
                if (
                    (text_before == "a" and text_after == "a")
                    or (
                        text_bebefore == "a"
                        and text_before == "<p:>"
                        and text_after == "<p:>"
                        and text_afterter == "a"
                    )
                    or (
                        text_bebefore == "a"
                        and text_before == "<p:>"
                        and text_after == "a"
                    )
                    or (
                        text_before == "a"
                        and text_after == "<p:>"
                        and text_afterter == "a"
                    )
                ):
                    new_row = pd.DataFrame(
                        {"tmin": [tmin], "text": [text], "tmax": [tmax]}
                    )
                    consonants_df = pd.concat([consonants_df, new_row], ignore_index=True)
        return consonants_df
    else:
        raise ValueError(f"DataFrame must have at least 660 rows, but has {len(df)}")