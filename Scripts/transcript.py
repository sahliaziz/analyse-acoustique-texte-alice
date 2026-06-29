import html
import difflib

import pandas as pd


def words_to_phones(words_df: pd.DataFrame, phones_df: pd.DataFrame) -> list[list[str]]:
    result = []
    for _, word in words_df.iterrows():
        start = word["tmin"]
        end = word["tmax"]

        matches = phones_df[(phones_df["tmin"] >= start) & (phones_df["tmax"] <= end)]

        result.append(matches["text"].tolist())

    return result


def word_diff_html(
    word_ref_df: pd.DataFrame,
    word_hyp_df: pd.DataFrame,
    phone_ref_df: pd.DataFrame,
    phone_hyp_df: pd.DataFrame,
) -> str:
    word_ref_df = word_ref_df[word_ref_df["text"] != "<eps>"].reset_index(drop=True)
    word_hyp_df = word_hyp_df[word_hyp_df["text"] != "<eps>"].reset_index(drop=True)

    ref_word_phones = words_to_phones(word_ref_df, phone_ref_df)
    hyp_word_phones = words_to_phones(word_hyp_df, phone_hyp_df)

    ref_words = word_ref_df["text"].tolist()
    query_words = word_hyp_df["text"].tolist()

    sm = difflib.SequenceMatcher(None, ref_words, query_words, autojunk=False)

    chunks = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ref_seg = ref_words[i1:i2]
        query_seg = query_words[j1:j2]

        if tag == "equal":
            chunks.extend(
                f'<span class="diff-equal">{html.escape(w)}</span>' for w in ref_seg
            )

        elif tag == "replace":
            ref_phones = []
            query_phones = []

            for phones in ref_word_phones[i1:i2]:
                ref_phones += phones

            for phones in hyp_word_phones[j1:j2]:
                query_phones += phones

            if ref_phones == query_phones:
                chunks.extend(
                    f'<span class="diff-equal">{html.escape(w)}</span>' for w in ref_seg
                )
            else:
                chunks.append(
                    f'<span class="diff-replace">'
                    f"{html.escape(' '.join(ref_seg))} -> "
                    f"{html.escape(' '.join(query_seg))}"
                    f"</span>"
                )

        elif tag == "delete":
            chunks.append(
                f'<span class="diff-delete">{html.escape(" ".join(ref_seg))}</span>'
            )

        elif tag == "insert":
            chunks.append(
                f'<span class="diff-insert">{html.escape(" ".join(query_seg))}</span>'
            )

    return '<div class="diff-container">' + " ".join(chunks) + "</div>"
