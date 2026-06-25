import html
import pandas as pd
import difflib

def word_to_phone(words_df, phones_df):
    result = {}

    for _, word in words_df.iterrows():
        start = word["tmin"]
        end = word["tmax"]

        matches = phones_df[
            (phones_df["tmin"] >= start) &
            (phones_df["tmax"] <= end)
        ]

        result[word["text"]] = matches["text"].tolist()

    return result


def word_diff_html(word_ref_df: pd.DataFrame, word_hyp_df: pd.DataFrame, phone_ref_df: pd.DataFrame, phone_hyp_df: pd.DataFrame) -> str:

    ref_word_to_phones = word_to_phone(word_ref_df, phone_ref_df)
    hyp_word_to_phones = word_to_phone(word_hyp_df, phone_hyp_df)

    ref_words = word_ref_df[word_ref_df["text"] != "<eps>"]["text"].tolist()
    query_words = word_hyp_df[word_hyp_df["text"] != "<eps>"]["text"].tolist()

    sm = difflib.SequenceMatcher(
        None,
        ref_words,
        query_words,
        autojunk=False
    )

    chunks = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():

        ref_seg = ref_words[i1:i2]
        query_seg = query_words[j1:j2]

        if tag == "equal":

            chunks.extend(
                f'<span class="diff-equal">{html.escape(w)}</span>'
                for w in ref_seg
            )

        elif tag == "replace":

            ref_phones = []
            query_phones = []

            for w in ref_seg:
                ref_phones += ref_word_to_phones[w]

            for w in query_seg:
                query_phones += hyp_word_to_phones[w]

            if ref_phones == query_phones:
                chunks.extend(
                    f'<span class="diff-equal">{html.escape(w)}</span>'
                    for w in ref_seg
                )
            else:
                chunks.append(
                    f'<span class="diff-replace">'
                    f'{" ".join(ref_seg)} → {" ".join(query_seg)}'
                    f'</span>'
                )

        elif tag == "delete":

            chunks.append(
                f'<span class="diff-delete">'
                f'{" ".join(ref_seg)}'
                f'</span>'
            )

        elif tag == "insert":

            chunks.append(
                f'<span class="diff-insert">'
                f'{" ".join(query_seg)}'
                f'</span>'
            )

    return (
        '<div class="diff-container">'
        + " ".join(chunks)
        + "</div>"
    )