import time
import random
import re
import unicodedata

import pandas as pd
import streamlit as st

from scholarly import scholarly, ProxyGenerator
from Levenshtein import ratio


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Reference Validator (Google Scholar)",
    layout="wide"
)

st.title("📚 Reference Validator")
st.caption(
    "Paste one reference per line. "
    "The app extracts the title, searches Google Scholar, "
    "and checks whether the titles match."
)


# ============================================================
# Defaults
# ============================================================
DEFAULT_REFERENCES = [
    "Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
    "Goodfellow, I., et al. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems, 27.",
    "Smith, J. (2020). A non-existent paper about flying spaghettis in deep learning. Journal of Imaginary Research, 10(2), 1–10.",
]


# ============================================================
# Text normalization
# ============================================================
def normalize_title(text: str) -> str:
    """
    Normalize titles before comparison.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Extract title from reference
# ============================================================
def extract_title_from_reference(reference: str) -> str:
    """
    Extract the title from an APA-like reference.

    Example:
    Braun, V., & Clarke, V. (2006). Using thematic analysis
    in psychology. Qualitative Research in Psychology, 3(2), ...

    -> Using thematic analysis in psychology

    If no year pattern is found, the entire line is treated as a title.
    """

    reference = reference.strip()

    if not reference:
        return ""

    # Supports:
    # (2006)
    # (2018a)
    # (2021b)
    # (n.d.)
    year_pattern = r"\((?:\d{4}[a-z]?|n\.d\.)\)"

    match = re.search(
        year_pattern,
        reference,
        flags=re.IGNORECASE
    )

    if not match:
        return reference.strip(" .")

    # Everything after publication year
    remainder = reference[match.end():].strip()

    # Remove leading punctuation
    remainder = re.sub(
        r"^[\.\s]+",
        "",
        remainder
    )

    if not remainder:
        return reference

    # Protect some common abbreviations
    protected = remainder

    abbreviations = [
        "e.g.",
        "i.e.",
        "et al.",
        "U.S.",
        "U.K.",
        "Ph.D.",
        "Ed.D.",
    ]

    placeholders = {}

    for i, abbreviation in enumerate(abbreviations):

        placeholder = f"__ABBR{i}__"

        pattern = re.compile(
            re.escape(abbreviation),
            flags=re.IGNORECASE
        )

        found = pattern.search(protected)

        if found:
            original = found.group()
            placeholders[placeholder] = original
            protected = pattern.sub(
                placeholder,
                protected
            )

    # The first sentence after the year is treated as the title
    parts = re.split(
        r"\.\s+",
        protected,
        maxsplit=1
    )

    title = parts[0].strip()

    # Restore abbreviations
    for placeholder, original in placeholders.items():
        title = title.replace(
            placeholder,
            original
        )

    title = title.strip(" .")

    return title


# ============================================================
# Similarity
# ============================================================
def title_similarity(title1: str, title2: str) -> float:
    """
    Calculate Levenshtein similarity after normalization.
    Returns 0-100.
    """

    t1 = normalize_title(title1)
    t2 = normalize_title(title2)

    if not t1 or not t2:
        return 0.0

    score = ratio(t1, t2) * 100

    return round(score, 2)


# ============================================================
# Match classification
# ============================================================
def classify_match(
    similarity: float,
    match_threshold: int,
    mismatch_threshold: int
):
    """
    Default:
    >= 90 : Match
    75-89 : Possible mismatch
    < 75  : Mismatch
    """

    if similarity >= match_threshold:
        return "✅ Match"

    elif similarity >= mismatch_threshold:
        return "⚠️ Possible mismatch"

    else:
        return "❌ Mismatch"


# ============================================================
# Optional proxy support
# ============================================================
def configure_proxy_if_available() -> bool:

    try:
        secrets = st.secrets

        if (
            "proxy_host" not in secrets
            or "proxy_port" not in secrets
        ):
            return False

        host = secrets["proxy_host"]
        port = int(secrets["proxy_port"])

        user = secrets.get(
            "proxy_user",
            None
        )

        pw = secrets.get(
            "proxy_pass",
            None
        )

        pg = ProxyGenerator()

        ok = pg.SingleProxy(
            http=f"socks5://{host}:{port}",
            https=f"socks5://{host}:{port}",
            user=user,
            password=pw,
        )

        if ok:
            scholarly.use_proxy(pg)
            return True

        return False

    except Exception:
        return False


# ============================================================
# Google Scholar check
# ============================================================
def check_reference_validity(
    original_reference: str,
    title_to_check: str,
    match_threshold: int,
    mismatch_threshold: int,
    max_candidates: int = 5,
):
    """
    Search Google Scholar using the extracted title.

    Several search results are checked and the result with
    the highest title similarity is selected.
    """

    try:

        search_query = scholarly.search_pubs(
            title_to_check
        )

        candidates = []

        for _ in range(max_candidates):

            try:
                result = next(search_query)

            except StopIteration:
                break

            found_title = (
                result
                .get("bib", {})
                .get("title", "")
            )

            if not found_title:
                continue

            similarity = title_similarity(
                title_to_check,
                found_title
            )

            candidates.append({
                "result": result,
                "found_title": found_title,
                "similarity": similarity,
            })

        # No Google Scholar result
        if not candidates:

            return {
                "status": "Not Found",
                "match_status": "❓ Not Found",
                "original_reference": original_reference,
                "extracted_title": title_to_check,
                "found_title": None,
                "similarity": 0.0,
                "url": None,
                "error": None,
            }

        # Best matching Scholar result
        best = max(
            candidates,
            key=lambda x: x["similarity"]
        )

        best_result = best["result"]
        best_title = best["found_title"]
        best_similarity = best["similarity"]

        match_status = classify_match(
            best_similarity,
            match_threshold,
            mismatch_threshold
        )

        url = (
            best_result.get("pub_url")
            or best_result.get("eprint_url")
            or "N/A"
        )

        return {
            "status": "Found",
            "match_status": match_status,
            "original_reference": original_reference,
            "extracted_title": title_to_check,
            "found_title": best_title,
            "similarity": best_similarity,
            "url": url,
            "error": None,
        }

    except Exception as e:

        return {
            "status": "Error",
            "match_status": "🚫 Error",
            "original_reference": original_reference,
            "extracted_title": title_to_check,
            "found_title": None,
            "similarity": 0.0,
            "url": None,
            "error": str(e),
        }


# ============================================================
# Sidebar settings
# ============================================================
st.sidebar.header("⚙️ Settings")


# ------------------------------------------------------------
# Match thresholds
# ------------------------------------------------------------
st.sidebar.subheader("Title matching")

match_threshold = st.sidebar.slider(
    "Match threshold (%)",
    min_value=80,
    max_value=100,
    value=90,
    step=1,
)

mismatch_threshold = st.sidebar.slider(
    "Possible mismatch threshold (%)",
    min_value=50,
    max_value=89,
    value=75,
    step=1,
)

max_candidates = st.sidebar.slider(
    "Scholar results to compare",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)


# ------------------------------------------------------------
# Delay settings
# ------------------------------------------------------------
st.sidebar.subheader("Scholar requests")

min_delay = st.sidebar.slider(
    "Minimum delay (seconds)",
    2,
    15,
    5,
    1,
)

max_delay = st.sidebar.slider(
    "Maximum delay (seconds)",
    min_delay,
    25,
    10,
    1,
)

max_retries = st.sidebar.slider(
    "Retries on Error",
    0,
    3,
    1,
    1,
)


# ------------------------------------------------------------
# Proxy
# ------------------------------------------------------------
use_proxy = st.sidebar.checkbox(
    "Try proxy from secrets.toml",
    value=False
)

if use_proxy:

    ok = configure_proxy_if_available()

    st.sidebar.write(
        "Proxy status:",
        "✅ enabled"
        if ok
        else "❌ not configured / failed"
    )


# ============================================================
# Input references
# ============================================================
st.subheader("📚 References to verify")

st.write(
    "Enter **one complete reference per line**. "
    "The title will be extracted automatically."
)

raw_text = st.text_area(
    "References",
    value="\n".join(DEFAULT_REFERENCES),
    height=280,
)

references = [
    r.strip()
    for r in raw_text.splitlines()
    if r.strip()
]

if not references:
    st.warning(
        "Please enter at least one reference."
    )
    st.stop()


# ============================================================
# Preview extracted titles
# ============================================================
preview_data = []

for ref in references:

    extracted = extract_title_from_reference(
        ref
    )

    preview_data.append({
        "Original Reference": ref,
        "Extracted Title": extracted,
    })

preview_df = pd.DataFrame(
    preview_data
)

with st.expander(
    "🔎 Preview extracted titles",
    expanded=True
):

    st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# Run button
# ============================================================
colA, colB = st.columns(
    [1, 2]
)

with colA:

    run = st.button(
        "🔍 Run verification",
        type="primary"
    )

with colB:

    st.write(
        f"Total references: **{len(references)}**"
    )


# ============================================================
# Run verification
# ============================================================
if run:

    results = []

    progress = st.progress(0)

    log_box = st.empty()

    with st.status(
        "Running Google Scholar checks...",
        expanded=True
    ) as status:

        for i, reference in enumerate(
            references,
            start=1
        ):

            extracted_title = (
                extract_title_from_reference(
                    reference
                )
            )

            log_box.write(
                f"🔍 Searching: **{extracted_title}**"
            )

            attempt = 0
            result = None

            while True:

                attempt += 1

                result = check_reference_validity(
                    original_reference=reference,
                    title_to_check=extracted_title,
                    match_threshold=match_threshold,
                    mismatch_threshold=mismatch_threshold,
                    max_candidates=max_candidates,
                )

                if (
                    result["status"] == "Error"
                    and attempt <= max_retries
                ):

                    st.write(
                        f"⚠️ Error "
                        f"(attempt {attempt}/{max_retries}). "
                        f"Retrying..."
                    )

                    time.sleep(
                        random.uniform(
                            min_delay,
                            max_delay
                        )
                    )

                    continue

                break

            results.append(
                result
            )


            # =================================================
            # Display individual result
            # =================================================
            if result["status"] == "Found":

                message = (
                    f"{result['match_status']} | "
                    f"Matching Rate: "
                    f"{result['similarity']}%"
                )

                if result["match_status"] == "✅ Match":

                    st.success(
                        message
                    )

                elif (
                    result["match_status"]
                    == "⚠️ Possible mismatch"
                ):

                    st.warning(
                        message
                    )

                else:

                    st.error(
                        message
                    )

                st.write(
                    f"**Reference title:** "
                    f"{result['extracted_title']}"
                )

                st.write(
                    f"**Scholar title:** "
                    f"{result['found_title']}"
                )

                if (
                    result["url"]
                    and result["url"] != "N/A"
                ):

                    st.write(
                        f"**URL:** "
                        f"{result['url']}"
                    )


            elif result["status"] == "Not Found":

                st.warning(
                    f"❓ Not Found: "
                    f"{result['extracted_title']}"
                )


            else:

                st.error(
                    f"🚫 Error: "
                    f"{result.get('error', 'Unknown error')}"
                )


            # Delay between requests
            if i < len(references):

                time.sleep(
                    random.uniform(
                        min_delay,
                        max_delay
                    )
                )

            progress.progress(
                i / len(references)
            )


        status.update(
            label="Done",
            state="complete",
            expanded=False
        )


    # ========================================================
    # Create results dataframe
    # ========================================================
    df = pd.DataFrame(
        results
    )


    # ========================================================
    # Rename columns
    # ========================================================
    df = df.rename(
        columns={
            "match_status": "Match Status",
            "similarity": "Matching Rate (%)",
            "extracted_title": "Reference Title",
            "found_title": "Scholar Title",
            "original_reference": "Original Reference",
            "status": "Search Status",
            "url": "URL",
            "error": "Error",
        }
    )


    # ========================================================
    # Preferred column order
    # ========================================================
    preferred_columns = [
        "Match Status",
        "Matching Rate (%)",
        "Reference Title",
        "Scholar Title",
        "Original Reference",
        "Search Status",
        "URL",
        "Error",
    ]

    available_columns = [
        c
        for c in preferred_columns
        if c in df.columns
    ]

    df = df[
        available_columns
    ]


    # ========================================================
    # Show all results
    # ========================================================
    st.subheader(
        "📊 All Results"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # Titles requiring review
    # ========================================================
    st.subheader(
        "🚨 Titles requiring review"
    )

    review_df = df[
        df["Match Status"].isin([
            "⚠️ Possible mismatch",
            "❌ Mismatch",
            "❓ Not Found",
            "🚫 Error",
        ])
    ].copy()


    if len(review_df) == 0:

        st.success(
            "No title mismatches were detected."
        )

    else:

        st.dataframe(
            review_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            f"{len(review_df)} of "
            f"{len(df)} references require review."
        )


    # ========================================================
    # Summary
    # ========================================================
    st.subheader(
        "📌 Summary"
    )

    counts = (
        df["Match Status"]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    col1.metric(
        "Match",
        counts.get(
            "✅ Match",
            0
        )
    )

    col2.metric(
        "Possible mismatch",
        counts.get(
            "⚠️ Possible mismatch",
            0
        )
    )

    col3.metric(
        "Mismatch",
        counts.get(
            "❌ Mismatch",
            0
        )
    )

    col4.metric(
        "Not found / Error",
        (
            counts.get(
                "❓ Not Found",
                0
            )
            +
            counts.get(
                "🚫 Error",
                0
            )
        )
    )


    # ========================================================
    # Download all results
    # ========================================================
    csv_bytes = (
        df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8-sig"
        )
    )

    st.download_button(
        "⬇️ Download all results",
        data=csv_bytes,
        file_name="reference_validation_results.csv",
        mime="text/csv",
    )


    # ========================================================
    # Download review items
    # ========================================================
    if len(review_df) > 0:

        review_csv = (
            review_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8-sig"
            )
        )

        st.download_button(
            "⬇️ Download items requiring review",
            data=review_csv,
            file_name="reference_title_mismatches.csv",
            mime="text/csv",
        )
