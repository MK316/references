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
    "and checks whether the reference title matches the Scholar record."
)


# ============================================================
# Default examples
# ============================================================
DEFAULT_REFERENCES = [
    "Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
    "Goodfellow, I., et al. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems, 27.",
    "Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. Qualitative Research in Psychology, 3(2), 77–101.",
]


# ============================================================
# Normalize title
# ============================================================
def normalize_title(text: str) -> str:
    """
    Normalize title strings before similarity comparison.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    # Normalize punctuation variants
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# Extract title from reference
# ============================================================
def extract_title_from_reference(reference: str) -> str:
    """
    Extract title from an APA-like reference.

    Example:

    Braun, V., & Clarke, V. (2006).
    Using thematic analysis in psychology.
    Qualitative Research in Psychology, 3(2), 77–101.

    -> Using thematic analysis in psychology

    If no publication year is detected,
    the whole input is treated as a title.
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

    remainder = reference[
        match.end():
    ].strip()

    # Remove punctuation immediately after year
    remainder = re.sub(
        r"^[\.\s]+",
        "",
        remainder
    )

    if not remainder:
        return reference.strip()

    # Protect common abbreviations from period splitting
    protected = remainder

    abbreviations = [
        "e.g.",
        "i.e.",
        "et al.",
        "U.S.",
        "U.K.",
        "Ph.D.",
        "Ed.D.",
        "No.",
        "Vol.",
    ]

    placeholders = {}

    for i, abbreviation in enumerate(abbreviations):

        placeholder = f"__ABBR_{i}__"

        pattern = re.compile(
            re.escape(abbreviation),
            flags=re.IGNORECASE
        )

        match_abbr = pattern.search(protected)

        if match_abbr:

            original = match_abbr.group()

            placeholders[
                placeholder
            ] = original

            protected = pattern.sub(
                placeholder,
                protected
            )

    # First sentence after year is treated as title
    parts = re.split(
        r"\.\s+",
        protected,
        maxsplit=1
    )

    title = parts[0].strip()

    # Restore protected abbreviations
    for placeholder, original in placeholders.items():

        title = title.replace(
            placeholder,
            original
        )

    title = title.strip(" .")

    return title


# ============================================================
# Title similarity
# ============================================================
def title_similarity(
    title1: str,
    title2: str
) -> float:
    """
    Levenshtein ratio after normalization.
    Returns similarity percentage 0-100.
    """

    t1 = normalize_title(title1)
    t2 = normalize_title(title2)

    if not t1 or not t2:
        return 0.0

    score = ratio(
        t1,
        t2
    ) * 100

    return round(
        score,
        2
    )


# ============================================================
# Match classification
# ============================================================
def classify_match(
    similarity: float,
    match_threshold: int,
    mismatch_threshold: int
) -> str:

    if similarity >= match_threshold:
        return "✅ Match"

    elif similarity >= mismatch_threshold:
        return "⚠️ Possible mismatch"

    else:
        return "❌ Mismatch"


# ============================================================
# DOI helpers
# ============================================================
def clean_doi(
    doi: str | None
) -> str | None:
    """
    Clean DOI string.
    """

    if not doi:
        return None

    doi = str(
        doi
    ).strip()

    # Remove doi: prefix
    doi = re.sub(
        r"^doi:\s*",
        "",
        doi,
        flags=re.IGNORECASE
    )

    # Remove DOI URL prefix
    doi = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        doi,
        flags=re.IGNORECASE
    )

    # Remove trailing punctuation
    doi = doi.rstrip(
        ".,;)"
    )

    return doi or None


def extract_doi_from_text(
    text: str | None
) -> str | None:
    """
    Extract DOI pattern from arbitrary text or URL.
    """

    if not text:
        return None

    doi_pattern = (
        r"10\.\d{4,9}/"
        r"[-._;()/:A-Z0-9]+"
    )

    match = re.search(
        doi_pattern,
        str(text),
        flags=re.IGNORECASE
    )

    if not match:
        return None

    return clean_doi(
        match.group(0)
    )


def extract_doi_from_result(
    result: dict
) -> str | None:
    """
    Attempt DOI extraction from a scholarly search result.
    """

    bib = (
        result.get(
            "bib",
            {}
        )
        or {}
    )

    # --------------------------------------------------------
    # 1. Direct DOI field
    # --------------------------------------------------------
    doi = bib.get(
        "doi"
    )

    doi = clean_doi(
        doi
    )

    if doi:
        return doi

    # --------------------------------------------------------
    # 2. Other possible fields in bib
    # --------------------------------------------------------
    possible_bib_fields = [
        "url",
        "pub_url",
        "eprint",
        "citation",
    ]

    for field in possible_bib_fields:

        doi = extract_doi_from_text(
            bib.get(
                field
            )
        )

        if doi:
            return doi

    # --------------------------------------------------------
    # 3. Top-level result URLs
    # --------------------------------------------------------
    possible_result_fields = [
        "pub_url",
        "eprint_url",
    ]

    for field in possible_result_fields:

        doi = extract_doi_from_text(
            result.get(
                field
            )
        )

        if doi:
            return doi

    return None


# ============================================================
# Optional proxy support
# ============================================================
def configure_proxy_if_available() -> bool:

    try:

        secrets = st.secrets

        if (
            "proxy_host" not in secrets
            or
            "proxy_port" not in secrets
        ):
            return False

        host = secrets[
            "proxy_host"
        ]

        port = int(
            secrets[
                "proxy_port"
            ]
        )

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

            scholarly.use_proxy(
                pg
            )

            return True

        return False

    except Exception:

        return False


# ============================================================
# Google Scholar reference check
# ============================================================
def check_reference_validity(
    original_reference: str,
    title_to_check: str,
    match_threshold: int,
    mismatch_threshold: int,
    max_candidates: int = 5,
):
    """
    Search Google Scholar using extracted title.

    Multiple Scholar results are inspected.
    The result with the highest title similarity is selected.
    """

    try:

        search_query = scholarly.search_pubs(
            title_to_check
        )

        candidates = []

        # ----------------------------------------------------
        # Check several results instead of only first result
        # ----------------------------------------------------
        for _ in range(
            max_candidates
        ):

            try:

                result = next(
                    search_query
                )

            except StopIteration:

                break

            bib = (
                result.get(
                    "bib",
                    {}
                )
                or {}
            )

            found_title = bib.get(
                "title",
                ""
            )

            if not found_title:
                continue

            similarity = title_similarity(
                title_to_check,
                found_title
            )

            candidates.append(
                {
                    "result": result,
                    "found_title": found_title,
                    "similarity": similarity,
                }
            )

        # ----------------------------------------------------
        # No Scholar result
        # ----------------------------------------------------
        if not candidates:

            return {
                "status": "Not Found",
                "match_status": "❓ Not Found",
                "original_reference": original_reference,
                "extracted_title": title_to_check,
                "found_title": None,
                "similarity": 0.0,
                "doi": None,
                "doi_url": None,
                "url": None,
                "error": None,
            }

        # ----------------------------------------------------
        # Select best title match
        # ----------------------------------------------------
        best = max(
            candidates,
            key=lambda x: x[
                "similarity"
            ]
        )

        best_result = best[
            "result"
        ]

        best_title = best[
            "found_title"
        ]

        best_similarity = best[
            "similarity"
        ]

        match_status = classify_match(
            best_similarity,
            match_threshold,
            mismatch_threshold
        )

        # ----------------------------------------------------
        # URL
        # ----------------------------------------------------
        url = (
            best_result.get(
                "pub_url"
            )
            or
            best_result.get(
                "eprint_url"
            )
            or
            None
        )

        # ----------------------------------------------------
        # DOI
        # ----------------------------------------------------
        doi = extract_doi_from_result(
            best_result
        )

        doi_url = (
            f"https://doi.org/{doi}"
            if doi
            else None
        )

        return {
            "status": "Found",
            "match_status": match_status,
            "original_reference": original_reference,
            "extracted_title": title_to_check,
            "found_title": best_title,
            "similarity": best_similarity,
            "doi": doi,
            "doi_url": doi_url,
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
            "doi": None,
            "doi_url": None,
            "url": None,
            "error": str(e),
        }


# ============================================================
# Sidebar
# ============================================================
st.sidebar.header(
    "⚙️ Settings"
)


# ------------------------------------------------------------
# Matching settings
# ------------------------------------------------------------
st.sidebar.subheader(
    "Title matching"
)

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
# Request delay
# ------------------------------------------------------------
st.sidebar.subheader(
    "Scholar requests"
)

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
        (
            "✅ enabled"
            if ok
            else
            "❌ not configured / failed"
        )
    )


# ============================================================
# Reference input
# ============================================================
st.subheader(
    "📚 References to verify"
)

st.write(
    "Enter **one complete reference per line**. "
    "The reference title will be extracted automatically."
)

raw_text = st.text_area(
    "References",
    value="\n".join(
        DEFAULT_REFERENCES
    ),
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

    extracted = (
        extract_title_from_reference(
            ref
        )
    )

    preview_data.append(
        {
            "Original Reference": ref,
            "Extracted Title": extracted,
        }
    )

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
# Run controls
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
        f"Total references: "
        f"**{len(references)}**"
    )


# ============================================================
# Run verification
# ============================================================
if run:

    results = []

    progress = st.progress(
        0
    )

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
                f"🔍 Searching: "
                f"**{extracted_title}**"
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

                # --------------------------------------------
                # Retry on error
                # --------------------------------------------
                if (
                    result["status"]
                    == "Error"
                    and
                    attempt <= max_retries
                ):

                    st.write(
                        f"⚠️ Error "
                        f"(attempt "
                        f"{attempt}/"
                        f"{max_retries}). "
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
            # Individual result display
            # =================================================
            if (
                result["status"]
                == "Found"
            ):

                result_message = (
                    f"{result['match_status']} | "
                    f"Matching Rate: "
                    f"{result['similarity']}%"
                )

                if (
                    result["match_status"]
                    == "✅ Match"
                ):

                    st.success(
                        result_message
                    )

                elif (
                    result["match_status"]
                    == "⚠️ Possible mismatch"
                ):

                    st.warning(
                        result_message
                    )

                else:

                    st.error(
                        result_message
                    )

                st.write(
                    "**Reference title:** "
                    f"{result['extracted_title']}"
                )

                st.write(
                    "**Scholar title:** "
                    f"{result['found_title']}"
                )

                if result["doi"]:

                    st.write(
                        "**DOI:** "
                        f"{result['doi']}"
                    )

                if result["doi_url"]:

                    st.write(
                        "**DOI URL:** "
                        f"{result['doi_url']}"
                    )

                if result["url"]:

                    st.write(
                        "**Source URL:** "
                        f"{result['url']}"
                    )


            elif (
                result["status"]
                == "Not Found"
            ):

                st.warning(
                    "❓ Not Found: "
                    f"{result['extracted_title']}"
                )


            else:

                st.error(
                    "🚫 Error: "
                    f"{result.get('error', 'Unknown error')}"
                )


            # =================================================
            # Delay between Scholar requests
            # =================================================
            if i < len(
                references
            ):

                time.sleep(
                    random.uniform(
                        min_delay,
                        max_delay
                    )
                )

            progress.progress(
                i / len(
                    references
                )
            )


        status.update(
            label="Done",
            state="complete",
            expanded=False
        )


    # ========================================================
    # Results dataframe
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
            "doi": "DOI",
            "doi_url": "DOI URL",
            "url": "URL",
            "original_reference": "Original Reference",
            "status": "Search Status",
            "error": "Error",
        }
    )


    # ========================================================
    # Column order
    # ========================================================
    preferred_columns = [
        "Match Status",
        "Matching Rate (%)",
        "Reference Title",
        "Scholar Title",
        "DOI",
        "DOI URL",
        "URL",
        "Original Reference",
        "Search Status",
        "Error",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    df = df[
        available_columns
    ]


    # ========================================================
    # All results
    # ========================================================
    st.subheader(
        "📊 All Results"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Matching Rate (%)":
                st.column_config.NumberColumn(
                    "Matching Rate (%)",
                    format="%.2f"
                ),

            "DOI URL":
                st.column_config.LinkColumn(
                    "DOI URL"
                ),

            "URL":
                st.column_config.LinkColumn(
                    "URL"
                ),
        }
    )


    # ========================================================
    # Review items
    # ========================================================
    st.subheader(
        "🚨 Titles requiring review"
    )

    review_df = df[
        df[
            "Match Status"
        ].isin(
            [
                "⚠️ Possible mismatch",
                "❌ Mismatch",
                "❓ Not Found",
                "🚫 Error",
            ]
        )
    ].copy()


    if len(
        review_df
    ) == 0:

        st.success(
            "No title mismatches were detected."
        )

    else:

        st.dataframe(
            review_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Matching Rate (%)":
                    st.column_config.NumberColumn(
                        "Matching Rate (%)",
                        format="%.2f"
                    ),

                "DOI URL":
                    st.column_config.LinkColumn(
                        "DOI URL"
                    ),

                "URL":
                    st.column_config.LinkColumn(
                        "URL"
                    ),
            }
        )

        st.info(
            f"{len(review_df)} of "
            f"{len(df)} references "
            f"require review."
        )


    # ========================================================
    # Summary
    # ========================================================
    st.subheader(
        "📌 Summary"
    )

    counts = (
        df[
            "Match Status"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
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
        file_name=(
            "reference_validation_results.csv"
        ),
        mime="text/csv",
    )


    # ========================================================
    # Download review items
    # ========================================================
    if len(
        review_df
    ) > 0:

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
            file_name=(
                "reference_title_mismatches.csv"
            ),
            mime="text/csv",
        )
