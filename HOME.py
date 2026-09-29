import re
import time
import unicodedata

import pandas as pd
import requests
import streamlit as st

from Levenshtein import ratio


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Reference Validator (Crossref)",
    layout="wide"
)

st.title("📚 Reference Validator")
st.caption(
    "Paste one complete reference per line. "
    "The app extracts the title, searches Crossref, "
    "and compares the reference title with registered metadata."
)


# ============================================================
# Crossref settings
# ============================================================
CROSSREF_API = "https://api.crossref.org/works"

# Change this to your actual contact email if possible.
# Crossref recommends supplying a mailto address.
CONTACT_EMAIL = "your_email@example.com"

USER_AGENT = (
    "ReferenceValidator/1.0 "
    f"(mailto:{CONTACT_EMAIL})"
)


# ============================================================
# Default examples
# ============================================================
DEFAULT_REFERENCES = [
    "Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. Qualitative Research in Psychology, 3(2), 77–101.",
    "Munro, M. J., & Derwing, T. M. (2006). The functional load principle in ESL pronunciation instruction: An exploratory study. System, 34(4), 520–531.",
    "Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
]


# ============================================================
# Normalize title
# ============================================================
def normalize_title(text):
    """
    Normalize title before comparison.
    """

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    text = text.lower()

    # Normalize punctuation variants
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# Extract title from APA-like reference
# ============================================================
def extract_title_from_reference(reference):
    """
    Extract title from APA-like reference.

    Example:

    Braun, V., & Clarke, V. (2006).
    Using thematic analysis in psychology.
    Qualitative Research in Psychology, 3(2), 77–101.

    -> Using thematic analysis in psychology
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

    year_match = re.search(
        year_pattern,
        reference,
        flags=re.IGNORECASE
    )

    # If no year exists, treat whole line as title
    if not year_match:
        return reference.strip(" .")

    remainder = reference[
        year_match.end():
    ].strip()

    # Remove initial period after year
    remainder = re.sub(
        r"^[\.\s]+",
        "",
        remainder
    )

    if not remainder:
        return ""

    # Protect abbreviations
    abbreviations = [
        "e.g.",
        "i.e.",
        "U.S.",
        "U.K.",
        "Ph.D.",
        "Ed.D.",
        "No.",
        "Vol.",
    ]

    protected = remainder
    placeholders = {}

    for i, abbreviation in enumerate(
        abbreviations
    ):

        placeholder = f"__ABBR{i}__"

        pattern = re.compile(
            re.escape(abbreviation),
            flags=re.IGNORECASE
        )

        match = pattern.search(
            protected
        )

        if match:

            original = match.group()

            placeholders[
                placeholder
            ] = original

            protected = pattern.sub(
                placeholder,
                protected
            )

    # First sentence after year
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

    return title.strip(
        " ."
    )


# ============================================================
# Extract year from original reference
# ============================================================
def extract_reference_year(reference):
    """
    Extract a 4-digit publication year from reference.
    """

    match = re.search(
        r"\((\d{4})[a-z]?\)",
        reference
    )

    if match:
        return int(
            match.group(1)
        )

    return None


# ============================================================
# Title similarity
# ============================================================
def title_similarity(
    title1,
    title2
):
    """
    Levenshtein similarity percentage.
    """

    t1 = normalize_title(
        title1
    )

    t2 = normalize_title(
        title2
    )

    if not t1 or not t2:
        return 0.0

    return round(
        ratio(
            t1,
            t2
        ) * 100,
        2
    )


# ============================================================
# Match classification
# ============================================================
def classify_match(
    similarity,
    match_threshold,
    possible_threshold
):

    if similarity >= match_threshold:
        return "✅ Match"

    if similarity >= possible_threshold:
        return "⚠️ Possible mismatch"

    return "❌ Mismatch"


# ============================================================
# Crossref helper: get first value
# ============================================================
def first_value(value):
    """
    Crossref often stores values as lists.
    """

    if isinstance(
        value,
        list
    ):

        if len(value) > 0:
            return value[0]

        return None

    return value


# ============================================================
# Crossref year
# ============================================================
def get_crossref_year(item):
    """
    Retrieve publication year from Crossref record.
    """

    date_fields = [
        "published-print",
        "published-online",
        "published",
        "issued",
        "created",
    ]

    for field in date_fields:

        data = item.get(
            field
        )

        if not data:
            continue

        date_parts = data.get(
            "date-parts",
            []
        )

        if (
            date_parts
            and
            date_parts[0]
        ):

            try:
                return int(
                    date_parts[0][0]
                )

            except Exception:
                pass

    return None


# ============================================================
# Crossref authors
# ============================================================
def get_crossref_authors(item):
    """
    Convert Crossref author metadata to readable string.
    """

    authors = item.get(
        "author",
        []
    )

    names = []

    for author in authors:

        given = (
            author.get(
                "given",
                ""
            )
            or ""
        )

        family = (
            author.get(
                "family",
                ""
            )
            or ""
        )

        name = (
            f"{given} {family}"
        ).strip()

        if name:
            names.append(
                name
            )

    return "; ".join(
        names
    )


# ============================================================
# Extract DOI from original reference
# ============================================================
def extract_reference_doi(reference):
    """
    Detect DOI already present in original reference.
    """

    if not reference:
        return None

    doi_pattern = (
        r"10\.\d{4,9}/"
        r"[-._;()/:A-Z0-9]+"
    )

    match = re.search(
        doi_pattern,
        reference,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    doi = match.group(0)

    return doi.rstrip(
        ".,;)"
    )


# ============================================================
# Crossref request
# ============================================================
def search_crossref(
    reference,
    title,
    rows=5,
    timeout=15
):
    """
    Search Crossref using query.bibliographic.

    Whole reference is used for candidate retrieval,
    while extracted title is used for final matching.
    """

    params = {
        "query.bibliographic": reference,
        "rows": rows,
        "mailto": CONTACT_EMAIL,
    }

    headers = {
        "User-Agent": USER_AGENT
    }

    response = requests.get(
        CROSSREF_API,
        params=params,
        headers=headers,
        timeout=timeout
    )

    response.raise_for_status()

    data = response.json()

    return (
        data
        .get(
            "message",
            {}
        )
        .get(
            "items",
            []
        )
    )


# ============================================================
# Check one reference
# ============================================================
def validate_reference(
    reference,
    match_threshold,
    possible_threshold,
    max_candidates
):
    """
    Validate one reference against Crossref.
    """

    extracted_title = (
        extract_title_from_reference(
            reference
        )
    )

    reference_year = (
        extract_reference_year(
            reference
        )
    )

    reference_doi = (
        extract_reference_doi(
            reference
        )
    )

    try:

        candidates = search_crossref(
            reference=reference,
            title=extracted_title,
            rows=max_candidates
        )

        if not candidates:

            return {
                "Match Status": "❓ Not Found",
                "Matching Rate (%)": 0.0,
                "Reference Title": extracted_title,
                "Crossref Title": None,
                "Reference Year": reference_year,
                "Crossref Year": None,
                "Year Match": None,
                "Reference DOI": reference_doi,
                "Crossref DOI": None,
                "DOI Match": None,
                "DOI URL": None,
                "URL": None,
                "Publisher": None,
                "Journal / Container": None,
                "Authors": None,
                "Original Reference": reference,
                "Search Status": "Not Found",
                "Error": None,
            }

        evaluated = []

        for item in candidates:

            crossref_title = first_value(
                item.get(
                    "title"
                )
            )

            if not crossref_title:
                continue

            similarity = title_similarity(
                extracted_title,
                crossref_title
            )

            crossref_year = (
                get_crossref_year(
                    item
                )
            )

            # Small tie-breaking bonus for matching year
            year_bonus = 0

            if (
                reference_year
                and
                crossref_year
                and
                reference_year == crossref_year
            ):
                year_bonus = 2

            selection_score = (
                similarity
                +
                year_bonus
            )

            evaluated.append(
                {
                    "item": item,
                    "title": crossref_title,
                    "similarity": similarity,
                    "selection_score": selection_score,
                    "year": crossref_year,
                }
            )

        if not evaluated:

            return {
                "Match Status": "❓ Not Found",
                "Matching Rate (%)": 0.0,
                "Reference Title": extracted_title,
                "Crossref Title": None,
                "Reference Year": reference_year,
                "Crossref Year": None,
                "Year Match": None,
                "Reference DOI": reference_doi,
                "Crossref DOI": None,
                "DOI Match": None,
                "DOI URL": None,
                "URL": None,
                "Publisher": None,
                "Journal / Container": None,
                "Authors": None,
                "Original Reference": reference,
                "Search Status": "Not Found",
                "Error": None,
            }

        # Select best candidate
        best = max(
            evaluated,
            key=lambda x:
                x["selection_score"]
        )

        item = best[
            "item"
        ]

        crossref_title = best[
            "title"
        ]

        similarity = best[
            "similarity"
        ]

        crossref_year = best[
            "year"
        ]

        # ----------------------------------------------------
        # DOI
        # ----------------------------------------------------
        crossref_doi = item.get(
            "DOI"
        )

        if crossref_doi:
            crossref_doi = str(
                crossref_doi
            ).strip()

        doi_url = (
            f"https://doi.org/{crossref_doi}"
            if crossref_doi
            else None
        )

        # ----------------------------------------------------
        # Publisher URL
        # ----------------------------------------------------
        url = item.get(
            "URL"
        )

        # If Crossref URL is absent, DOI URL is still useful
        if not url and doi_url:
            url = doi_url

        # ----------------------------------------------------
        # Year match
        # ----------------------------------------------------
        if (
            reference_year is not None
            and
            crossref_year is not None
        ):

            year_match = (
                "Yes"
                if reference_year == crossref_year
                else "No"
            )

        else:
            year_match = None

        # ----------------------------------------------------
        # DOI match
        # ----------------------------------------------------
        if (
            reference_doi
            and
            crossref_doi
        ):

            doi_match = (
                "Yes"
                if reference_doi.lower()
                == crossref_doi.lower()
                else "No"
            )

        else:
            doi_match = None

        # ----------------------------------------------------
        # Journal/container
        # ----------------------------------------------------
        container = first_value(
            item.get(
                "container-title"
            )
        )

        publisher = item.get(
            "publisher"
        )

        authors = (
            get_crossref_authors(
                item
            )
        )

        match_status = classify_match(
            similarity,
            match_threshold,
            possible_threshold
        )

        return {
            "Match Status": match_status,
            "Matching Rate (%)": similarity,
            "Reference Title": extracted_title,
            "Crossref Title": crossref_title,
            "Reference Year": reference_year,
            "Crossref Year": crossref_year,
            "Year Match": year_match,
            "Reference DOI": reference_doi,
            "Crossref DOI": crossref_doi,
            "DOI Match": doi_match,
            "DOI URL": doi_url,
            "URL": url,
            "Publisher": publisher,
            "Journal / Container": container,
            "Authors": authors,
            "Original Reference": reference,
            "Search Status": "Found",
            "Error": None,
        }

    except requests.exceptions.Timeout:

        error_message = (
            "Crossref request timed out."
        )

    except requests.exceptions.HTTPError as e:

        status_code = (
            e.response.status_code
            if e.response is not None
            else None
        )

        if status_code == 429:
            error_message = (
                "Crossref rate limit reached (429)."
            )

        elif status_code == 403:
            error_message = (
                "Crossref request blocked (403)."
            )

        else:
            error_message = str(e)

    except requests.exceptions.RequestException as e:

        error_message = str(e)

    except Exception as e:

        error_message = str(e)

    return {
        "Match Status": "🚫 Error",
        "Matching Rate (%)": 0.0,
        "Reference Title": extracted_title,
        "Crossref Title": None,
        "Reference Year": reference_year,
        "Crossref Year": None,
        "Year Match": None,
        "Reference DOI": reference_doi,
        "Crossref DOI": None,
        "DOI Match": None,
        "DOI URL": None,
        "URL": None,
        "Publisher": None,
        "Journal / Container": None,
        "Authors": None,
        "Original Reference": reference,
        "Search Status": "Error",
        "Error": error_message,
    }


# ============================================================
# Sidebar
# ============================================================
st.sidebar.header(
    "⚙️ Settings"
)

st.sidebar.subheader(
    "Title matching"
)

match_threshold = (
    st.sidebar.slider(
        "Match threshold (%)",
        min_value=80,
        max_value=100,
        value=90,
        step=1
    )
)

possible_threshold = (
    st.sidebar.slider(
        "Possible mismatch threshold (%)",
        min_value=50,
        max_value=89,
        value=75,
        step=1
    )
)

max_candidates = (
    st.sidebar.slider(
        "Crossref candidates to compare",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )
)

st.sidebar.subheader(
    "Requests"
)

request_delay = (
    st.sidebar.slider(
        "Delay between requests (seconds)",
        min_value=0.0,
        max_value=3.0,
        value=0.2,
        step=0.1
    )
)


# ============================================================
# Input
# ============================================================
st.subheader(
    "📚 References to verify"
)

st.write(
    "Enter **one complete reference per line**."
)

raw_text = st.text_area(
    "References",
    value="\n".join(
        DEFAULT_REFERENCES
    ),
    height=300
)

references = [
    line.strip()
    for line in raw_text.splitlines()
    if line.strip()
]

if not references:

    st.warning(
        "Please enter at least one reference."
    )

    st.stop()


# ============================================================
# Preview extraction
# ============================================================
preview_rows = []

for reference in references:

    preview_rows.append(
        {
            "Reference Title":
                extract_title_from_reference(
                    reference
                ),

            "Year":
                extract_reference_year(
                    reference
                ),

            "DOI in Reference":
                extract_reference_doi(
                    reference
                ),

            "Original Reference":
                reference,
        }
    )

preview_df = pd.DataFrame(
    preview_rows
)

with st.expander(
    "🔎 Preview extracted information",
    expanded=True
):

    st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# Run
# ============================================================
col1, col2 = st.columns(
    [1, 2]
)

with col1:

    run = st.button(
        "🔍 Run verification",
        type="primary"
    )

with col2:

    st.write(
        f"Total references: "
        f"**{len(references)}**"
    )


# ============================================================
# Validation
# ============================================================
if run:

    results = []

    progress = st.progress(
        0
    )

    status_text = st.empty()

    for index, reference in enumerate(
        references,
        start=1
    ):

        title = (
            extract_title_from_reference(
                reference
            )
        )

        status_text.write(
            f"🔍 Checking {index}/{len(references)}: "
            f"**{title}**"
        )

        result = validate_reference(
            reference=reference,
            match_threshold=match_threshold,
            possible_threshold=possible_threshold,
            max_candidates=max_candidates
        )

        results.append(
            result
        )

        progress.progress(
            index / len(references)
        )

        # Small delay to be considerate of Crossref
        if (
            request_delay > 0
            and
            index < len(references)
        ):

            time.sleep(
                request_delay
            )

    status_text.success(
        "✅ Verification completed."
    )


    # ========================================================
    # DataFrame
    # ========================================================
    df = pd.DataFrame(
        results
    )


    # ========================================================
    # Preferred output order
    # ========================================================
    column_order = [
        "Match Status",
        "Matching Rate (%)",
        "Reference Title",
        "Crossref Title",
        "Reference Year",
        "Crossref Year",
        "Year Match",
        "Reference DOI",
        "Crossref DOI",
        "DOI Match",
        "DOI URL",
        "URL",
        "Journal / Container",
        "Publisher",
        "Authors",
        "Original Reference",
        "Search Status",
        "Error",
    ]

    df = df[
        [
            col
            for col in column_order
            if col in df.columns
        ]
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
    # Items requiring review
    # ========================================================
    st.subheader(
        "🚨 References requiring review"
    )

    review_mask = (
        df["Match Status"].isin(
            [
                "⚠️ Possible mismatch",
                "❌ Mismatch",
                "❓ Not Found",
                "🚫 Error",
            ]
        )
        |
        (
            df["Year Match"]
            == "No"
        )
        |
        (
            df["DOI Match"]
            == "No"
        )
    )

    review_df = df[
        review_mask
    ].copy()


    if review_df.empty:

        st.success(
            "No obvious reference mismatches were detected."
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
        .value_counts()
        .to_dict()
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Match",
        counts.get(
            "✅ Match",
            0
        )
    )

    c2.metric(
        "Possible mismatch",
        counts.get(
            "⚠️ Possible mismatch",
            0
        )
    )

    c3.metric(
        "Mismatch",
        counts.get(
            "❌ Mismatch",
            0
        )
    )

    c4.metric(
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
    all_csv = (
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
        data=all_csv,
        file_name="reference_validation_results.csv",
        mime="text/csv"
    )


    # ========================================================
    # Download review results
    # ========================================================
    if not review_df.empty:

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
            "⬇️ Download references requiring review",
            data=review_csv,
            file_name="reference_review_items.csv",
            mime="text/csv"
        )
