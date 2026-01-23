import time
import random
import pandas as pd
import streamlit as st

from scholarly import scholarly, ProxyGenerator
from Levenshtein import ratio


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Reference Validator (Google Scholar)", layout="wide")
st.title("📚 Reference Validator (Google Scholar)")
st.caption("⚠️ Google Scholar는 자동 요청을 차단할 수 있어 결과가 'Error'로 나올 수 있습니다. 특히 Streamlit Cloud에서는 빈번합니다.")


# ----------------------------
# Defaults
# ----------------------------
DEFAULT_TITLES = [
    "Attention Is All You Need",
    "Generative Adversarial Nets",
    "A Non-existent Paper about Flying Spaghettis in Deep Learning",
    "국가 AI 정책과 영어영문학의 비밀에 대한 연구",
]


# ----------------------------
# Optional: Proxy support
# ----------------------------
def configure_proxy_if_available() -> bool:
    """
    Streamlit Secrets에 아래 키가 있으면 SOCKS5 프록시를 설정합니다.
    - proxy_host
    - proxy_port
    - proxy_user (optional)
    - proxy_pass (optional)

    예) .streamlit/secrets.toml
    proxy_host="xxx"
    proxy_port="1080"
    proxy_user="user"
    proxy_pass="pass"
    """
    try:
        secrets = st.secrets
        if "proxy_host" not in secrets or "proxy_port" not in secrets:
            return False

        host = secrets["proxy_host"]
        port = int(secrets["proxy_port"])
        user = secrets.get("proxy_user", None)
        pw = secrets.get("proxy_pass", None)

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


# ----------------------------
# Scholar check function
# ----------------------------
def check_reference_validity(title_to_check: str):
    """
    Google Scholar에서 title_to_check를 검색하고 첫 결과의 제목과 유사도(%)를 반환.
    """
    try:
        search_query = scholarly.search_pubs(title_to_check)
        first_result = next(search_query)  # StopIteration if no results

        found_title = first_result.get("bib", {}).get("title", "")
        similarity = ratio(title_to_check.lower(), found_title.lower()) if found_title else 0.0

        return {
            "status": "Found",
            "searched_title": title_to_check,
            "found_title": found_title,
            "similarity": round(similarity * 100, 2),
            "url": first_result.get("pub_url") or first_result.get("eprint_url") or "N/A",
        }

    except StopIteration:
        return {
            "status": "Not Found",
            "searched_title": title_to_check,
            "found_title": None,
            "similarity": 0.0,
            "url": None,
        }
    except Exception as e:
        return {
            "status": "Error",
            "searched_title": title_to_check,
            "found_title": str(e),
            "similarity": 0.0,
            "url": None,
        }


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("⚙️ Settings")
min_delay = st.sidebar.slider("Minimum delay (seconds)", 2, 15, 5, 1)
max_delay = st.sidebar.slider("Maximum delay (seconds)", min_delay, 25, 10, 1)
max_retries = st.sidebar.slider("Retries on Error", 0, 3, 1, 1)

use_proxy = st.sidebar.checkbox("Try proxy from secrets.toml (optional)", value=False)
if use_proxy:
    ok = configure_proxy_if_available()
    st.sidebar.write("Proxy status:", "✅ enabled" if ok else "❌ not configured / failed")


# ----------------------------
# Input titles
# ----------------------------
st.subheader("✅ Titles to verify")
raw_text = st.text_area(
    "Enter one title per line",
    value="\n".join(DEFAULT_TITLES),
    height=160,
)

titles = [t.strip() for t in raw_text.splitlines() if t.strip()]
if not titles:
    st.warning("Please enter at least one title.")
    st.stop()

colA, colB = st.columns([1, 2])
with colA:
    run = st.button("🔍 Run verification", type="primary")
with colB:
    st.write(f"Total titles: **{len(titles)}**")


# ----------------------------
# Run
# ----------------------------
if run:
    results = []
    progress = st.progress(0)
    log_box = st.empty()

    with st.status("Running Google Scholar checks...", expanded=True) as status:
        for i, title in enumerate(titles, start=1):
            log_box.write(f"🔍 Searching: **{title}**")

            attempt = 0
            result = None

            while True:
                attempt += 1
                result = check_reference_validity(title)

                # If Error and retries remain, wait and retry
                if result["status"] == "Error" and attempt <= max_retries:
                    st.write(f"⚠️ Error (attempt {attempt}/{max_retries}). Retrying after a delay...")
                    time.sleep(random.uniform(min_delay, max_delay))
                    continue
                break

            results.append(result)

            # Update UI per item
            if result["status"] == "Found":
                st.success(f"✅ Found | Similarity: {result['similarity']}%")
                st.write(f"- Searched: {result['searched_title']}")
                st.write(f"- Found: {result['found_title']}")
                if result["url"] and result["url"] != "N/A":
                    st.write(f"- URL: {result['url']}")
            elif result["status"] == "Not Found":
                st.warning("❌ Not Found on Google Scholar (or no results returned).")
            else:
                st.error(f"🚫 Error: {result['found_title']}")

            # Delay to reduce blocking risk (even after success)
            time.sleep(random.uniform(min_delay, max_delay))

            progress.progress(i / len(titles))

        status.update(label="Done", state="complete", expanded=False)

    # Show results table
    st.subheader("📊 Results")
    df = pd.DataFrame(results)

    # Make URLs clickable in Streamlit dataframe by showing separately
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv_bytes,
        file_name="reference_validation_results.csv",
        mime="text/csv",
    )

    # Quick summary
    counts = df["status"].value_counts(dropna=False).to_dict()
    st.info(f"Summary: {counts}")
