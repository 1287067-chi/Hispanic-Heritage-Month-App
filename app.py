import io
import math
import re
from datetime import date, datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
API_ROOT = "https://api.worldbank.org/v2"
COUNTRIES = {
    "Mexico (MX)": "MEX",
    "Brazil (BR)": "BRA",
    "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL",
    "Chile (CL)": "CHL",
    "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN",
    "Ecuador (EC)": "ECU",
    "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": {
        "code": "SP.DYN.LE00.IN", "unit": "years",
        "definition": "Life expectancy at birth: the number of years a newborn would live if current age-specific mortality patterns persisted."
    },
    "Unemployment, national estimate": {
        "code": "SL.UEM.TOTL.NE.ZS", "unit": "% of total labor force",
        "definition": "Unemployment as estimated by national sources; methodology and comparability may differ across countries."
    },
    "GDP per capita": {
        "code": "NY.GDP.PCAP.CD", "unit": "current US dollars per person",
        "definition": "Gross domestic product divided by midyear population, in current US dollars. It is not average wealth or household income."
    },
    "Crude birth rate": {
        "code": "SP.DYN.CBRT.IN", "unit": "births per 1,000 people per year",
        "definition": "Annual number of live births per 1,000 people. This is a rate, not a total birth count."
    },
    "Net migration": {
        "code": "SM.POP.NETM", "unit": "people, net",
        "definition": "Net number of migrants over a period (immigrants minus emigrants); it is not an emigration count."
    },
    "Intentional homicide rate": {
        "code": "VC.IHR.PSRC.P5", "unit": "per 100,000 people",
        "definition": "Intentional homicide victims per 100,000 population, as reported in the World Bank indicator series."
    },
    "Consumer price inflation": {
        "code": "FP.CPI.TOTL.ZG", "unit": "% annual change",
        "definition": "Annual percentage change in the consumer price index."
    },
    "Central government debt": {
        "code": "GC.DOD.TOTL.GD.ZS", "unit": "% of GDP",
        "definition": "Central government debt expressed as a percentage of GDP."
    },
    "Total reserves including gold": {
        "code": "FI.RES.TOTL.CD", "unit": "current US dollars",
        "definition": "Total reserves held by monetary authorities, including monetary gold. This measures financial reserves, not gold mined."
    },
    "External balance on goods and services": {
        "code": "NE.RSB.GNFS.ZS", "unit": "% of GDP",
        "definition": "External balance on goods and services as a share of GDP."
    },
    "Gold production (Peru; INEI series reproduced in World Gold Council report)": {
        "code": "CUSTOM:PERU_GOLD_PRODUCTION", "unit": "metric tonnes",
        "definition": "Annual gold production in Peru, reported in a chart sourced to Peru's INEI in the World Gold Council report. Values are rounded tonnes. The 2010 value is excluded because the report says it covers only the first nine months."
    },
}

# Historical values transcribed from the cited chart, rounded to metric tonnes.
# Source: World Gold Council, The economic contribution of large-scale gold mining in Peru,
# Chart 7; chart source is INEI, Government of Peru.
PERU_GOLD_PRODUCTION = pd.DataFrame({
    "year": [1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
    "value": [45, 56, 64, 78, 94, 127, 129, 132, 155, 172, 175, 210, 202, 168, 175, 176],
})
COUNTRY_NOTES = {
    "BRA": "Brazil is included as a Latin American comparison country. It is not a Spanish-speaking country; Portuguese is its official language, but it is commonly included in regional Latin American economic comparisons."
}
MIN_YEAR, MAX_YEAR = 1950, 2100
REQUEST_TIMEOUT = 20
SOURCE_URL = "https://data.worldbank.org/indicator/"

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.title("📊 " + APP_TITLE)
st.caption("HISPANIC HERITAGE DATA LAB • Calculus pathway")
st.info(
    "Research focus: Peru gold production, 2000–2006. The ten assigned World Bank indicators below "
    "do not directly report gold mined per year. “Total reserves including gold” is a financial-reserves "
    "measure, not gold production. Use a production-specific source before drawing a conclusion about mined gold."
)

def world_bank_indicator_url(code: str) -> str:
    if code == "CUSTOM:PERU_GOLD_PRODUCTION":
        return "https://app.ingemmet.gob.pe/biblioteca/pdf/Econ-120.pdf"
    return f"{SOURCE_URL}{code}"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code: str, indicator_code: str, start_year: int, end_year: int):
    """Fetch an indicator from the World Bank API. Returns data and a human-readable status."""
    url = f"{API_ROOT}/country/{country_code}/indicator/{indicator_code}"
    params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            return pd.DataFrame(columns=["year", "value"]), "The API returned no observations."
        records = []
        for item in payload[1]:
            year_raw = item.get("date")
            val = item.get("value")
            try:
                year = int(year_raw)
            except (TypeError, ValueError):
                continue
            if val is None:
                continue
            try:
                value = float(val)
            except (TypeError, ValueError):
                continue
            records.append({"year": year, "value": value})
        df = pd.DataFrame(records, columns=["year", "value"])
        if df.empty:
            return df, "The API response contained no non-null numeric observations for this period."
        df = df.sort_values("year").drop_duplicates("year", keep="last").reset_index(drop=True)
        return df, f"Retrieved {len(df)} non-null observations."
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), f"Request timed out after {REQUEST_TIMEOUT} seconds."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Network/API error: {exc}"
    except (ValueError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not parse API response: {exc}"

def valid_value_for_indicator(value: float, indicator_name: str):
    if not np.isfinite(value):
        return False, "Values must be finite numbers."
    if indicator_name == "Life expectancy at birth" and not 0 <= value <= 125:
        return False, "Life expectancy must be between 0 and 125 years."
    if indicator_name == "Unemployment, national estimate" and not 0 <= value <= 100:
        return False, "Unemployment rate must be between 0 and 100%."
    if indicator_name == "GDP per capita" and value < 0:
        return False, "GDP per capita cannot be negative."
    if indicator_name == "Crude birth rate" and value < 0:
        return False, "Crude birth rate cannot be negative."
    if indicator_name == "Intentional homicide rate" and value < 0:
        return False, "Homicide rate cannot be negative."
    if indicator_name == "Consumer price inflation" and value < -100:
        return False, "Inflation below -100% is not accepted by this app's validation rule."
    if indicator_name == "Central government debt" and value < 0:
        return False, "Central government debt share cannot be negative in this app's validation rule."
    if indicator_name == "Total reserves including gold" and value < 0:
        return False, "Reserves cannot be negative."
    if indicator_name.startswith("Gold production") and value < 0:
        return False, "Gold production cannot be negative."
    return True, ""

def validate_table(raw: pd.DataFrame, indicator_name: str):
    errors = []
    if not {"year", "value"}.issubset(raw.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must contain year and value columns."]
    work = raw[["year", "value"]].copy()
    years = []
    for idx, item in enumerate(work["year"].tolist(), start=1):
        try:
            numeric = float(item)
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError
            years.append(int(numeric))
        except (TypeError, ValueError):
            years.append(None)
            errors.append(f"Row {idx}: year must be an integer.")
    work["year"] = years
    numeric_values = pd.to_numeric(work["value"], errors="coerce")
    for idx, val in enumerate(numeric_values.tolist(), start=1):
        if pd.isna(val) or not np.isfinite(float(val)):
            errors.append(f"Row {idx}: value must be a finite number.")
        else:
            ok, msg = valid_value_for_indicator(float(val), indicator_name)
            if not ok:
                errors.append(f"Row {idx}: {msg}")
    work["value"] = numeric_values
    if work["year"].isna().any():
        work = work.dropna(subset=["year"])
    if work["year"].duplicated().any():
        duplicates = sorted(work.loc[work["year"].duplicated(keep=False), "year"].astype(int).unique().tolist())
        errors.append("Duplicate year(s): " + ", ".join(map(str, duplicates)) + ". Keep one row per year.")
    work["year"] = work["year"].astype(int) if not work.empty else pd.Series(dtype=int)
    work = work.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
    if not work.empty and (work["year"] < 1800).any():
        errors.append("Years earlier than 1800 are not accepted.")
    return work, errors

def fit_polynomial(df: pd.DataFrame, degree: int):
    """Fit y = b0 + b1*t + ... + bd*t^d, with t = year - base_year."""
    clean = df[["year", "value"]].dropna().sort_values("year")
    if clean["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct years; only {clean['year'].nunique()} are available.")
    base_year = int(clean["year"].min())
    x = clean["year"].to_numpy(dtype=float) - base_year
    y = clean["value"].to_numpy(dtype=float)
    coeff = np.polynomial.polynomial.polyfit(x, y, degree)
    pred = np.polynomial.polynomial.polyval(x, coeff)
    r2 = float(r2_score(y, pred)) if len(y) > 1 else float("nan")
    mae = float(mean_absolute_error(y, pred))
    residuals = y - pred
    return {
        "base_year": base_year, "degree": degree, "coeff": coeff, "years": clean["year"].to_numpy(dtype=int),
        "actual": y, "fitted": pred, "residuals": residuals, "r2": r2, "mae": mae,
        "mean": float(np.mean(y)), "min_year": int(clean["year"].min()), "max_year": int(clean["year"].max()),
        "n": int(len(clean)), "data": clean.copy()
    }

def predict_model(model, year):
    t = float(year) - model["base_year"]
    return float(np.polynomial.polynomial.polyval(t, model["coeff"]))

def equation_text(model, unit):
    terms = []
    for power, coefficient in enumerate(model["coeff"]):
        if abs(coefficient) < 1e-12:
            continue
        if power == 0:
            terms.append(f"{coefficient:.8g}")
        else:
            terms.append(f"{coefficient:+.8g}·t^{power}")
    return f"ŷ = {' '.join(terms)}; t = year − {model['base_year']} ({unit})"

def chronological_holdout(df, degree):
    """Last 20% (at least one row) is held out; train only on earlier years."""
    clean = df[["year", "value"]].dropna().sort_values("year").reset_index(drop=True)
    n = len(clean)
    test_n = max(1, int(math.ceil(n * 0.2)))
    if n - test_n < degree + 1:
        return None
    train = clean.iloc[:-test_n]
    test = clean.iloc[-test_n:]
    base = int(train["year"].min())
    coeff = np.polynomial.polynomial.polyfit(train["year"].to_numpy(float) - base, train["value"].to_numpy(float), degree)
    pred = np.polynomial.polynomial.polyval(test["year"].to_numpy(float) - base, coeff)
    return {"n_test": len(test), "mae": float(mean_absolute_error(test["value"], pred)),
            "r2": float(r2_score(test["value"], pred)) if len(test) > 1 else None,
            "test": test.assign(predicted=pred, residual=test["value"].to_numpy(float) - pred)}

def derivative_coeff(coeff, order=1):
    out = np.polynomial.polynomial.polyder(coeff, m=order)
    return out

def calc_calculus(model):
    """Numerically classify stationary points and inflections within observed range."""
    coeff = model["coeff"]
    base, lo, hi = model["base_year"], model["min_year"], model["max_year"]
    d1 = derivative_coeff(coeff, 1)
    d2 = derivative_coeff(coeff, 2)
    def real_roots_in(poly, left, right):
        if len(poly) <= 1:
            return []
        roots = np.polynomial.polynomial.polyroots(poly)
        vals = []
        for root in roots:
            if abs(float(np.imag(root))) < 1e-7:
                year = base + float(np.real(root))
                if left - 1e-7 <= year <= right + 1e-7:
                    vals.append(float(year))
        return sorted(set(round(v, 6) for v in vals))
    critical = real_roots_in(d1, lo, hi)
    inflections = real_roots_in(d2, lo, hi)
    crit_rows = []
    for yr in critical:
        eps = max(1e-3, min(0.1, (hi-lo)/10000 if hi > lo else 0.01))
        left_y = predict_model(model, yr-eps)
        center_y = predict_model(model, yr)
        right_y = predict_model(model, yr+eps)
        second = float(np.polynomial.polynomial.polyval(yr-base, d2))
        if second > 1e-7:
            kind = "Local minimum"
        elif second < -1e-7:
            kind = "Local maximum"
        else:
            kind = "Stationary point; higher-order test needed"
        crit_rows.append({"year": yr, "estimate": center_y, "classification": kind, "second_derivative": second})
    endpoints = [{"year": lo, "estimate": predict_model(model, lo), "type": "Endpoint"},
                 {"year": hi, "estimate": predict_model(model, hi), "type": "Endpoint"}]
    candidates = endpoints + [{"year": r["year"], "estimate": r["estimate"], "type": r["classification"]} for r in crit_rows]
    min_candidate = min(candidates, key=lambda z: z["estimate"])
    max_candidate = max(candidates, key=lambda z: z["estimate"])
    infl_rows = []
    for yr in inflections:
        eps = max(1e-3, min(0.1, (hi-lo)/10000 if hi > lo else 0.01))
        before = float(np.polynomial.polynomial.polyval((yr-eps)-base, d2))
        after = float(np.polynomial.polynomial.polyval((yr+eps)-base, d2))
        changes = before * after < 0
        infl_rows.append({"year": yr, "estimate": predict_model(model, yr), "verified_sign_change": changes})
    return d1, d2, crit_rows, min_candidate, max_candidate, infl_rows

def make_chart(df, model, sample_increment, single_result=None, batch_results=None, compare_df=None, compare_label=None):
    fig, ax = plt.subplots(figsize=(10, 5.3))
    ax.scatter(df["year"], df["value"], label="Observed data", zorder=4)
    x_curve = np.linspace(model["min_year"], model["max_year"], max(300, (model["max_year"]-model["min_year"]+1)*5))
    y_curve = [predict_model(model, yr) for yr in x_curve]
    ax.plot(x_curve, y_curve, label=f"Degree {model['degree']} fitted curve", linewidth=2)
    sampled = np.arange(model["min_year"], model["max_year"] + 1, sample_increment)
    ax.scatter(sampled, [predict_model(model, yr) for yr in sampled], marker="x",
               label=f"Model markers every {sample_increment} year(s)")
    if model.get("display_end", model["max_year"]) > model["max_year"]:
        x_ext = np.linspace(model["max_year"], model["display_end"], 120)
        ax.plot(x_ext, [predict_model(model, yr) for yr in x_ext], "--", label="Extrapolated segment")
    ax.axvline(model["max_year"], linestyle=":", linewidth=1.4, label="Historical boundary")
    if single_result is not None:
        ax.scatter([single_result["year"]], [single_result["estimate"]], marker="*", s=160, label="Single-year estimate")
    if batch_results is not None and not batch_results.empty:
        ax.scatter(batch_results["year"], batch_results["estimate"], marker="D", s=35, label="Batch estimates")
    if compare_df is not None and not compare_df.empty:
        ax.plot(compare_df["year"], compare_df["value"], linestyle=":", marker=".", label=compare_label or "Comparison country")
    ax.set_xlabel("Year")
    ax.set_ylabel(model["unit"])
    ax.set_title(model["chart_title"])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def make_excel(summary: dict, working_data: pd.DataFrame, calculated: pd.DataFrame, residuals: pd.DataFrame):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="Summary")
        working_data.to_excel(writer, index=False, sheet_name="Working Data")
        calculated.to_excel(writer, index=False, sheet_name="Calculated Years")
        residuals.to_excel(writer, index=False, sheet_name="Residuals")
        for sheet, frame in [("Summary", pd.DataFrame([summary])), ("Working Data", working_data),
                             ("Calculated Years", calculated), ("Residuals", residuals)]:
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(1, len(frame)), max(0, len(frame.columns)-1))
            for i, col in enumerate(frame.columns):
                max_len = min(42, max(12, len(str(col)) + 2,
                                      int(frame[col].astype(str).map(len).max() + 2) if len(frame) else 12))
                ws.set_column(i, i, max_len)
    buffer.seek(0)
    return buffer.getvalue()

def make_pdf(summary, chart_bytes, preview, calculated, findings):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=0.45*inch,
                            leftMargin=0.45*inch, topMargin=0.45*inch, bottomMargin=0.45*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story = [Paragraph(APP_TITLE, styles["Title"]), Spacer(1, 8)]
    for key in ["country", "indicator", "indicator_code", "units", "retrieved", "data_status", "edit_status",
                "base_year", "degree", "equation", "r_squared_in_sample", "mae_in_sample", "observation_count",
                "historical_range", "historical_mean", "holdout_summary"]:
        if key in summary:
            story.append(Paragraph(f"<b>{key.replace('_',' ').title()}:</b> {str(summary[key])}", styles["Small"]))
    story.append(Spacer(1, 10))
    story.append(Image(io.BytesIO(chart_bytes), width=9.5*inch, height=4.7*inch))
    story.append(Paragraph("Findings and limitations", styles["Heading2"]))
    for item in findings:
        story.append(Paragraph("• " + str(item), styles["Small"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Data preview", styles["Heading2"]))
    def table_from_df(frame, max_rows=14):
        f = frame.head(max_rows).copy()
        data = [list(map(str, f.columns))] + f.astype(str).values.tolist()
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8edf3")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.black),
            ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f9fb")]),
        ]))
        return t
    story.append(table_from_df(preview))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Calculated-year results", styles["Heading2"]))
    story.append(table_from_df(calculated, max_rows=24))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Sidebar controls
with st.sidebar:
    st.header("Explorer controls")
    country_label = st.selectbox("Country", list(COUNTRIES.keys()), index=list(COUNTRIES.keys()).index("Peru (PE)"))
    country_code = COUNTRIES[country_label]
    indicator_name = st.selectbox("Indicator", list(INDICATORS.keys()), index=8)
    indicator = INDICATORS[indicator_name]
    this_year = date.today().year
    start_year, end_year = this_year - 70, this_year - 1
    degree = st.slider("Polynomial degree", min_value=3, max_value=8, value=3)
    sample_increment = st.slider("Graph sample increment (years)", 1, 10, 1)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 0)
    comparison_options = ["None"] if indicator["code"] == "CUSTOM:PERU_GOLD_PRODUCTION" else ["None"] + [x for x in COUNTRIES if COUNTRIES[x] != country_code]
    comparison_label = st.selectbox("Comparison country (same indicator)", comparison_options)
    printer_mode = st.checkbox("Printer-friendly report view", value=False)

st.caption(f"Requested World Bank period: {start_year}–{end_year} (current year minus 70 through current year minus 1).")
st.markdown(f"**Indicator:** {indicator_name} (`{indicator['code']}`) · **Unit:** {indicator['unit']}")
st.write(indicator["definition"])
st.markdown(f"Source: [{indicator['code']} on World Bank Data]({world_bank_indicator_url(indicator['code'])})")
st.caption("API: World Bank Indicators API v2 · Retrieval date: " + date.today().isoformat())

if indicator["code"] == "CUSTOM:PERU_GOLD_PRODUCTION":
    if country_code != "PER":
        st.error("This production-specific series is available for Peru only. Select Peru to analyze these observations.")
        st.stop()
    source_df = PERU_GOLD_PRODUCTION[
        (PERU_GOLD_PRODUCTION["year"] >= start_year) &
        (PERU_GOLD_PRODUCTION["year"] <= end_year)
    ].copy()
    fetch_status = f"Loaded {len(source_df)} published chart observations (rounded tonnes); source chart attributes data to INEI, Government of Peru."
else:
    source_df, fetch_status = fetch_world_bank(country_code, indicator["code"], start_year, end_year)
st.write(f"**Source status:** {fetch_status}")
if country_code in COUNTRY_NOTES:
    st.caption(COUNTRY_NOTES[country_code])
if source_df.empty:
    st.error("No source observations are available to fit a model. Check your connection, API status, country, or indicator.")
    st.stop()

# Keep original source data and editable working copy separate.
source_df = source_df[["year", "value"]].copy().sort_values("year").reset_index(drop=True)
source_df["value"] = pd.to_numeric(source_df["value"], errors="coerce")
source_df = source_df.dropna().reset_index(drop=True)
if "edit_key" not in st.session_state:
    st.session_state.edit_key = 0
if "scenario_data" not in st.session_state:
    st.session_state.scenario_data = None
if "scenario_signature" not in st.session_state:
    st.session_state.scenario_signature = None
source_signature = (country_code, indicator["code"], tuple(source_df["year"].tolist()), tuple(np.round(source_df["value"], 10).tolist()))
if st.session_state.scenario_signature != source_signature:
    st.session_state.scenario_data = source_df.copy()
    st.session_state.scenario_signature = source_signature
    st.session_state.edit_key += 1

st.subheader("1. Historical data and editable working table")
c1, c2, c3 = st.columns(3)
c1.metric("Available observations", f"{len(source_df)}")
c2.metric("Actual coverage", f"{int(source_df.year.min())}–{int(source_df.year.max())}" if not source_df.empty else "None")
c3.metric("Missing years within range", f"{max(0, int(source_df.year.max()-source_df.year.min()+1-len(source_df)))}" if len(source_df) else "—")
st.caption("The API may not have a value for every year. Coverage and observation count reflect non-null numeric observations, not a promise of 70 rows.")

edit_col1, edit_col2 = st.columns([1, 4])
with edit_col1:
    if st.button("Reset to source", help="Discard edits and restore the API observations."):
        st.session_state.scenario_data = source_df.copy()
        st.session_state.edit_key += 1
        st.rerun()
with edit_col2:
    current_is_edited = not st.session_state.scenario_data.equals(source_df)
    if current_is_edited:
        st.warning("User-modified scenario: the working data differs from the original World Bank data.")
    else:
        st.success("Working table currently matches source data.")

edited = st.data_editor(
    st.session_state.scenario_data,
    key=f"editor_{st.session_state.edit_key}",
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "year": st.column_config.NumberColumn("Year", min_value=1800, max_value=2200, step=1, format="%d"),
        "value": st.column_config.NumberColumn(f"Value ({indicator['unit']})", format="%.8g"),
    },
    disabled=False,
)
validated, validation_errors = validate_table(edited, indicator_name)
if validation_errors:
    st.error("Please correct the working table before fitting the model:")
    for err in validation_errors[:12]:
        st.write("• " + err)
if validated.empty:
    st.error("The validated table is empty.")
    st.stop()
if validated["year"].duplicated().any():
    st.error("Duplicate years must be fixed before fitting.")
    st.stop()
if validation_errors:
    st.stop()
st.session_state.scenario_data = validated.copy()
is_edited = not validated.reset_index(drop=True).equals(source_df.reset_index(drop=True))
edit_status = "User-modified scenario" if is_edited else "Original source data"

st.subheader("2. Fit the polynomial model")
try:
    model = fit_polynomial(validated, degree)
except ValueError as exc:
    st.error(str(exc))
    st.info("Reduce the degree or add enough distinct, valid year/value rows. A degree-d model requires at least d+1 distinct years.")
    st.stop()
model["unit"] = indicator["unit"]
model["display_end"] = model["max_year"] + horizon
model["chart_title"] = f"{country_label}: {indicator_name}"
equation = equation_text(model, indicator["unit"])
holdout = chronological_holdout(validated, degree)
residual_table = pd.DataFrame({
    "year": model["years"], "actual": model["actual"], "fitted": model["fitted"], "residual": model["residuals"]
})
summary = {
    "country": country_label, "country_code": country_code, "indicator": indicator_name,
    "indicator_code": indicator["code"], "units": indicator["unit"], "definition": indicator["definition"],
    "retrieved": date.today().isoformat(), "data_status": fetch_status, "edit_status": edit_status,
    "base_year": model["base_year"], "degree": degree, "equation": equation,
    "r_squared_in_sample": model["r2"], "mae_in_sample": model["mae"], "observation_count": model["n"],
    "historical_range": f"{model['min_year']}–{model['max_year']}", "historical_mean": model["mean"],
    "sample_increment": sample_increment, "forward_display_horizon": horizon,
    "holdout_summary": "Not enough training observations for chronological holdout." if holdout is None else
        f"Last {holdout['n_test']} observation(s): MAE={holdout['mae']:.6g}; R²=" +
        ("not defined for one test point" if holdout["r2"] is None else f"{holdout['r2']:.6g}")
}
st.caption(f"Base year = {model['base_year']}; t = year − {model['base_year']}. The model fits values using centered time, not raw calendar years.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("R² (in-sample)", f"{model['r2']:.5f}")
m2.metric("MAE (in-sample)", f"{model['mae']:.6g} {indicator['unit']}")
m3.metric("Observation count", str(model["n"]))
m4.metric("Historical mean", f"{model['mean']:.6g} {indicator['unit']}")
st.code(equation)
st.caption("R² and MAE are in-sample fit metrics, not proof of causation or future accuracy.")
if holdout is not None:
    st.write(f"**Chronological holdout:** last {holdout['n_test']} observation(s) held out from fitting. Holdout MAE = {holdout['mae']:.6g} {indicator['unit']}; " +
             ("holdout R² is not defined for one test observation." if holdout["r2"] is None else f"holdout R² = {holdout['r2']:.5f}."))
else:
    st.write("**Chronological holdout:** not reported because there are too few training observations for this degree.")
with st.expander("Residuals and fitted values at observed years"):
    st.dataframe(residual_table, use_container_width=True)

st.subheader("3. Calculate one year or a batch of years")
st.caption("Single-year and batch estimates use the exact same fitted polynomial. Values at observed years are model-fitted estimates, not the original observations.")
if "single_result" not in st.session_state:
    st.session_state.single_result = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = pd.DataFrame()
model_signature = (country_code, indicator["code"], degree, tuple(validated["year"].tolist()), tuple(np.round(validated["value"], 10).tolist()))
if st.session_state.get("result_signature") != model_signature:
    st.session_state.single_result = None
    st.session_state.batch_results = pd.DataFrame()
    st.session_state.result_signature = model_signature

left, right = st.columns(2)
with left:
    single_year = st.number_input("Single year (1950–2100)", min_value=MIN_YEAR, max_value=MAX_YEAR, value=int(model["max_year"]), step=1)
    if st.button("Calculate Value"):
        st.session_state.single_result = {"year": int(single_year), "estimate": predict_model(model, int(single_year))}
with right:
    batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
    batch_input = ""
    if batch_mode == "Year range":
        range_col1, range_col2, range_col3 = st.columns(3)
        range_start = range_col1.number_input("Start", min_value=MIN_YEAR, max_value=MAX_YEAR, value=int(model["min_year"]), step=1)
        range_end = range_col2.number_input("End", min_value=MIN_YEAR, max_value=MAX_YEAR, value=min(MAX_YEAR, int(model["max_year"])), step=1)
        range_step = range_col3.number_input("Step", min_value=1, max_value=10, value=1, step=1)
    else:
        batch_input = st.text_input("Comma-separated years", value=f"{model['min_year']}, {model['max_year']}", help="Enter whole years from 1950 through 2100, separated by commas.")
    if st.button("Calculate Batch"):
        requested, invalid = [], []
        if batch_mode == "Year range":
            if range_end < range_start:
                invalid.append("End year must be greater than or equal to start year.")
            else:
                requested = list(range(int(range_start), int(range_end) + 1, int(range_step)))
        else:
            for token in batch_input.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    number = float(token)
                    if not number.is_integer():
                        invalid.append(f"'{token}' is not an integer year.")
                    elif not MIN_YEAR <= int(number) <= MAX_YEAR:
                        invalid.append(f"'{token}' is outside {MIN_YEAR}–{MAX_YEAR}.")
                    else:
                        requested.append(int(number))
                except ValueError:
                    invalid.append(f"'{token}' is not a valid year.")
        requested = list(dict.fromkeys(requested))
        if invalid:
            st.error("Invalid batch input: " + " ".join(invalid))
        elif not requested:
            st.error("Enter at least one valid year.")
        else:
            st.session_state.batch_results = pd.DataFrame([
                {"year": yr, "estimate": predict_model(model, yr),
                 "range_status": "Within fitted historical range" if model["min_year"] <= yr <= model["max_year"] else "Outside fitted historical range",
                 "observed_year": "Yes — fitted value at an observed year" if yr in set(model["years"].tolist()) else "No"}
                for yr in requested
            ]).sort_values("year").reset_index(drop=True)
            st.session_state.batch_error = None
if st.session_state.single_result is not None:
    result = st.session_state.single_result
    inside = model["min_year"] <= result["year"] <= model["max_year"]
    observed = result["year"] in set(model["years"].tolist())
    st.success(f"**Year:** {result['year']} · **Model estimate:** {result['estimate']:.8g} {indicator['unit']} · "
               f"**Fitted historical range:** {model['min_year']}–{model['max_year']} · "
               f"**Classification:** {'within range' if inside else 'outside range'}")
    if observed:
        st.caption("This is the fitted model value at a year with an observation. It is not the same as the actual source value.")
    else:
        st.caption("No observed value for this exact year is being presented as an actual observation.")
batch_results = st.session_state.batch_results
if not batch_results.empty:
    st.dataframe(batch_results, use_container_width=True)
    if st.session_state.single_result is not None:
        sy = st.session_state.single_result["year"]
        match = batch_results[batch_results["year"] == sy]
        if not match.empty:
            difference = abs(float(match.iloc[0]["estimate"]) - st.session_state.single_result["estimate"])
            st.caption(f"Single/batch consistency check for {sy}: absolute difference = {difference:.3g} (should be 0 apart from floating-point rounding).")

st.subheader("4. Visualization")
compare_df = None
compare_name = None
if comparison_label != "None":
    comp_code = COUNTRIES[comparison_label]
    compare_df, compare_status = fetch_world_bank(comp_code, indicator["code"], start_year, end_year)
    if compare_df.empty:
        st.warning(f"Comparison country data unavailable: {compare_status}")
        compare_df = None
    else:
        compare_name = f"{comparison_label} observed data"
        st.caption(f"Comparison: {comparison_label}; {compare_status}. Different coverage or missing years can make direct comparisons incomplete.")
fig = make_chart(validated, model, sample_increment, st.session_state.single_result,
                 batch_results if not batch_results.empty else None, compare_df, compare_name)
st.pyplot(fig, use_container_width=True)
chart_buffer = io.BytesIO()
fig.savefig(chart_buffer, format="png", dpi=160, bbox_inches="tight")
chart_bytes = chart_buffer.getvalue()
plt.close(fig)

st.subheader("5. Calculus interpretation")
d1, d2, critical_rows, interval_min, interval_max, inflection_rows = calc_calculus(model)
st.write("Let the fitted model be ŷ(t), where t = year − base year.")
st.latex(r"\frac{d\hat y}{dt} = \hat y'(t)")
st.write("The first derivative estimates the model's instantaneous rate of change per year. The second derivative describes how that rate changes.")
st.latex(r"\frac{d^2\hat y}{dt^2} = \hat y''(t)")
deriv_text = "First derivative coefficients in ascending powers of t: " + np.array2string(d1, precision=6, separator=", ")
second_text = "Second derivative coefficients in ascending powers of t: " + np.array2string(d2, precision=6, separator=", ")
st.code(deriv_text + "\n" + second_text)
st.write(f"**Domain shown:** {model['min_year']}–{model['max_year']} for historical interpretation. **Model range on that domain:** "
         f"{min(model['fitted']):.6g} to {max(model['fitted']):.6g} {indicator['unit']} (range of fitted values at observed-year locations; continuous-domain extrema are assessed below).")
if critical_rows:
    st.write("**Critical points inside the historical interval:**")
    st.dataframe(pd.DataFrame(critical_rows), use_container_width=True)
else:
    st.write("No real critical points were found inside the historical interval.")
st.write(f"**Interval minimum (including endpoints):** year {interval_min['year']:.4f}, model estimate {interval_min['estimate']:.6g} ({interval_min['type']}).")
st.write(f"**Interval maximum (including endpoints):** year {interval_max['year']:.4f}, model estimate {interval_max['estimate']:.6g} ({interval_max['type']}).")
if inflection_rows:
    st.write("**Second-derivative roots within the historical interval:**")
    st.dataframe(pd.DataFrame(inflection_rows), use_container_width=True)
    verified = [x for x in inflection_rows if x["verified_sign_change"]]
    st.caption(f"{len(verified)} candidate(s) show a numerical sign change in the second derivative; only these are verified inflection candidates.")
else:
    st.write("No real second-derivative roots were found inside the historical interval.")
if model["degree"] >= 3:
    st.write("**Concavity:** where ŷ''(t) is positive the fitted curve is concave up; where negative it is concave down. Candidate inflection points require a sign change, not just a second-derivative root.")
st.write("**Average rate of change:** for two years a and b, (ŷ(b) − ŷ(a)) / (b − a). The derivative is the instantaneous model rate; neither is automatically a causal explanation.")

findings = [
    f"The model uses {model['n']} valid observations from {model['min_year']} to {model['max_year']}.",
    f"In-sample R² is {model['r2']:.5f}; in-sample MAE is {model['mae']:.6g} {indicator['unit']}. These describe fit to the same data used for fitting.",
    f"Historical mean is {model['mean']:.6g} {indicator['unit']}.",
    "Polynomial extrapolation outside the observed range is uncertain. Residuals and MAE describe historical fit, not validated 95% prediction intervals.",
    "High-degree polynomials can oscillate and produce implausible values outside the fitted range. Predictions are not clipped; inspect them critically.",
    "The World Bank indicator selected here is not a direct measure of annual gold production."
]
if horizon > 0:
    ext_year = model["max_year"] + horizon
    ext_value = predict_model(model, ext_year)
    if not np.isfinite(ext_value) or abs(ext_value) > 1e12 or (indicator_name not in ["Consumer price inflation", "External balance on goods and services"] and ext_value < 0):
        st.warning(f"Extrapolation warning: the model estimate at {ext_year} is {ext_value:.6g}. This may be implausible for the selected indicator. It is shown without clipping.")
    findings.append(f"At the display horizon endpoint {ext_year}, the polynomial estimates {ext_value:.6g} {indicator['unit']}; treat this as a mathematical extrapolation, not a forecast guarantee.")

st.subheader("6. Export current analysis")
calculated_for_export = batch_results.copy()
if calculated_for_export.empty:
    calculated_for_export = pd.DataFrame(columns=["year", "estimate", "range_status", "observed_year"])
if st.session_state.single_result is not None:
    single_df = pd.DataFrame([{
        "year": st.session_state.single_result["year"],
        "estimate": st.session_state.single_result["estimate"],
        "range_status": "Within fitted historical range" if model["min_year"] <= st.session_state.single_result["year"] <= model["max_year"] else "Outside fitted historical range",
        "observed_year": "Yes — fitted value at an observed year" if st.session_state.single_result["year"] in set(model["years"].tolist()) else "No"
    }])
    calculated_for_export = pd.concat([single_df, calculated_for_export], ignore_index=True).drop_duplicates(subset=["year"], keep="first").sort_values("year")
working_export = validated.copy()
working_export["data_status"] = edit_status
csv_bytes = working_export.to_csv(index=False).encode("utf-8")
excel_bytes = make_excel(summary, working_export, calculated_for_export, residual_table)
pdf_bytes = make_pdf(summary, chart_bytes, working_export, calculated_for_export, findings)
download_cols = st.columns(3)
download_cols[0].download_button("Download working data CSV", data=csv_bytes, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
download_cols[1].download_button("Download analysis Excel (.xlsx)", data=excel_bytes, file_name="hispanic_heritage_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
download_cols[2].download_button("Download report PDF", data=pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
st.caption("CSV contains the validated working table; Excel contains Summary, Working Data, Calculated Years, and Residuals; PDF contains metadata, equation, metrics, chart, findings/limitations, data preview, and calculated results.")

# A focused research note for the requested Peru 2000–2006 question.
st.subheader("7. Featured research question: Peru gold production, 2000–2006")
st.write(
    "The World Bank indicators cannot by themselves establish annual gold production. The separate Peru gold-production option uses "
    "rounded annual tonnes transcribed from Chart 7 of the cited World Gold Council report, which credits INEI, Government of Peru. "
    "For this series, the reported value rises from 129 tonnes in 2000 to 202 tonnes in 2006, an increase of 73 tonnes (about 56.6%). "
    "Production peaks at 210 tonnes in 2005 before falling in 2006. These are rounded chart values; consult the original official series "
    "for precision. The indicator “Total reserves including gold” is a monetary stock of official reserves, not the quantity of gold mined."
)
st.markdown("Sources: [World Bank API documentation](https://datahelpdesk.worldbank.org/knowledgebase/articles/889386-about-the-indicators-api-documentation), "
            "[World Gold Council report hosted by INGEMMET (Chart 7; source listed as INEI)](https://app.ingemmet.gob.pe/biblioteca/pdf/Econ-120.pdf), "
            "[USGS Minerals Yearbook](https://www.usgs.gov/centers/national-minerals-information-center/minerals-yearbook-metals-and-minerals), "
            "[World Mining Data](https://www.world-mining-data.info/). The included gold-production values are rounded tonnes transcribed from the report chart; verify exact figures and table notes if precision beyond rounded tonnes is needed.")

if printer_mode:
    st.markdown("---")
    st.subheader("Printer-friendly summary")
    st.write(f"**{country_label} — {indicator_name}** ({indicator['unit']})")
    st.write(f"Coverage: {model['min_year']}–{model['max_year']} · n={model['n']} · degree={degree} · base year={model['base_year']}")
    st.write(f"Equation: {equation}")
    st.write(f"In-sample R²={model['r2']:.5f}; in-sample MAE={model['mae']:.6g}; mean={model['mean']:.6g}.")
    st.write("Interpretation is descriptive, not causal. Extrapolations are uncertain and are not validated prediction intervals.")
    st.dataframe(validated, use_container_width=True)
