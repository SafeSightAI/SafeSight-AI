import streamlit as st
import pandas as pd
import folium
import plotly.express as px
import cv2
import numpy as np
import re
import html

from pathlib import Path
from streamlit_folium import st_folium


# =====================================
# 1. PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="SafeSight AI",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LOGO_FILE = BASE_DIR / "assets" / "safesight_logo.png.jpeg"

CRIME_FILE = DATA_DIR / "district_crime_2025.csv"
TRAFFIC_FILE = DATA_DIR / "traffic_accidents_yearly.csv"
PUNJAB_TRAFFIC_FILE = DATA_DIR / "punjab_traffic_yearly.csv"
PREDICTION_FILE = DATA_DIR / "traffic_predictions.csv"
CRIME_COMPARISON_FILE = DATA_DIR / "crime_comparison_2023_2025.csv"


# =====================================
# 2. DARK NAVY + CYAN DESIGN
# =====================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F4F6FA;
    color: #263445;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #101D32;
    border-right: 2px solid #20B8B0;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #F1F5F9;
}

/* Main headings */
h1, h2, h3 {
    color: #172B4D;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE4EC;
    border-left: 4px solid #20B8B0;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(20, 40, 70, 0.06);
}

/* Metric numbers */
[data-testid="stMetricValue"] {
    color: #087E83;
}

/* Captions */
.stCaption {
    color: #64748B;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# 3. LOAD DATA
# =====================================

@st.cache_data
def load_data():

    crime = pd.read_csv(CRIME_FILE, encoding="utf-8-sig")

    traffic = pd.read_csv(TRAFFIC_FILE, encoding="utf-8-sig")

    punjab = pd.read_csv(
        PUNJAB_TRAFFIC_FILE,
        encoding="utf-8-sig"
    )

    rename_map = {
        "district": "District",
        "allreported": "All Reported",
        "murder": "Murder",
        "attemptedmurder": "Attempted Murder",
        "socide": "Suicide",
        "attemptedsocide": "Attempted Suicide",
        "gangrapezina": "Gang Rape / Zina",
        "kidnappingabduction": "Kidnapping / Abduction",
        "dacoity": "Dacoity",
        "robbery": "Robbery",
        "traficaccident": "Traffic Accident",
        "burglary": "Burglary",
        "cattletheft": "Cattle Theft",
        "motervehicaltheft": "Motor Vehicle Theft",
        "othertheft": "Other Theft",
        "miscellaneous": "Miscellaneous",
        "latitude": "Latitude",
        "longitude": "Longitude"
    }

    for column in crime.columns:

        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            str(column).lower()
        )

        if normalized in rename_map:

            crime.rename(
                columns={column: rename_map[normalized]},
                inplace=True
            )

    crime["District"] = crime["District"].astype(str).str.strip()

    crime = crime.drop_duplicates(
        subset=["District"],
        keep="first"
    )

    for column in crime.columns:

        if column != "District":

            crime[column] = pd.to_numeric(
                crime[column],
                errors="coerce"
            )

    for df in [traffic, punjab]:

        df["Year"] = df["Year"].astype(str)

        numeric_columns = [
            "Total_Accidents",
            "Fatal",
            "Non_Fatal",
            "Killed",
            "Injured",
            "Vehicles_Involved"
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return crime, traffic, punjab


# =====================================
# 4. CHECK FILES
# =====================================

required_files = [
    CRIME_FILE,
    TRAFFIC_FILE,
    PUNJAB_TRAFFIC_FILE
]

missing_files = [
    file.name for file in required_files
    if not file.exists()
]

if missing_files:

    st.error("Required data files are missing.")

    st.write(missing_files)

    st.stop()


if not LOGO_FILE.exists():

    st.warning(
        "Logo missing. Check assets/safesight_logo.png"
    )


try:

    crime_df, traffic_df, punjab_df = load_data()

except Exception as error:

    st.error("Data loading error.")

    st.exception(error)

    st.stop()


if not {"All Reported", "District", "Latitude", "Longitude"}.issubset(
    crime_df.columns
):

    st.error("Required crime dataset columns are missing.")

    st.stop()

# =====================================
# DISTRICT RISK SCORE CALCULATION
# =====================================

minimum_crime = crime_df["All Reported"].min()
maximum_crime = crime_df["All Reported"].max()

if maximum_crime == minimum_crime:

    crime_df["Crime_Index"] = 0

else:

    crime_df["Crime_Index"] = (
        (crime_df["All Reported"] - minimum_crime)
        / (maximum_crime - minimum_crime)
    ) * 100

# Rank districts by reported crime volume
crime_df["Crime_Rank_Percentile"] = (
    crime_df["All Reported"].rank(pct=True)
)

def classify_risk(percentile):

    if percentile <= 1/3:
        return "Low"

    elif percentile <= 2/3:
        return "Moderate"

    else:
        return "High"

crime_df["Risk_Level"] = (
    crime_df["Crime_Rank_Percentile"].apply(classify_risk)
)

crime_df["Crime_Index"] = crime_df["Crime_Index"].round(1)

# =====================================
# 5. SIDEBAR WITH CUSTOM LOGO
# =====================================

if LOGO_FILE.exists():

    st.sidebar.image(
        str(LOGO_FILE),
        use_container_width=True
    )

st.sidebar.title("SafeSight AI")

st.sidebar.caption("Together for a Safer Punjab")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Safety Overview",
        "Crime Intelligence",
        "Crime Trend Analysis",
        "District Risk Analysis",
        "Traffic Intelligence",
        "AI Traffic Predictions",
        "Interactive Safety Map",
        "SafeSight Vision",
        "About Data"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption("Punjab, Pakistan")

st.sidebar.caption("AI & Data Analytics Project")


# =====================================
# 6. MAIN HEADER
# =====================================

st.title("SafeSight AI")

st.subheader(
    "Punjab All-in-One Safety Intelligence Platform"
)

st.caption(
    "Crime Analysis | Traffic Intelligence | Interactive Map"
)

st.info(
    "Crime data currently covers seven districts. "
    "Map markers show approximate district centres, "
    "not exact incident locations."
)


# =====================================
# PAGE 1: SAFETY OVERVIEW
# =====================================

if page == "Safety Overview":

    st.header("Safety Overview")

    latest_year = sorted(
        punjab_df["Year"].dropna().unique()
    )[-1]

    latest_traffic = punjab_df[
        punjab_df["Year"] == latest_year
    ]

    total_crime = int(
        crime_df["All Reported"].sum()
    )

    total_murders = int(
        crime_df["Murder"].sum()
    )

    latest_accidents = int(
        latest_traffic["Total_Accidents"].sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Districts Covered",
        crime_df["District"].nunique()
    )

    col2.metric(
        "Reported Crime",
        f"{total_crime:,}"
    )

    col3.metric(
        "Reported Murders",
        f"{total_murders:,}"
    )

    col4.metric(
        f"Punjab Accidents ({latest_year})",
        f"{latest_accidents:,}"
    )

    st.divider()
    # Punjab Safety Highlights

    st.subheader("Punjab Safety Highlights")

    highest_crime = crime_df.loc[
        crime_df["All Reported"].idxmax()
    ]

    lowest_crime = crime_df.loc[
        crime_df["All Reported"].idxmin()
    ]

    high_risk_districts = (
        crime_df["Risk_Level"] == "High"
    ).sum()

    moderate_risk_districts = (
        crime_df["Risk_Level"] == "Moderate"
    ).sum()

    low_risk_districts = (
        crime_df["Risk_Level"] == "Low"
    ).sum()

    col1, col2 = st.columns(2)

    col1.metric(
        "Highest Reported Crime",
        highest_crime["District"],
        f"{int(highest_crime['All Reported']):,} cases"
    )

    col2.metric(
        "Lowest Reported Crime",
        lowest_crime["District"],
        f"{int(lowest_crime['All Reported']):,} cases"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("High Relative Crime Volume", int(high_risk_districts))
    col2.metric("Moderate Relative Crime Volume", int(moderate_risk_districts))
    col3.metric("Low Relative Crime Volume", int(low_risk_districts))

    st.subheader("Punjab District Risk Distribution")

    risk_summary = crime_df["Risk_Level"].value_counts().reset_index()

    risk_summary.columns = [
        "Risk Level",
        "Number of Districts"
    ]

    fig_risk = px.pie(
        risk_summary,
        names="Risk Level",
        values="Number of Districts",
        hole=0.45,
        color="Risk Level",
        color_discrete_map={
            "Low": "#20B8B0",
            "Moderate": "#F4B942",
            "High": "#E45B64"
        },
        title="Relative Crime Volume Across Available Districts"
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )

    st.subheader("District-wise Reported Crime")

    fig = px.bar(
        crime_df.sort_values(
            "All Reported",
            ascending=False
        ),
        x="District",
        y="All Reported",
        color="All Reported",
        title="Reported Crime by District - 2025"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Punjab Traffic Accident Trend")

    fig2 = px.line(
        punjab_df.sort_values("Year"),
        x="Year",
        y="Total_Accidents",
        markers=True,
        title="Yearly Traffic Accidents in Punjab"
    )

    st.plotly_chart(fig2, use_container_width=True)


# =====================================
# PAGE 2: CRIME INTELLIGENCE
# =====================================

elif page == "Crime Intelligence":

    st.header("Crime Intelligence")

    st.write(
        "Explore 2025 crime statistics across available districts."
    )

    selected_districts = st.multiselect(
        "Select Districts",
        sorted(crime_df["District"].unique()),
        default=sorted(crime_df["District"].unique())
    )

    crime_filtered = crime_df[
        crime_df["District"].isin(selected_districts)
    ]

    crime_columns = [
        col for col in crime_df.select_dtypes(
            include="number"
        ).columns
        if col not in ["Latitude", "Longitude"]
    ]

    selected_metric = st.selectbox(
        "Choose Crime Category",
        crime_columns,
        index=crime_columns.index("All Reported")
    )

    if not crime_filtered.empty:
        total_crimes = crime_filtered["All Reported"].sum()

        district_count = crime_filtered["District"].nunique()

        highest_district = crime_filtered.loc[
            crime_filtered["All Reported"].idxmax(),
            "District"
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Reported Crimes",
            f"{int(total_crimes):,}"
        )

        col2.metric(
            "Selected Districts",
            district_count
        )

        col3.metric(
            "Highest Crime District",
            highest_district
        )

        fig = px.bar(
            crime_filtered.sort_values(
                selected_metric,
                ascending=False
            ),
            x="District",
            y=selected_metric,
            color=selected_metric,
            title=f"{selected_metric} by District"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        st.subheader("Crime Category Distribution")

        category_columns = [
            col for col in crime_filtered.select_dtypes(
                include="number"
            ).columns
            if col not in ["Latitude", "Longitude", "All Reported"]
            and crime_filtered[col].sum() > 0
        ]

        if category_columns:

            category_totals = (
                crime_filtered[category_columns]
                .sum()
                .reset_index()
            )

            category_totals.columns = [
                "Crime Category",
                "Reported Cases"
            ]

            fig_pie = px.pie(
                category_totals,
                names="Crime Category",
                values="Reported Cases",
                title="Crime Category Distribution",
                hole=0.4
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )
            st.subheader("Top 5 Crime Categories")

        if category_columns:

            top_categories = category_totals.sort_values(
                "Reported Cases",
                ascending=False
            ).head(5)

            fig_top = px.bar(
                top_categories.sort_values(
                    "Reported Cases",
                    ascending=True
                ),
                x="Reported Cases",
                y="Crime Category",
                orientation="h",
                title="Top 5 Reported Crime Categories",
                text="Reported Cases"
            )

            st.plotly_chart(
                fig_top,
                use_container_width=True
            )

        st.subheader("Crime Records")

        st.dataframe(
            crime_filtered,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("Select at least one district.")
elif page == "Crime Trend Analysis":

    st.title("Crime Trend Analysis")
    st.write(
        "Compare reported crime categories between 2023 and 2025 "
        "for districts available in both datasets."
    )

    if CRIME_COMPARISON_FILE.exists():

        comparison_df = pd.read_csv(CRIME_COMPARISON_FILE)

        categories = [
            col.replace("_2023", "")
            for col in comparison_df.columns
            if col.endswith("_2023")
            and f"{col.replace('_2023', '')}_2025" in comparison_df.columns
            and f"{col.replace('_2023', '')}_Change" in comparison_df.columns
        ]

        if categories:

            districts = sorted(comparison_df["District"].dropna().unique())

            selected_districts = st.multiselect(
                "Select Districts",
                districts,
                default=districts
            )

            selected_category = st.selectbox(
                "Select Crime Category",
                categories,
                index=categories.index("Murder") if "Murder" in categories else 0
            )

            filtered_df = comparison_df[
                comparison_df["District"].isin(selected_districts)
            ].copy()

            if not filtered_df.empty:

                st.subheader(f"{selected_category}: 2023 vs 2025")

                chart_df = filtered_df[
                    ["District", f"{selected_category}_2023",
                     f"{selected_category}_2025"]
                ].melt(
                    id_vars="District",
                    var_name="Year",
                    value_name="Reported Cases"
                )

                chart_df["Year"] = chart_df["Year"].str.replace(
                    f"{selected_category}_", "", regex=False
                )

                st.bar_chart(
                    chart_df,
                    x="District",
                    y="Reported Cases",
                    color="Year"
                )

                change_col = f"{selected_category}_Change"

                st.subheader("Crime Change by District")

                st.bar_chart(
                    filtered_df.set_index("District")[[change_col]]
                )

                increase = (filtered_df[change_col] > 0).sum()
                decrease = (filtered_df[change_col] < 0).sum()
                unchanged = (filtered_df[change_col] == 0).sum()

                col1, col2, col3 = st.columns(3)

                col1.metric("Districts with Increase", int(increase))
                col2.metric("Districts with Decrease", int(decrease))
                col3.metric("No Change", int(unchanged))
                st.subheader("District Highlights")

                highest_increase = filtered_df.loc[
                    filtered_df[change_col].idxmax()
                ]

                largest_decrease = filtered_df.loc[
                    filtered_df[change_col].idxmin()
                ]

                col1, col2 = st.columns(2)

                col1.metric(
                    "Highest Increase",
                    highest_increase["District"],
                    f"+{int(highest_increase[change_col])} cases"
                )

                col2.metric(
                    "Largest Decrease",
                    largest_decrease["District"],
                    f"{int(largest_decrease[change_col])} cases"
                )

                st.subheader("Detailed Comparison")

                st.dataframe(
                    filtered_df[
                        ["District",
                         f"{selected_category}_2023",
                         f"{selected_category}_2025",
                         change_col]
                    ],
                    use_container_width=True
                )

            else:
                st.info("Please select at least one district.")

        else:
            st.warning("No comparable crime categories found in the CSV.")

    else:
        st.error("Crime comparison CSV file was not found.")

# =====================================
# PAGE 3: TRAFFIC INTELLIGENCE
# =====================================

elif page == "Traffic Intelligence":

    st.header("Traffic Intelligence")

    st.write(
        "Explore yearly road accident statistics and regional comparisons."
    )

    regions = sorted(
        traffic_df["Region"].dropna().unique()
    )

    selected_region = st.selectbox(
        "Select Region",
        regions,
        index=regions.index("Punjab")
        if "Punjab" in regions else 0
    )

    region_data = traffic_df[
        traffic_df["Region"] == selected_region
    ].sort_values("Year")
    if not region_data.empty:

        latest_region = region_data.iloc[-1]

        st.subheader("Traffic Statistics Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Accidents",
            f"{int(latest_region['Total_Accidents']):,}"
        )

        col2.metric(
            "Fatal Accidents",
            f"{int(latest_region['Fatal']):,}"
        )

        col3.metric(
            "Non-Fatal Accidents",
            f"{int(latest_region['Non_Fatal']):,}"
        )

        st.caption(
            f"Statistics for {selected_region} — {latest_region['Year']}"
        )

    st.subheader(
        f"{selected_region} Accident Trend"
    )

    fig = px.line(
        region_data,
        x="Year",
        y="Total_Accidents",
        markers=True,
        title=f"Total Accidents in {selected_region}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.subheader("Accident Severity Distribution")

    severity_data = pd.DataFrame({
        "Accident Type": ["Fatal", "Non-Fatal"],
        "Cases": [
            int(region_data["Fatal"].sum()),
            int(region_data["Non_Fatal"].sum())
        ]
    })

    fig_severity = px.pie(
        severity_data,
        names="Accident Type",
        values="Cases",
        hole=0.45,
        color="Accident Type",
        color_discrete_map={
            "Fatal": "#E45B64",
            "Non-Fatal": "#20B8B0"
        },
        title=f"Accident Severity - {selected_region}"
    )

    st.plotly_chart(
        fig_severity,
        use_container_width=True
    )

    st.subheader("Fatal vs Non-Fatal Accidents")

    fig2 = px.bar(
        region_data,
        x="Year",
        y=["Fatal", "Non_Fatal"],
        barmode="group",
        title="Fatal and Non-Fatal Accidents"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Regional Comparison")

    years = sorted(
        traffic_df["Year"].dropna().unique()
    )

    comparison_year = st.selectbox(
        "Choose Financial Year",
        years,
        index=len(years) - 1
    )

    comparison_data = traffic_df[
        traffic_df["Year"] == comparison_year
    ]

    fig3 = px.bar(
        comparison_data,
        x="Region",
        y="Total_Accidents",
        color="Region",
        title=f"Regional Comparison - {comparison_year}"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.subheader("Traffic Records")

    st.dataframe(
        region_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Five regions are included. A separate Pakistan aggregate "
        "is not present in this regional CSV."
    )


# =====================================
# PAGE 4: INTERACTIVE SAFETY MAP
# =====================================

elif page == "Interactive Safety Map":

    st.header("Punjab Risk Intelligence Map")

    st.write(
        "Explore districts according to their relative reported-crime volume."
    )

    st.markdown("""
    🔴 **High** &nbsp;&nbsp; 🟡 **Moderate** &nbsp;&nbsp; 🟢 **Low**
    """)

    map_districts = st.multiselect(
        "Select Districts",
        sorted(crime_df["District"].unique()),
        default=sorted(crime_df["District"].unique())
    )

    map_df = crime_df[
        crime_df["District"].isin(map_districts)
    ].copy()

    safety_map = folium.Map(
        location=[31.1704, 72.7097],
        zoom_start=7,
        tiles="OpenStreetMap"
    )

    risk_colors = {
        "High": "red",
        "Moderate": "orange",
        "Low": "green"
    }

    for _, row in map_df.iterrows():

        district_name = html.escape(
            str(row["District"])
        )

        risk_level = row["Risk_Level"]

        marker_color = risk_colors.get(
            risk_level,
            "blue"
        )

        crime_index = row["Crime_Index"]

        reported_crime = row["All Reported"]

        popup_html = f"""
        <div style="width:220px; font-family:Arial;">

            <h3 style="color:#172B4D;">
                {district_name}
            </h3>

            <hr>

            <b>Relative Crime Index:</b>
            {crime_index}/100
            <br><br>

            <b>Category:</b>
            {risk_level}
            <br><br>

            <b>Reported Crime:</b>
            {reported_crime:,.0f}
            <br><br>

            <b>Year:</b> 2025

        </div>
        """

        folium.CircleMarker(
            location=[
                row["Latitude"],
                row["Longitude"]
            ],
            radius=12,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.75,
            weight=2,
            popup=folium.Popup(
                popup_html,
                max_width=300
            ),
            tooltip=f"{district_name} - {risk_level}"
        ).add_to(safety_map)

    if not map_df.empty:

        st_folium(
            safety_map,
            height=600,
            use_container_width=True,
            key="risk_intelligence_map"
        )

    else:

        st.warning(
            "Please select at least one district."
        )

    st.caption(
        "Map markers represent approximate district centres. "
        "Colours indicate relative reported-crime volume, "
        "not verified real-world danger levels."
    )
# =====================================
# DISTRICT RISK ANALYSIS PAGE
# =====================================

elif page == "District Risk Analysis":

    st.header("District Risk Analysis")

    st.write(
        "Compare districts using the relative reported-crime index."
    )

    st.info(
        "This is a relative index based on reported crime totals "
        "in the available 2025 dataset. It is not a prediction "
        "or a population-adjusted safety measurement."
    )

    # Sort districts by crime index
    risk_data = crime_df.sort_values(
        "Crime_Index",
        ascending=False
    )

    # Summary cards
    high_count = (
        risk_data["Risk_Level"] == "High"
    ).sum()

    moderate_count = (
        risk_data["Risk_Level"] == "Moderate"
    ).sum()

    low_count = (
        risk_data["Risk_Level"] == "Low"
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("High Relative Crime Volume", int(high_count))
    col2.metric("Moderate Relative Crime Volume", int(moderate_count))
    col3.metric("Low Relative Crime Volume", int(low_count))

    st.divider()
    # Highest and lowest relative crime index districts

    highest_risk = risk_data.iloc[0]
    lowest_risk = risk_data.iloc[-1]

    st.subheader("District Risk Highlights")

    col1, col2 = st.columns(2)

    col1.metric(
        "Highest Relative Crime Index",
        highest_risk["District"],
        f"{highest_risk['Crime_Index']}/100"
    )

    col2.metric(
        "Lowest Relative Crime Index",
        lowest_risk["District"],
        f"{lowest_risk['Crime_Index']}/100"
    )
    # Ranking chart
    st.subheader("District Crime Index Ranking")

    fig = px.bar(
        risk_data,
        x="District",
        y="Crime_Index",
        color="Risk_Level",
        text="Crime_Index",
        color_discrete_map={
            "Low": "#20B8B0",
            "Moderate": "#F4B942",
            "High": "#E45B64"
        },
        title="Relative Crime Volume Index (0–100)"
    )

    fig.update_layout(
        yaxis_title="Relative Crime Index",
        xaxis_title="District",
        yaxis_range=[0, 110]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()
    st.subheader("Risk Distribution")

    risk_counts = risk_data["Risk_Level"].value_counts().reset_index()

    risk_counts.columns = ["Risk Level", "District Count"]

    fig_risk = px.pie(
        risk_counts,
        names="Risk Level",
        values="District Count",
        hole=0.4,
        color="Risk Level",
        color_discrete_map={
            "Low": "#20B8B0",
            "Moderate": "#F4B942",
            "High": "#E45B64"
        },
        title="Districts by Relative Crime Volume"
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )

    # Select a district
    st.subheader("Explore a District")

    selected_district = st.selectbox(
        "Choose District",
        sorted(risk_data["District"].unique())
    )

    district_info = risk_data[
        risk_data["District"] == selected_district
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Crime Index",
        f"{district_info['Crime_Index']}/100"
    )

    col2.metric(
        "Risk Category",
        district_info["Risk_Level"]
    )

    col3.metric(
        "Reported Crime",
        f"{int(district_info['All Reported']):,}"
    )

    st.subheader("District Details")

    details = {
        "District": district_info["District"],
        "Reported Crime": int(district_info["All Reported"]),
        "Murder": int(district_info["Murder"]),
        "Robbery": int(district_info["Robbery"]),
        "Burglary": int(district_info["Burglary"]),
        "Relative Crime Index": district_info["Crime_Index"],
        "Relative Category": district_info["Risk_Level"]
    }

    st.dataframe(
        pd.DataFrame([details]),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "High, Moderate and Low describe relative positions among "
        "the seven districts in this dataset, not absolute danger levels."
    )
# =====================================
# PAGE: AI TRAFFIC PREDICTIONS
# =====================================

elif page == "AI Traffic Predictions":

    st.header("AI Traffic Predictions")

    st.write(
        "Explore experimental machine-learning estimates "
        "for traffic accidents in the next financial year."
    )

    if not PREDICTION_FILE.exists():

        st.error(
            "Prediction file not found. "
            "Please run python train_model.py first."
        )

    else:

        try:

            prediction_df = pd.read_csv(
                PREDICTION_FILE,
                encoding="utf-8-sig"
            )

            required_columns = {
                "Region",
                "Predicted_Accidents",
                "Financial_Year"
            }

            if not required_columns.issubset(
                prediction_df.columns
            ):

                st.error(
                    "Prediction file is missing required columns."
                )

            else:

                prediction_df["Predicted_Accidents"] = (
                    pd.to_numeric(
                        prediction_df["Predicted_Accidents"],
                        errors="coerce"
                    )
                )

                prediction_df = prediction_df.dropna(
                    subset=["Region", "Predicted_Accidents"]
                )

                if prediction_df.empty:

                    st.warning("No valid predictions available.")

                else:

                    prediction_year = (
                        prediction_df["Financial_Year"].iloc[0]
                    )

                    total_predicted = int(
                        prediction_df["Predicted_Accidents"].sum()
                    )

                    highest_region = prediction_df.loc[
                        prediction_df["Predicted_Accidents"].idxmax()
                    ]

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Regions Predicted",
                        len(prediction_df)
                    )

                    col2.metric(
                        f"Estimated Accidents ({prediction_year})",
                        f"{total_predicted:,}"
                    )

                    col3.metric(
    "Highest Estimate",
    str(highest_region["Region"]),
    f"{int(highest_region['Predicted_Accidents']):,} accidents"
)

                    st.divider()
                    # Prediction Highlights

                    st.subheader("Prediction Highlights")

                    lowest_region = prediction_df.loc[
                        prediction_df["Predicted_Accidents"].idxmin()
                    ]

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Highest Predicted Accidents",
                        highest_region["Region"],
                        f"{int(highest_region['Predicted_Accidents']):,} cases"
                    )

                    col2.metric(
                        "Lowest Predicted Accidents",
                        lowest_region["Region"],
                        f"{int(lowest_region['Predicted_Accidents']):,} cases"
                    )

                    # Region-wise prediction chart

                    st.subheader(
                        f"Predicted Accidents by Region - {prediction_year}"
                    )

                    fig = px.bar(
                        prediction_df.sort_values(
                            "Predicted_Accidents",
                            ascending=False
                        ),
                        x="Region",
                        y="Predicted_Accidents",
                        color="Predicted_Accidents",
                        text="Predicted_Accidents",
                        title="AI-Based Regional Traffic Estimates"
                    )
                    # Prediction Distribution

                    st.subheader("Prediction Distribution")

                    fig_distribution = px.pie(
                        prediction_df,
                        names="Region",
                        values="Predicted_Accidents",
                        hole=0.45,
                        title="Share of Predicted Accidents by Region"
                    )

                    st.plotly_chart(
                        fig_distribution,
                        use_container_width=True
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.divider()
                    st.subheader("Detailed Prediction Records")

                    display_df = prediction_df[
                        ["Region", "Financial_Year", "Predicted_Accidents"]
                    ].copy()

                    display_df["Predicted_Accidents"] = (
                        display_df["Predicted_Accidents"].astype(int)
                    )

                    st.dataframe(
                        display_df.sort_values(
                            "Predicted_Accidents",
                            ascending=False
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "These values are experimental machine-learning estimates, "
                        "not guaranteed future accident counts."
                    )

                    # Historical vs predicted comparison

                    st.subheader(
                        "Historical vs AI Prediction"
                    )

                    historical_year = "2023-24"

                    historical_data = traffic_df[
                        traffic_df["Year"] == historical_year
                    ][["Region", "Total_Accidents"]].copy()

                    if not historical_data.empty:

                        historical_data = historical_data.rename(
                            columns={
                                "Total_Accidents": "Accidents"
                            }
                        )

                        historical_data["Type"] = (
                            f"Actual ({historical_year})"
                        )

                        predicted_comparison = prediction_df[
                            ["Region", "Predicted_Accidents"]
                        ].copy()

                        predicted_comparison = (
                            predicted_comparison.rename(
                                columns={
                                    "Predicted_Accidents": "Accidents"
                                }
                            )
                        )

                        predicted_comparison["Type"] = (
                            f"AI Estimate ({prediction_year})"
                        )

                        comparison = pd.concat(
                            [
                                historical_data,
                                predicted_comparison
                            ],
                            ignore_index=True
                        )

                        fig2 = px.bar(
                            comparison,
                            x="Region",
                            y="Accidents",
                            color="Type",
                            barmode="group",
                            title=(
                                "Actual 2023-24 vs "
                                f"Estimated {prediction_year}"
                            )
                        )

                        st.plotly_chart(
                            fig2,
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Historical comparison data is unavailable."
                        )

                    st.divider()

                    # Prediction records

                    st.subheader("Prediction Records")

                    st.dataframe(
                        prediction_df[
                            [
                                "Region",
                                "Financial_Year",
                                "Predicted_Accidents"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.info(
                        "These estimates were generated using a "
                        "Random Forest machine-learning model trained "
                        "on historical regional traffic data."
                    )

                    st.warning(
                        "Important limitation: The model uses a small "
                        "historical dataset. These are experimental "
                        "estimates, not confirmed accident counts or "
                        "reliable forecasts. Actual results may differ."
                    )

        except Exception as error:

            st.error("Could not load prediction data.")

            st.exception(error)
        st.divider()
# MODEL EVALUATION

        st.subheader("Model Performance & Evaluation")

        evaluation_file = (
            BASE_DIR / "data" / "model_evaluation.csv"
        )

        if evaluation_file.exists():

            evaluation_df = pd.read_csv(evaluation_file)
            evaluation = evaluation_df.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Training Records",
                int(evaluation["Training_Records"])
            )

            col2.metric(
                "Testing Records",
                int(evaluation["Testing_Records"])
            )

            col3.metric(
                "Mean Absolute Error (MAE)",
                evaluation["Mean_Absolute_Error"]
            )
            st.divider()

            st.subheader("Model Evaluation Highlights")

            mae_value = float(evaluation["Mean_Absolute_Error"])

            col1, col2 = st.columns(2)

            col1.metric(
                "Model Error (MAE)",
                f"{mae_value:.2f} accidents"
            )

            col2.metric(
                "Evaluation Status",
                "Experimental"
            )

            st.subheader("Model Information")

            col1, col2 = st.columns(2)

            col1.info(
                f"Machine Learning Model: {evaluation['Model']}"
            )

            col2.info(
                f"Test Year: {int(evaluation['Test_Year'])}"
            )

            st.info(
                "MAE (Mean Absolute Error) measures the average "
                "absolute difference between actual and predicted "
                "accident counts."
            )

            st.warning(
                f"This model was evaluated using only "
                f"{int(evaluation['Testing_Records'])} testing records. "
                "Results are experimental and should not be treated "
                "as guaranteed prediction accuracy."
            )

        else:

            st.warning(
                "Model evaluation file not found. "
                "Run python train_model.py first."
            )


elif page == "SafeSight Vision":

    st.header("SafeSight Vision")
    st.write(
        "Upload a CCTV image to begin visual enhancement "
        "and image analysis."
    )

    st.subheader("Upload CCTV Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image_bytes = np.frombuffer(
            uploaded_file.getvalue(),
            np.uint8
        )

        original_image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if original_image is not None:

            original_rgb = cv2.cvtColor(
                original_image,
                cv2.COLOR_BGR2RGB
            )
            st.subheader("Image Enhancement Controls")
            if st.button("Reset All Controls"):

                st.session_state["vision_brightness"] = 1.0
                st.session_state["vision_contrast"] = 1.0
                st.session_state["vision_sharpness"] = 0.8
                st.session_state["vision_noise"] = 5

            brightness = st.slider(
                "Brightness",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.1,
                key="vision_brightness"

            )
            contrast = st.slider(
                "Contrast",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.1,
                key="vision_contrast"
            )
            sharpness = st.slider(
                "Sharpness",
                min_value=0.0,
                max_value=2.0,
                value=0.8,
                step=0.1,
                key="vision_sharpness"
            )
            noise_reduction = st.slider(
                "Noise Reduction",
                min_value=1,
                max_value=15,
                value=5,
                step=2,
                key="vision_noise"
            )
            st.subheader("Image Information")

            file_size = len(uploaded_file.getvalue()) / 1024

            image_height, image_width = original_image.shape[:2]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Image Width", f"{image_width} px")

            with col2:
                st.metric("Image Height", f"{image_height} px")

            with col3:
                st.metric("File Size", f"{file_size:.2f} KB")

            st.caption(
                f"File Name: {uploaded_file.name} | "
                f"Format: {uploaded_file.type}"
            )

            st.subheader("Original CCTV Image")

            st.image(
                original_rgb,
                caption="Uploaded Original Image",
                use_container_width=True
            )
            # Apply Brightness Enhancement

            brightened_image = cv2.convertScaleAbs(
                original_rgb,
                alpha=brightness,
                beta=0
            )

            st.subheader("Brightness Enhanced Image")

            st.image(
                brightened_image,
                caption="Brightness Adjusted Result",
                use_container_width=True
            )
            # Apply Contrast Enhancement

            contrast_image = cv2.convertScaleAbs(
                brightened_image,
                alpha=contrast,
                beta=0
            )

            st.subheader("Contrast Enhanced Image")

            st.image(
                contrast_image,
                caption="Brightness and Contrast Adjusted",
                use_container_width=True
            )
            # Apply Image Sharpening

            blurred_image = cv2.GaussianBlur(
                contrast_image,
                (0, 0),
                3
            )

            sharpened_image = cv2.addWeighted(
                contrast_image,
                1 + sharpness,
                blurred_image,
                -sharpness,
                0
            )

            st.subheader("Sharpened CCTV Image")

            st.image(
                sharpened_image,
                caption="Brightness, Contrast and Sharpness Enhanced",
                use_container_width=True
            )
            # Apply Noise Reduction

            denoised_image = cv2.fastNlMeansDenoisingColored(
                sharpened_image,
                None,
                noise_reduction,
                noise_reduction,
                7,
                21
            )

            st.subheader("Noise Reduced CCTV Image")

            st.image(
                denoised_image,
                caption="Enhanced and Noise Reduced Image",
                use_container_width=True
            )
            st.divider()

            st.subheader("Before & After Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    original_rgb,
                    caption="Original CCTV Image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    denoised_image,
                    caption="Final Enhanced Image",
                    use_container_width=True
                )
                # Download Enhanced Image

            success, encoded_image = cv2.imencode(
                ".png",
                denoised_image
            )

            if success:

                st.download_button(
                    label="Download Enhanced Image",
                    data=encoded_image.tobytes(),
                    file_name="SafeSight_Enhanced_Image.png",
                    mime="image/png"
                )

        else:
            st.error("Unable to read this image.")
  

# =====================================
# PAGE 5: ABOUT DATA
# =====================================

elif page == "About Data":

    st.header("About SafeSight AI")

    st.write("""
    SafeSight AI is a student-built safety intelligence
    platform designed to explore crime and traffic
    accident statistics using available datasets.
    """)

    st.subheader("Data Coverage")
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Crime Districts",
        crime_df["District"].nunique()
    )

    col2.metric(
        "Traffic Regions",
        traffic_df["Region"].nunique()
    )

    col3.metric(
        "Traffic Financial Years",
        traffic_df["Year"].nunique()
    )

    st.write(
        f"Crime data: {crime_df['District'].nunique()} districts, 2025."
    )

    st.write(
        f"Traffic data: {traffic_df['Region'].nunique()} regions, "
        f"{traffic_df['Year'].nunique()} financial years."
    )

    st.subheader("Files Used")
    st.info(
        "SafeSight AI uses structured CSV datasets to explore "
        "district-level crime statistics, regional traffic accidents, "
        "and experimental machine-learning predictions."
    )

    st.caption(
        "The available datasets have different coverage periods "
        "and geographic coverage. Comparisons should be interpreted "
        "within each dataset's limitations."
    )

    st.write("""
    - district_crime_2025.csv
    - crime_2023.csv
    - crime_comparison_2023_2025.csv
    - punjab_traffic_yearly.csv
    - traffic_accidents_yearly.csv
    - traffic_predictions.csv
    - model_evaluation.csv
    """)

    st.subheader("Important Limitations")

    st.warning("""
    1. Crime data covers seven districts only.
    2. Map markers show approximate district centres.
    3. Traffic records are yearly regional statistics.
    4. No separate Pakistan aggregate is available in the regional CSV.
    5. Machine learning predictions are experimental and should not be treated as
      confirmed data.
    """)

    st.subheader("Future AI Features")

    roadmap = [
        "Advanced machine-learning prediction models",
        "CCTV image enhancement using computer vision",
        "More verified district-level datasets",
        "Improved model evaluation and validation",
        "Interactive safety analytics"
    ]

    for index, feature in enumerate(roadmap, start=1):
        st.write(f"**{index}.** {feature}")


# =====================================
# FOOTER
# =====================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "SafeSight AI | Together for a Safer Punjab"
)