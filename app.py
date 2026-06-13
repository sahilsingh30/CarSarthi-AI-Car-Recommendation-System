import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from recommendation_engine import get_recommendations, generate_explanation, load_data, train_model
from emi_calculator import calculate_emi
from ownership_cost import calculate_ownership_cost

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarSarthi – Smart Car Finder",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1724 0%, #1a2942 100%);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label { color: #94a3b8 !important; font-size: 0.8rem !important; }

/* Hero */
.hero-section {
    background: linear-gradient(135deg, #0f1724 0%, #1e3a5f 50%, #0f4c81 100%);
    border-radius: 20px;
    padding: 60px 40px;
    text-align: center;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:0.5} 50%{opacity:1} }
.hero-title {
    font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #34d399, #60a5fa);
    background-size: 200%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin: 0; position: relative; z-index: 1;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
.hero-subtitle { color: #94a3b8; font-size: 1.15rem; margin-top: 12px; position: relative; z-index: 1; }

/* Cards */
.car-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}
.car-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
.rank-badge {
    position: absolute; top: 16px; right: 16px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white; font-weight: 700; font-size: 0.75rem;
    padding: 4px 12px; border-radius: 20px;
}
.car-name { font-size: 1.4rem; font-weight: 700; color: #f1f5f9; margin: 0 0 4px; }
.car-brand { color: #60a5fa; font-size: 0.9rem; font-weight: 500; }
.price-tag { font-size: 1.6rem; font-weight: 800; color: #34d399; margin: 12px 0 4px; }
.price-onroad { color: #64748b; font-size: 0.8rem; }

/* Score bars */
.score-label { color: #94a3b8; font-size: 0.75rem; margin-bottom: 3px; }
.score-bar-bg { background: #1e293b; border-radius: 4px; height: 8px; }
.score-bar { height: 8px; border-radius: 4px; }

/* Feature chips */
.feature-chip {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.4);
    color: #93c5fd;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; margin: 3px 2px;
}
.feature-chip.missing {
    background: rgba(100,116,139,0.1);
    border-color: #334155; color: #475569;
}

/* Section header */
.section-header {
    font-size: 1.6rem; font-weight: 700; color: #f1f5f9;
    border-left: 4px solid #3b82f6;
    padding-left: 14px; margin: 30px 0 20px;
}

/* Metric cards */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 800; color: #34d399; }
.metric-label { color: #64748b; font-size: 0.8rem; margin-top: 4px; }

/* Page background */
.stApp { background-color: #0a0f1e; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar nav ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 CarSarthi")
    st.markdown("*India's Smart Car Advisor*")
    st.divider()
    page = st.radio("Navigation", [
        "🏠 Home",
        "🔍 Get Recommendations",
        "📊 Compare Cars",
        "🧮 EMI Calculator",
        "💰 Ownership Cost",
        "📈 Analytics Dashboard",
    ])
    st.divider()
    st.caption("Built with ❤️ using Python, Streamlit & KNN")

# ── Pre-train model ────────────────────────────────────────────────────────────
@st.cache_resource
def init_model():
    return train_model()

init_model()
df_all = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🚗 CarSarthi</div>
        <div class="hero-subtitle">India's Smartest Car Recommendation System for Every Budget</div>
        <div style="margin-top:20px;color:#475569;font-size:0.9rem;">
            Powered by KNN Machine Learning · 103+ Indian Cars · Real-Time Scoring
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">103+</div><div class="metric-label">Indian Cars</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">15+</div><div class="metric-label">Brands Covered</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">KNN</div><div class="metric-label">AI Matching</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">₹3.5L+</div><div class="metric-label">Budget Range</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    steps = [
        ("1️⃣", "Tell Us Your Needs", "Budget, family size, fuel preference, and must-have features"),
        ("2️⃣", "AI Filters Cars", "KNN algorithm finds the closest matches from 103+ cars"),
        ("3️⃣", "Smart Scoring", "40% Budget + 20% Mileage + 20% Safety + 10% Maintenance + 10% Resale"),
        ("4️⃣", "Get Top 5", "Ranked recommendations with detailed explanations and cost breakdowns"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], steps):
        with col:
            st.markdown(f"""<div class="car-card" style="text-align:center;padding:20px">
                <div style="font-size:2rem">{icon}</div>
                <div style="color:#60a5fa;font-weight:700;margin:8px 0">{title}</div>
                <div style="color:#64748b;font-size:0.82rem">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Cars by Segment</div>', unsafe_allow_html=True)
    seg_counts = df_all['body_type'].value_counts()
    fig = px.pie(values=seg_counts.values, names=seg_counts.index,
                 color_discrete_sequence=px.colors.sequential.Blues_r,
                 hole=0.45)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#94a3b8', showlegend=True, height=350,
                      legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Get Recommendations":
    st.markdown('<div class="section-header">🔍 Find Your Perfect Car</div>', unsafe_allow_html=True)

    with st.form("recommendation_form"):
        st.markdown("#### 💰 Financial Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            budget = st.number_input("Maximum Budget (₹)", min_value=300000, max_value=10000000,
                                      value=1000000, step=50000, format="%d")
        with c2:
            down_payment = st.number_input("Down Payment (₹)", min_value=0, max_value=5000000,
                                            value=200000, step=10000, format="%d")
        with c3:
            emi_range = st.number_input("Preferred EMI Range (₹/month)", min_value=0, max_value=100000,
                                         value=15000, step=1000, format="%d")

        st.markdown("#### 👨‍👩‍👧‍👦 Personal Usage")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            family_size = st.selectbox("Family Size", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        with c2:
            daily_km = st.number_input("Daily Driving (km)", min_value=5, max_value=300, value=40)
        with c3:
            annual_km = st.number_input("Annual Running (km)", min_value=5000, max_value=100000, value=15000, step=1000)
        with c4:
            city_type = st.selectbox("City Type", ["Urban", "Semi-Urban", "Rural"])

        st.markdown("#### 🚗 Vehicle Preferences")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            fuel_type = st.selectbox("Fuel Type", ["Any", "Petrol", "Diesel", "CNG", "Electric"])
        with c2:
            transmission = st.selectbox("Transmission", ["Any", "Manual", "Automatic"])
        with c3:
            body_type = st.selectbox("Body Type", ["Any", "Hatchback", "Sedan", "Compact SUV", "SUV", "MPV"])
        with c4:
            car_condition = st.selectbox("Condition", ["New", "Used"])

        st.markdown("#### ⭐ Your Priorities (Rate 1–5)")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: mileage_imp = st.slider("Mileage", 1, 5, 3)
        with c2: safety_imp = st.slider("Safety", 1, 5, 4)
        with c3: maint_imp = st.slider("Maintenance", 1, 5, 3)
        with c4: comfort_imp = st.slider("Comfort", 1, 5, 3)
        with c5: resale_imp = st.slider("Resale Value", 1, 5, 3)

        st.markdown("#### ✨ Required Features")
        all_features = ["Touchscreen", "Sunroof", "Rear Camera", "360 Camera",
                        "Cruise Control", "Wireless Android Auto", "ADAS", "Rear AC Vents"]
        required_features = st.multiselect("Select must-have features", all_features,
                                            default=["Touchscreen", "Rear Camera"])

        submitted = st.form_submit_button("🔍 Find My Perfect Car", use_container_width=True,
                                           type="primary")

    if submitted:
        user_input = {
            'budget': budget, 'down_payment': down_payment, 'preferred_emi': emi_range,
            'family_size': family_size, 'daily_km': daily_km, 'annual_km': annual_km,
            'city_type': city_type, 'fuel_type': fuel_type, 'transmission': transmission,
            'body_type': body_type, 'car_condition': car_condition,
            'mileage_importance': mileage_imp, 'safety_importance': safety_imp,
            'maintenance_importance': maint_imp, 'comfort_importance': comfort_imp,
            'resale_importance': resale_imp, 'required_features': required_features,
            'desired_mileage': 20
        }
        st.session_state['user_input'] = user_input

        with st.spinner("🤖 AI is analysing 103+ cars for you..."):
            results, error = get_recommendations(user_input)

        if error:
            st.error(error)
        elif results.empty:
            st.warning("No cars found. Try relaxing your filters.")
        else:
            st.success(f"✅ Found {len(results)} perfect matches for you!")
            st.session_state['results'] = results

            feature_map = {
                'Touchscreen': 'has_touchscreen', 'Sunroof': 'has_sunroof',
                'Rear Camera': 'has_rear_camera', '360 Camera': 'has_360_camera',
                'Cruise Control': 'has_cruise_control', 'Wireless Android Auto': 'has_wireless_aa',
                'ADAS': 'has_adas', 'Rear AC Vents': 'has_rear_ac'
            }

            for i, (_, car) in enumerate(results.iterrows()):
                rank_labels = ["🥇 #1 Best Match", "🥈 #2 Runner-up", "🥉 #3 Great Pick", "⭐ #4 Good Option", "✅ #5 Consider This"]
                score_pct = min(100, car.get('recommendation_score', 0))
                budget_match = min(100, max(0, 100 - abs(budget - car['price']) / budget * 100))

                feat_html = ""
                for feat, col in feature_map.items():
                    has = car.get(col, 0) == 1
                    css = "feature-chip" if has else "feature-chip missing"
                    icon = "✓" if has else "✗"
                    feat_html += f'<span class="{css}">{icon} {feat}</span>'

                reasons = generate_explanation(car.to_dict(), user_input)
                reasons_html = "".join(f"<li style='color:#94a3b8;font-size:0.82rem;margin:4px 0'>{r}</li>" for r in reasons)

                st.markdown(f"""
                <div class="car-card">
                    <span class="rank-badge">{rank_labels[i]}</span>
                    <div class="car-name">{car['car_name']}</div>
                    <div class="car-brand">by {car['brand']} · {car['body_type']} · {car['fuel_type']} · {car['transmission']}</div>
                    <div class="price-tag">₹{car['price']:,.0f}</div>
                    <div class="price-onroad">On-road ~₹{car['on_road_price']:,.0f} · EMI ~₹{car['emi_estimate']:,.0f}/mo</div>
                    <div style="margin:16px 0">{feat_html}</div>
                </div>""", unsafe_allow_html=True)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**Why this car?**")
                    st.markdown(f"<ul style='padding-left:16px'>{reasons_html}</ul>", unsafe_allow_html=True)

                with col2:
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Score", f"{score_pct:.0f}/100")
                        st.metric("Safety", f"{'⭐'*int(car['safety_rating'])}")
                    with m2:
                        st.metric("Budget Match", f"{budget_match:.0f}%")
                        st.metric("Mileage", f"{car['mileage']} km/l" if car['mileage'] > 0 else "Electric")

                # Score breakdown bar chart
                breakdown = {
                    'Budget Fit': car.get('budget_fit', 0),
                    'Mileage': car.get('mileage_score', 0),
                    'Safety': car.get('safety_score', 0),
                    'Maintenance': car.get('maintenance_score', 0),
                    'Resale': car.get('resale_score_norm', 0),
                }
                fig = go.Figure(go.Bar(
                    x=list(breakdown.values()), y=list(breakdown.keys()),
                    orientation='h',
                    marker_color=['#3b82f6','#34d399','#f59e0b','#8b5cf6','#ec4899'],
                    text=[f"{v:.0f}" for v in breakdown.values()],
                    textposition='inside'
                ))
                fig.update_layout(
                    xaxis_range=[0, 100], height=180,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8', margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COMPARE CARS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Compare Cars":
    st.markdown('<div class="section-header">📊 Compare Cars Side-by-Side</div>', unsafe_allow_html=True)

    all_cars = df_all['car_name'].tolist()
    c1, c2, c3 = st.columns(3)
    with c1: car1 = st.selectbox("Car 1", all_cars, index=0)
    with c2: car2 = st.selectbox("Car 2", all_cars, index=10)
    with c3: car3 = st.selectbox("Car 3", all_cars, index=30)

    selected = [c for c in [car1, car2, car3] if c]
    cars_df = df_all[df_all['car_name'].isin(selected)].set_index('car_name')

    compare_fields = {
        "Price (₹)": "price", "On-Road Price (₹)": "on_road_price",
        "Fuel Type": "fuel_type", "Mileage (km/l)": "mileage",
        "Transmission": "transmission", "Seating Capacity": "seating_capacity",
        "Safety Rating (/5)": "safety_rating", "Maintenance Cost": "maintenance_cost",
        "Ground Clearance (mm)": "ground_clearance", "Engine (cc)": "engine_cc",
        "Boot Space (L)": "boot_space", "Fuel Tank (L)": "fuel_tank",
        "Resale Score (/10)": "resale_score", "EMI Estimate (₹/mo)": "emi_estimate",
    }

    st.markdown("### 📋 Specification Comparison")
    rows = []
    for label, col in compare_fields.items():
        row = {"Specification": label}
        for car in selected:
            val = cars_df.loc[car, col] if car in cars_df.index else "N/A"
            if col == 'price' and isinstance(val, (int, float)):
                val = f"₹{val:,.0f}"
            elif col == 'on_road_price' and isinstance(val, (int, float)):
                val = f"₹{val:,.0f}"
            elif col == 'emi_estimate' and isinstance(val, (int, float)):
                val = f"₹{val:,.0f}"
            row[car] = val
        rows.append(row)

    compare_table = pd.DataFrame(rows).set_index("Specification")
    st.dataframe(compare_table, use_container_width=True, height=500)

    # Feature comparison
    st.markdown("### ✨ Feature Comparison")
    feature_map = {
        'Touchscreen': 'has_touchscreen', 'Sunroof': 'has_sunroof',
        'Rear Camera': 'has_rear_camera', '360 Camera': 'has_360_camera',
        'Cruise Control': 'has_cruise_control', 'Wireless Android Auto': 'has_wireless_aa',
        'ADAS': 'has_adas', 'Rear AC Vents': 'has_rear_ac'
    }
    feat_rows = []
    for feat, col in feature_map.items():
        row = {"Feature": feat}
        for car in selected:
            val = cars_df.loc[car, col] if car in cars_df.index else 0
            row[car] = "✅" if val == 1 else "❌"
        feat_rows.append(row)
    feat_table = pd.DataFrame(feat_rows).set_index("Feature")
    st.dataframe(feat_table, use_container_width=True)

    # Radar chart
    st.markdown("### 📡 Performance Radar")
    radar_metrics = ['safety_rating', 'resale_score', 'mileage', 'seating_capacity']
    radar_labels = ['Safety (0-5)', 'Resale (0-10)', 'Mileage (km/l)', 'Seating']
    colors = ['#3b82f6', '#34d399', '#f59e0b']
    fig = go.Figure()
    for i, car in enumerate(selected):
        if car in cars_df.index:
            vals = [float(cars_df.loc[car, m]) if cars_df.loc[car, m] != 0 else 0.1 for m in radar_metrics]
            # Normalize mileage to 0-10
            vals[2] = vals[2] / 4 if vals[2] > 0 else 2
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=radar_labels + [radar_labels[0]],
                fill='toself', name=car, line_color=colors[i % 3],
                fillcolor=colors[i % 3].replace('#', 'rgba(').replace(')', ',0.1)') if False else None,
                opacity=0.8
            ))
    fig.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)',
                   radialaxis=dict(visible=True, color='#475569'),
                   angularaxis=dict(color='#475569')),
        paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
        height=400, showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EMI CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 EMI Calculator":
    st.markdown('<div class="section-header">🧮 EMI Calculator</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Enter Loan Details")
        car_price = st.number_input("Car Price (₹)", min_value=100000, max_value=10000000,
                                     value=1000000, step=50000, format="%d")
        down_pmt = st.number_input("Down Payment (₹)", min_value=0, max_value=5000000,
                                    value=200000, step=10000, format="%d")
        interest = st.slider("Interest Rate (%)", min_value=6.0, max_value=18.0, value=8.5, step=0.1)
        tenure = st.selectbox("Loan Tenure", [12, 24, 36, 48, 60, 72, 84], index=4,
                               format_func=lambda x: f"{x} months ({x//12} years)")

        result = calculate_emi(car_price, down_pmt, interest, tenure)

    with c2:
        st.markdown("#### Results")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Monthly EMI", f"₹{result['emi']:,.0f}")
            st.metric("Loan Amount", f"₹{result['loan_amount']:,.0f}")
        with m2:
            st.metric("Total Interest", f"₹{result['total_interest']:,.0f}")
            st.metric("Total Payable", f"₹{result['total_amount']:,.0f}")

        # Pie chart: principal vs interest
        fig = go.Figure(go.Pie(
            labels=['Principal', 'Interest'],
            values=[car_price, result['total_interest']],
            hole=0.6,
            marker_colors=['#3b82f6', '#f59e0b'],
            textinfo='label+percent'
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                          height=280, showlegend=False,
                          annotations=[dict(text=f"₹{result['emi']:,.0f}<br>/mo",
                                            x=0.5, y=0.5, font_size=14, showarrow=False,
                                            font_color='#34d399')])
        st.plotly_chart(fig, use_container_width=True)

    # Amortization schedule
    st.markdown("#### 📅 Yearly Amortization")
    balance = result['loan_amount']
    r = interest / 100 / 12
    years_data = []
    for yr in range(1, tenure // 12 + 1):
        yr_principal = 0
        yr_interest = 0
        for _ in range(12):
            if balance <= 0:
                break
            int_pmt = balance * r
            prin_pmt = result['emi'] - int_pmt
            yr_interest += int_pmt
            yr_principal += prin_pmt
            balance -= prin_pmt
        years_data.append({'Year': f'Year {yr}', 'Principal': round(yr_principal), 'Interest': round(yr_interest)})

    amo_df = pd.DataFrame(years_data)
    fig = px.bar(amo_df, x='Year', y=['Principal', 'Interest'],
                 color_discrete_sequence=['#3b82f6', '#f59e0b'],
                 barmode='stack')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#94a3b8', height=300,
                      legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OWNERSHIP COST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Ownership Cost":
    st.markdown('<div class="section-header">💰 5-Year Ownership Cost Calculator</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        car_name = st.selectbox("Select Car", df_all['car_name'].tolist())
        annual_km = st.number_input("Annual Kilometres", min_value=5000, max_value=100000,
                                     value=15000, step=1000)
        fuel_prices = {'Petrol': 103, 'Diesel': 90, 'CNG': 75, 'Electric': 8}
        car_row = df_all[df_all['car_name'] == car_name].iloc[0].to_dict()
        fuel_type = car_row['fuel_type']
        default_fp = fuel_prices.get(fuel_type, 103)
        unit = "₹/kWh" if fuel_type == 'Electric' else "₹/litre" if fuel_type != 'CNG' else "₹/kg"
        fuel_price = st.number_input(f"Fuel Price ({unit})", min_value=1, max_value=500,
                                      value=default_fp)
        st.info(f"**{car_name}**\nPrice: ₹{car_row['price']:,.0f} | {fuel_type} | {car_row['transmission']}")

    costs = calculate_ownership_cost(car_row, annual_km, fuel_price)

    with c2:
        st.markdown("### 5-Year Cost Breakdown")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.metric("Annual Fuel Cost", f"₹{costs['annual_fuel_cost']:,.0f}")
        with r1c2:
            st.metric("Annual Maintenance", f"₹{costs['annual_maintenance']:,.0f}")
        with r1c3:
            st.metric("Monthly EMI", f"₹{costs['monthly_emi']:,.0f}")

        # Stacked bar: 5-year costs
        cost_items = {
            'Fuel (5yr)': costs['total_fuel_5yr'],
            'Insurance (5yr)': costs['total_insurance_5yr'],
            'Maintenance (5yr)': costs['total_maintenance_5yr'],
            'EMI Total': costs['total_emi_5yr'],
        }
        fig = go.Figure(go.Bar(
            x=list(cost_items.values()), y=list(cost_items.keys()),
            orientation='h',
            marker_color=['#34d399', '#f59e0b', '#8b5cf6', '#3b82f6'],
            text=[f"₹{v:,.0f}" for v in cost_items.values()],
            textposition='outside'
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#94a3b8', height=280, margin=dict(l=10, r=100, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        total_with_emi = costs['total_with_emi'] + costs['total_emi_5yr']
        st.markdown(f"""
        <div class="car-card" style="text-align:center">
            <div style="color:#94a3b8;font-size:0.9rem">Total 5-Year Cost of Ownership (excl. car price)</div>
            <div style="font-size:2.5rem;font-weight:800;color:#34d399;margin:8px 0">
                ₹{costs['total_with_emi']:,.0f}
            </div>
            <div style="color:#64748b;font-size:0.8rem">Fuel + Insurance + Maintenance + EMI</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics Dashboard":
    st.markdown('<div class="section-header">📈 Analytics Dashboard</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["💹 Price Analysis", "⛽ Fuel & Mileage", "🛡️ Safety", "🔧 Maintenance"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            price_dist = df_all.copy()
            price_dist['Price Range'] = pd.cut(price_dist['price'],
                bins=[0, 500000, 800000, 1200000, 1800000, 20000000],
                labels=['<5L', '5-8L', '8-12L', '12-18L', '18L+'])
            fig = px.histogram(price_dist, x='price', color='body_type', nbins=30,
                               title='Price Distribution by Body Type',
                               color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#94a3b8', height=350,
                               legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            brand_avg = df_all.groupby('brand')['price'].mean().sort_values().reset_index()
            fig = px.bar(brand_avg, x='price', y='brand', orientation='h',
                         title='Average Price by Brand',
                         color='price', color_continuous_scale='Blues')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#94a3b8', height=400,
                               coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fuel_counts = df_all['fuel_type'].value_counts().reset_index()
            fig = px.pie(fuel_counts, values='count', names='fuel_type',
                         title='Fuel Type Distribution', hole=0.4,
                         color_discrete_sequence=['#3b82f6','#34d399','#f59e0b','#8b5cf6','#ec4899'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                               height=350, legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            mileage_df = df_all[df_all['mileage'] > 0]
            fig = px.scatter(mileage_df, x='price', y='mileage', color='fuel_type',
                             size='safety_rating', hover_data=['car_name'],
                             title='Mileage vs Price (bubble = safety)',
                             color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#94a3b8', height=400,
                               legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        safety_counts = df_all['safety_rating'].value_counts().sort_index().reset_index()
        safety_counts.columns = ['Rating', 'Count']
        safety_counts['Label'] = safety_counts['Rating'].apply(lambda r: '⭐'*int(r))
        fig = px.bar(safety_counts, x='Label', y='Count',
                     title='Safety Rating Distribution',
                     color='Count', color_continuous_scale='Greens')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#94a3b8', height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top Safety Cars")
        top_safe = df_all[df_all['safety_rating'] == 5][['car_name','brand','price','body_type','safety_rating']].head(15)
        st.dataframe(top_safe.style.format({'price': '₹{:,.0f}'}), use_container_width=True)

    with tab4:
        maint_price = df_all.groupby(['maintenance_cost', 'body_type'])['price'].mean().reset_index()
        fig = px.bar(maint_price, x='body_type', y='price', color='maintenance_cost',
                     barmode='group', title='Avg Price by Body Type & Maintenance',
                     color_discrete_map={'Low': '#34d399', 'Medium': '#f59e0b', 'High': '#ef4444'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#94a3b8', height=380,
                           legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

