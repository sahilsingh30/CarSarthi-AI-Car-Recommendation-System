import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'cars.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'knn_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')

FEATURES = ['price', 'mileage', 'safety_rating', 'maintenance_cost_num',
            'resale_score', 'seating_capacity', 'ground_clearance', 'engine_cc']

def load_data():
    return pd.read_csv(DATA_PATH)

def train_model():
    df = load_data()
    X = df[FEATURES].fillna(0)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
    knn.fit(X_scaled)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return knn, scaler

def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_model()
    knn = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return knn, scaler

def rule_based_filter(df, user_input):
    filtered = df.copy()
    # Budget filter with 10% tolerance
    filtered = filtered[filtered['price'] <= user_input['budget'] * 1.10]
    # Fuel type
    if user_input.get('fuel_type') and user_input['fuel_type'] != 'Any':
        filtered = filtered[filtered['fuel_type'] == user_input['fuel_type']]
    # Transmission
    if user_input.get('transmission') and user_input['transmission'] != 'Any':
        filtered = filtered[filtered['transmission'] == user_input['transmission']]
    # Body type
    if user_input.get('body_type') and user_input['body_type'] != 'Any':
        filtered = filtered[filtered['body_type'].str.contains(user_input['body_type'], case=False, na=False)]
    # Family size / seating
    if user_input.get('family_size', 0) > 0:
        min_seats = user_input['family_size']
        if min_seats <= 4:
            filtered = filtered[filtered['seating_capacity'] >= 4]
        elif min_seats <= 5:
            filtered = filtered[filtered['seating_capacity'] >= 5]
        else:
            filtered = filtered[filtered['seating_capacity'] >= 6]
    return filtered

def calculate_scores(df, user_input):
    scored = df.copy()

    # Budget fit (0-100)
    budget = user_input['budget']
    scored['budget_fit'] = scored['price'].apply(
        lambda p: max(0, 100 - abs(budget - p) / budget * 100) if budget > 0 else 50
    )

    # Mileage score — EV gets 85 if user chose electric, else neutral
    max_mileage = scored['mileage'].replace(0, np.nan).max()
    if user_input.get('fuel_type') == 'Electric':
        scored['mileage_score'] = scored['mileage'].apply(lambda m: 85 if m == 0 else (m / max_mileage * 100))
    else:
        scored['mileage_score'] = scored['mileage'].apply(
            lambda m: (m / max_mileage * 100) if m > 0 else 30
        )

    # Safety score (0-100)
    scored['safety_score'] = scored['safety_rating'] / 5 * 100

    # Maintenance score (inverted — lower cost = higher score)
    scored['maintenance_score'] = scored['maintenance_cost_num'].map({1: 100, 2: 60, 3: 20})

    # Resale score (0-100)
    scored['resale_score_norm'] = scored['resale_score'] / 10 * 100

    # Priority weights from user (default equal)
    mileage_w = user_input.get('mileage_importance', 3) / 5
    safety_w = user_input.get('safety_importance', 3) / 5
    maintenance_w = user_input.get('maintenance_importance', 3) / 5
    resale_w = user_input.get('resale_importance', 3) / 5
    comfort_w = user_input.get('comfort_importance', 3) / 5

    total_w = mileage_w + safety_w + maintenance_w + resale_w + comfort_w
    if total_w == 0:
        total_w = 1

    mileage_pct = (mileage_w / total_w) * 30
    safety_pct = (safety_w / total_w) * 25
    maintenance_pct = (maintenance_w / total_w) * 20
    resale_pct = (resale_w / total_w) * 15
    comfort_pct = (comfort_w / total_w) * 10

    # Feature match bonus
    required_features = user_input.get('required_features', [])
    feature_map = {
        'Touchscreen': 'has_touchscreen', 'Sunroof': 'has_sunroof',
        'Rear Camera': 'has_rear_camera', '360 Camera': 'has_360_camera',
        'Cruise Control': 'has_cruise_control', 'Wireless Android Auto': 'has_wireless_aa',
        'ADAS': 'has_adas', 'Rear AC Vents': 'has_rear_ac'
    }
    if required_features:
        def feature_bonus(row):
            matched = sum(row.get(feature_map[f], 0) for f in required_features if f in feature_map)
            return (matched / len(required_features)) * 10
        scored['feature_bonus'] = scored.apply(feature_bonus, axis=1)
    else:
        scored['feature_bonus'] = 5

    # Final recommendation score
    scored['recommendation_score'] = (
        scored['budget_fit'] * 0.40 +
        scored['mileage_score'] * (mileage_pct / 30) * 0.20 +
        scored['safety_score'] * (safety_pct / 25) * 0.20 +
        scored['maintenance_score'] * (maintenance_pct / 20) * 0.10 +
        scored['resale_score_norm'] * (resale_pct / 15) * 0.10 +
        scored['feature_bonus']
    )

    return scored

def get_recommendations(user_input):
    df = load_data()
    knn, scaler = load_model()

    # Step 1: Rule-based filter
    filtered = rule_based_filter(df, user_input)

    if len(filtered) < 3:
        # Relax fuel/transmission filters
        filtered = df[df['price'] <= user_input['budget'] * 1.15]

    if len(filtered) == 0:
        return pd.DataFrame(), "No cars found within your budget. Try increasing your budget."

    # Step 2: KNN similarity
    user_vector = np.array([[
        user_input['budget'],
        user_input.get('desired_mileage', 20),
        user_input.get('safety_importance', 3),
        user_input.get('maintenance_importance', 2),
        user_input.get('resale_importance', 3),
        user_input.get('family_size', 4),
        185,  # avg ground clearance
        1200  # avg engine cc
    ]])
    user_scaled = scaler.transform(user_vector)

    # Get KNN neighbors from the full dataset
    distances, indices = knn.kneighbors(user_scaled)
    knn_car_names = df.iloc[indices[0]]['car_name'].tolist()

    # Boost KNN matches in filtered set
    filtered = filtered.copy()
    filtered['knn_boost'] = filtered['car_name'].apply(lambda x: 5 if x in knn_car_names else 0)

    # Step 3: Score
    scored = calculate_scores(filtered, user_input)
    scored['recommendation_score'] += scored['knn_boost']

    # Step 4: Rank and return top 5
    top5 = scored.sort_values('recommendation_score', ascending=False).head(5).reset_index(drop=True)
    return top5, None

def generate_explanation(row, user_input):
    budget = user_input['budget']
    budget_match = min(100, max(0, 100 - abs(budget - row['price']) / budget * 100))
    reasons = []

    if budget_match >= 85:
        reasons.append(f"✅ Excellent budget fit — {budget_match:.0f}% match to your ₹{budget:,.0f} budget")
    elif budget_match >= 70:
        reasons.append(f"✅ Good budget fit — {budget_match:.0f}% match to your ₹{budget:,.0f} budget")
    else:
        reasons.append(f"⚠️ Slightly over budget — {budget_match:.0f}% match (₹{row['price']:,.0f})")

    if row['fuel_type'] == 'Electric':
        reasons.append(f"⚡ Electric vehicle — zero fuel cost, low maintenance")
    elif row['mileage'] >= 25:
        reasons.append(f"⛽ Excellent mileage: {row['mileage']} km/l — ideal for high daily usage")
    elif row['mileage'] >= 18:
        reasons.append(f"⛽ Good mileage: {row['mileage']} km/l")

    if row['safety_rating'] == 5:
        reasons.append(f"🛡️ 5-star safety rating — top crash test score")
    elif row['safety_rating'] >= 4:
        reasons.append(f"🛡️ {row['safety_rating']}-star safety rating — very safe")

    if row['maintenance_cost'] == 'Low':
        reasons.append(f"🔧 Low maintenance cost — ideal for budget-conscious ownership")

    if row['resale_score'] >= 8:
        reasons.append(f"💰 High resale value (score: {row['resale_score']}/10)")

    if row['seating_capacity'] >= 7:
        reasons.append(f"👨‍👩‍👧‍👦 {row['seating_capacity']}-seater — great for large families")

    required_features = user_input.get('required_features', [])
    feature_map = {'Touchscreen': 'has_touchscreen', 'Sunroof': 'has_sunroof',
                   'Rear Camera': 'has_rear_camera', '360 Camera': 'has_360_camera',
                   'Cruise Control': 'has_cruise_control', 'Wireless Android Auto': 'has_wireless_aa',
                   'ADAS': 'has_adas', 'Rear AC Vents': 'has_rear_ac'}
    matched_features = [f for f in required_features if f in feature_map and row.get(feature_map[f], 0) == 1]
    if matched_features:
        reasons.append(f"✨ Includes your required features: {', '.join(matched_features)}")

    return reasons