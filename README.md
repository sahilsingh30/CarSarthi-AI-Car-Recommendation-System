# 🚗 CarSarthi – AI Car Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-KNN-orange?style=flat-square&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-green?style=flat-square&logo=plotly)
![Cars](https://img.shields.io/badge/Cars-103%2B%20Indian-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

*India's Smartest Car Advisor for Middle-Class & Budget Buyers*

</div>

---

## 🎯 Project Objective

CarSarthi is an end-to-end AI-powered car recommendation system that helps middle-class and lower-middle-class users in India find the most suitable car based on their **budget, family size, fuel preferences, safety requirements, mileage expectations, and feature priorities**.

The system recommends the **Top 5 cars** with detailed explanations of why each car was selected.

---

## 🖥️ Features

| Feature | Description |
|---|---|
| 🔍 **AI Recommendations** | KNN + Weighted Scoring Engine finds your top 5 cars |
| 📊 **Car Comparison** | Compare up to 3 cars side-by-side with radar charts |
| 🧮 **EMI Calculator** | Full amortization schedule with interest breakdown |
| 💰 **Ownership Cost** | 5-year cost: Fuel + Insurance + Maintenance + EMI |
| 📈 **Analytics Dashboard** | Price, Safety, Mileage, Fuel distributions |
| 🗃️ **103+ Cars** | Covers all major Indian brands (2023-24 pricing) |

---

## 🧠 How the AI Works

### Step 1 – Rule-Based Filtering
Filters cars based on budget (±10%), fuel type, transmission, body type, and family size.

### Step 2 – KNN Similarity
K-Nearest Neighbors identifies the 20 most similar cars to the user's ideal profile.

### Step 3 – Smart Scoring Engine

```
Recommendation Score =
  40% × Budget Fit Score
+ 20% × Mileage Score       ← weighted by user priority
+ 20% × Safety Score        ← weighted by user priority
+ 10% × Maintenance Score   ← weighted by user priority
+ 10% × Resale Score        ← weighted by user priority
+ Feature Match Bonus (0-10)
```

Priority weights dynamically adjust based on what matters most to you.

### Step 4 – Explainability
Every recommendation includes:
- Why it was chosen
- Budget match percentage
- Safety, mileage, and maintenance scores
- Feature availability breakdown

---

## 🚗 Dataset

103+ cars from the Indian market including:

| Brand | Examples |
|---|---|
| Maruti Suzuki | Alto, Swift, Brezza, Baleno, Ertiga, Jimny |
| Tata | Tiago, Nexon EV, Curvv, Harrier, Safari, Punch |
| Hyundai | i20, Creta EV, Verna, Alcazar, Exter |
| Kia | Sonet, Seltos, Carens, EV6 |
| Mahindra | XUV700, Scorpio-N, Thar, BE 6e |
| Toyota | Glanza, Hyryder, Innova HyCross, Fortuner |
| Honda | Amaze, City Hybrid, Elevate |
| MG | Astor, Comet EV, ZS EV, Hector |

Covers: Hatchbacks · Sedans · Compact SUVs · SUVs · MPVs · EVs

---

## 📁 Project Structure

```
car_rec/
├── app.py                    # Main Streamlit application
├── recommendation_engine.py  # KNN + scoring + explainability
├── emi_calculator.py         # EMI & amortization logic
├── ownership_cost.py         # 5-year ownership cost engine
├── requirements.txt          # Python dependencies
├── models/
│   ├── knn_model.pkl         # Trained KNN model (auto-generated)
│   └── scaler.pkl            # MinMaxScaler (auto-generated)
└── data/
    ├── cars.csv              # 103+ Indian cars dataset
    └── generate_data.py      # Dataset generation script
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/sahilsingh30/carsarthi.git
cd carsarthi
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

> **Note:** The KNN model trains automatically on first run — no separate training step needed.

---

## 📸 App Pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview, stats, how it works |
| 🔍 Recommendations | Full input form → Top 5 AI recommendations |
| 📊 Compare Cars | Side-by-side specs + radar chart |
| 🧮 EMI Calculator | Monthly EMI + amortization chart |
| 💰 Ownership Cost | 5-year total cost breakdown |
| 📈 Analytics | Dataset visualizations & insights |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Algorithm | Scikit-Learn KNN (K=20, Euclidean) |
| Scoring | Custom Weighted Engine |
| Visualization | Plotly (interactive), Matplotlib |
| Data | Pandas, NumPy |
| Model Persistence | Joblib |
| Language | Python 3.10+ |

---

## 👨‍💻 Author

**Sahil Singh** · [@sahilsingh30](https://github.com/sahilsingh30)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
⭐ Star this repo if CarSarthi helped you find your dream car!
</div>
