def calculate_ownership_cost(car_row, annual_km, fuel_price_per_unit=None):
    fuel_type = car_row.get('fuel_type', 'Petrol')
    mileage = car_row.get('mileage', 15)
    price = car_row.get('price', 0)

    # Fuel prices per unit (₹)
    default_prices = {'Petrol': 103, 'Diesel': 90, 'CNG': 75, 'Electric': 8}
    if fuel_price_per_unit is None:
        fuel_price_per_unit = default_prices.get(fuel_type, 103)

    # Fuel cost per year
    if fuel_type == 'Electric':
        # Electric: ~7 km/kWh on average (Nexon EV), price per kWh
        kwh_per_km = 1 / 7
        annual_fuel_cost = annual_km * kwh_per_km * fuel_price_per_unit
    elif mileage > 0:
        annual_fuel_cost = (annual_km / mileage) * fuel_price_per_unit
    else:
        annual_fuel_cost = 0

    # Insurance (depreciates each year)
    insurance_y1 = price * 0.035
    insurance_y2 = price * 0.025
    insurance_y3 = price * 0.02
    insurance_y4 = price * 0.018
    insurance_y5 = price * 0.016
    total_insurance = insurance_y1 + insurance_y2 + insurance_y3 + insurance_y4 + insurance_y5

    # Maintenance cost per year (Low/Medium/High)
    maint_map = {'Low': 8000, 'Medium': 15000, 'High': 25000}
    annual_maintenance = maint_map.get(car_row.get('maintenance_cost', 'Medium'), 15000)
    # Electric vehicles have lower maintenance
    if fuel_type == 'Electric':
        annual_maintenance *= 0.5

    total_maintenance = annual_maintenance * 5

    # EMI cost (standard 20% down, 8.5%, 5yr)
    loan = price * 0.80
    r = 8.5 / 100 / 12
    emi = loan * r * (1 + r)**60 / ((1 + r)**60 - 1)
    total_emi = emi * 60

    # Depreciation (approx 50% after 5 years)
    depreciation = price * 0.50

    total_5yr = (annual_fuel_cost * 5) + total_insurance + total_maintenance
    # (EMI is separate — user may not finance)

    return {
        "annual_fuel_cost": round(annual_fuel_cost),
        "total_fuel_5yr": round(annual_fuel_cost * 5),
        "total_insurance_5yr": round(total_insurance),
        "annual_maintenance": round(annual_maintenance),
        "total_maintenance_5yr": round(total_maintenance),
        "total_emi_5yr": round(total_emi),
        "depreciation_5yr": round(depreciation),
        "total_ownership_5yr": round(total_5yr),
        "total_with_emi": round(total_5yr + total_emi),
        "monthly_emi": round(emi),
    }