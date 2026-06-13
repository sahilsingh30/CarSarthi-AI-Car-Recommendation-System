def calculate_emi(principal, down_payment, annual_rate, tenure_months):
    loan_amount = principal - down_payment
    if loan_amount <= 0:
        return {"emi": 0, "total_interest": 0, "total_amount": principal, "loan_amount": 0}
    monthly_rate = annual_rate / 100 / 12
    if monthly_rate == 0:
        emi = loan_amount / tenure_months
    else:
        emi = loan_amount * monthly_rate * (1 + monthly_rate)**tenure_months / \
              ((1 + monthly_rate)**tenure_months - 1)
    total_amount = emi * tenure_months + down_payment
    total_interest = total_amount - principal
    return {
        "emi": round(emi),
        "loan_amount": round(loan_amount),
        "total_interest": round(total_interest),
        "total_amount": round(total_amount)
    }