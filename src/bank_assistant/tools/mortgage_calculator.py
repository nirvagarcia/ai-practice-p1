"""Deterministic French-amortization mortgage calculator."""

from decimal import ROUND_HALF_UP, Decimal

from bank_assistant.domain.schemas import MortgageInput, MortgageQuote

_CENT = Decimal("0.01")


def calculate_payment(data: MortgageInput) -> MortgageQuote:
    months = data.term_years * 12
    monthly_rate = data.annual_rate_percent / 100 / 12
    if monthly_rate == 0:
        payment = data.loan_amount / months
    else:
        payment = data.loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** -months)

    monthly_payment = payment.quantize(_CENT, ROUND_HALF_UP)
    total_paid = (payment * months).quantize(_CENT, ROUND_HALF_UP)
    return MortgageQuote(
        input=data,
        monthly_payment=monthly_payment,
        total_paid=total_paid,
        total_interest=total_paid - data.loan_amount,
    )
