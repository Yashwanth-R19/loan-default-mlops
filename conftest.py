"""Its presence at the repo root makes pytest add this directory to sys.path,
so `from src.x import y` works in every test file regardless of how pytest
is invoked. Also holds fixtures shared across the test suite."""
import pytest


@pytest.fixture
def valid_application() -> dict:
    return {
        "Age": 35,
        "Income": 65000,
        "LoanAmount": 15000,
        "CreditScore": 680,
        "MonthsEmployed": 48,
        "NumCreditLines": 3,
        "InterestRate": 12.5,
        "LoanTerm": 36,
        "DTIRatio": 0.35,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Married",
        "HasMortgage": "Yes",
        "HasDependents": "No",
        "LoanPurpose": "Auto",
        "HasCoSigner": "No",
    }
