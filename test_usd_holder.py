"""
Test USD holder mode with specified parameters.

Expected:
- USD Invested = 30,000 (exactly principal_usd)
- Break-even spot ≈ 43.1452
"""

from dashboard_core import USDHolderParams, CarryTradeModel

# Test parameters from user requirements
params = USDHolderParams(
    principal_usd=30_000,
    entry_bank_rate_usd_to_try=42.50,
    spot_entry=42.70,
    exit_spread_pct=0.0090,  # 0.90%
    deposit_rate_annual=0.395,  # 39.5%
    stopaj_rate=0.175,  # 17.5%
    term_days_calendar=32,
    tbill_rate_annual_usd=0.035,  # 3.5%
    swift_fee_usd=32.50,
)

print("="*80)
print("USD HOLDER MODE - SANITY TEST")
print("="*80)

print("\n--- INPUT PARAMETERS ---")
print(f"Principal USD:              ${params.principal_usd:,.2f}")
print(f"Entry Bank Rate (USD->TRY): {params.entry_bank_rate_usd_to_try:.4f}")
print(f"Spot at Entry:              {params.spot_entry:.4f}")
print(f"Exit Spread:                {params.exit_spread_pct*100:.2f}%")
print(f"Deposit Rate (annual):      {params.deposit_rate_annual*100:.1f}%")
print(f"Stopaj:                     {params.stopaj_rate*100:.1f}%")
print(f"Term:                       {params.term_days_calendar} days")
print(f"T-Bill Rate (annual):       {params.tbill_rate_annual_usd*100:.1f}%")
print(f"SWIFT Fee:                  ${params.swift_fee_usd:.2f}")

print("\n--- COMPUTED VALUES ---")
print(f"TRY Received (try0):        {params.try0:,.2f} TRY")
print(f"Final TRY (at maturity):    {params.final_try:,.2f} TRY")
print(f"USD Invested (usd0_exec):   ${params.usd0_exec:,.2f}")
print(f"T-Bill End Value:           ${params.usd_rf_end:,.2f}")

print("\n--- BREAK-EVEN ---")
print(f"Break-even Spot:            {params.spot_be:.4f}")
print(f"Break-even Move:            {params.be_move_pct:+.2f}%")

print("\n--- VERIFICATION ---")
# Critical check: usd0_exec must equal principal_usd
usd_invested_correct = (params.usd0_exec == params.principal_usd)
print(f"USD Invested = Principal?   {usd_invested_correct} (${params.usd0_exec:,.2f} vs ${params.principal_usd:,.2f})")

# Check break-even is approximately 43.1452
be_expected = 43.1452
be_error = abs(params.spot_be - be_expected)
be_correct = be_error < 0.01
print(f"Break-even ~= 43.1452?      {be_correct} (actual: {params.spot_be:.4f}, error: {be_error:.4f})")

# Verify break-even calculation
model = CarryTradeModel(params)
outcome_at_be = model.compute_outcome(params.spot_be)
excess_at_be = outcome_at_be.excess_ret_vs_tbill
be_math_correct = abs(excess_at_be) < 0.01
print(f"Excess at BE ~= 0?          {be_math_correct} (actual: {excess_at_be:.4f}%)")

print("\n--- OUTCOME AT ENTRY SPOT (unchanged) ---")
outcome_at_entry = model.compute_outcome(params.spot_entry)
print(f"USD End:                    ${outcome_at_entry.usd_end:,.2f}")
print(f"Return vs Principal:        {outcome_at_entry.ret_vs_convert_now:+.3f}%")
print(f"Excess vs T-Bill:           {outcome_at_entry.excess_ret_vs_tbill:+.3f}%")

print("\n" + "="*80)
if usd_invested_correct and be_correct and be_math_correct:
    print("PASS - ALL TESTS PASSED - USD HOLDER MODE IS CORRECT")
else:
    print("FAIL - SOME TESTS FAILED")
    if not usd_invested_correct:
        print("  - USD Invested does not equal principal_usd!")
    if not be_correct:
        print("  - Break-even spot is not approximately 43.1452!")
    if not be_math_correct:
        print("  - Excess return at break-even is not zero!")
print("="*80)
