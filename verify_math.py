"""
Complete mathematical breakdown of TRY Holder vs USD Holder modes.
"""

print('='*80)
print('COMPLETE MATH BREAKDOWN - USD HOLDER vs TRY HOLDER')
print('='*80)

# Common parameters
spot_entry = 42.72
deposit_rate_annual = 0.395
term_days = 32
stopaj_rate = 0.175
tbill_rate_annual = 0.035
swift_fee_usd = 32.50
exit_spread_pct = 0.0089

print('\n' + '='*80)
print('MODE 1: TRY HOLDER (Start with TRY, want USD)')
print('='*80)

principal_try = 1_430_000
entry_bank_rate_try_to_usd = 43.10  # TRY->USD rate (above spot)

print('\n--- STEP 1: ENTRY (Already have TRY) ---')
print(f'Principal:               {principal_try:,.0f} TRY')
print(f'Entry bank rate:         {entry_bank_rate_try_to_usd:.4f} (TRY->USD, above spot)')
print(f'Entry spread:            {(entry_bank_rate_try_to_usd/spot_entry - 1)*100:.2f}% above spot')
print('')
print('If converted now at entry bank rate:')
usd0_exec_try = principal_try / entry_bank_rate_try_to_usd
print(f'USD received:            ${usd0_exec_try:,.2f} (baseline reference)')

print('\n--- STEP 2: DEPOSIT (Earn TRY interest) ---')
T = term_days / 365.0
gross_interest = principal_try * deposit_rate_annual * T
stopaj = gross_interest * stopaj_rate
net_interest = gross_interest - stopaj
final_try = principal_try + net_interest

print(f'Gross interest:          {gross_interest:,.2f} TRY ({deposit_rate_annual*100:.1f}% x {principal_try:,.0f} TRY x {T:.4f} years)')
print(f'Stopaj tax ({stopaj_rate*100:.1f}%):       {stopaj:,.2f} TRY')
print(f'Net interest:            {net_interest:,.2f} TRY')
print(f'Final TRY:               {final_try:,.2f} TRY')

print('\n--- STEP 3: BASELINE (Convert at entry + T-bill) ---')
usd_rf_end_try = usd0_exec_try * (1 + tbill_rate_annual * T)
print(f'USD at entry:            ${usd0_exec_try:,.2f}')
print(f'T-bill rate:             {tbill_rate_annual*100:.1f}% annual')
print(f'T-bill for {term_days:.0f} days:      {(tbill_rate_annual * T)*100:.3f}%')
print(f'Baseline (USD + T-bill): ${usd_rf_end_try:,.2f}')

print('\n--- STEP 4: EXIT (Convert TRY->USD at exit) ---')
print(f'Spot at exit (assume):   {spot_entry:.4f} (unchanged)')
exit_bank_rate = spot_entry * (1 + exit_spread_pct)
print(f'Exit spread:             {exit_spread_pct*100:.2f}%')
print(f'Exit bank rate:          {exit_bank_rate:.4f} (spot x {1+exit_spread_pct:.4f})')
usd_gross = final_try / exit_bank_rate
usd_net = usd_gross - swift_fee_usd
print(f'USD from TRY conversion: ${usd_gross:,.2f}')
print(f'Less SWIFT fee:          ${swift_fee_usd:.2f}')
print(f'Net USD:                 ${usd_net:,.2f}')

print('\n--- STEP 5: RETURNS ---')
pnl_vs_convert = usd_net - usd0_exec_try
ret_vs_convert = (usd_net / usd0_exec_try - 1) * 100
pnl_vs_tbill = usd_net - usd_rf_end_try
excess_ret = (usd_net / usd_rf_end_try - 1) * 100

print(f'P&L vs convert now:      ${pnl_vs_convert:,.2f}')
print(f'Return vs convert:       {ret_vs_convert:+.3f}%')
print(f'P&L vs T-bill:           ${pnl_vs_tbill:,.2f}')
print(f'Excess vs T-bill:        {excess_ret:+.3f}%')

print('\n--- STEP 6: BREAK-EVEN ---')
# Solve: final_try / (spot_be * (1 + exit_spread)) - swift = usd_rf_end
# spot_be * (1 + exit_spread) = final_try / (usd_rf_end + swift)
bank_rate_be_try = final_try / (usd_rf_end_try + swift_fee_usd)
spot_be_try = bank_rate_be_try / (1 + exit_spread_pct)
be_move_try = (spot_be_try / spot_entry - 1) * 100

print(f'Bank rate at BE:         {bank_rate_be_try:.4f}')
print(f'Spot at BE:              {spot_be_try:.4f}')
print(f'BE move from entry:      {be_move_try:+.2f}%')

# Verify
usd_at_be = final_try / bank_rate_be_try - swift_fee_usd
print(f'Verify USD at BE:        ${usd_at_be:,.2f} (should = ${usd_rf_end_try:,.2f})')

print('\n\n' + '='*80)
print('MODE 2: USD HOLDER (Start with USD, convert to TRY, convert back)')
print('='*80)

principal_usd = 33_000
entry_bank_rate_usd_to_try = 42.34  # USD->TRY rate (below spot)

print('\n--- STEP 1: ENTRY (Convert USD->TRY) ---')
print(f'Principal:               ${principal_usd:,.0f} USD')
print(f'Spot at entry:           {spot_entry:.4f}')
print(f'Entry bank rate:         {entry_bank_rate_usd_to_try:.4f} (USD->TRY, below spot)')
print(f'Entry haircut:           {(1 - entry_bank_rate_usd_to_try/spot_entry)*100:.2f}% below spot (market rate)')
try0 = principal_usd * entry_bank_rate_usd_to_try
print(f'TRY received:            {try0:,.2f} TRY')
print('')
print('NOTE: This is NOT a spread charge - just market achievable rate')

print('\n--- STEP 2: DEPOSIT (Earn TRY interest) ---')
gross_interest_usd = try0 * deposit_rate_annual * T
stopaj_usd = gross_interest_usd * stopaj_rate
net_interest_usd = gross_interest_usd - stopaj_usd
final_try_usd = try0 + net_interest_usd

print(f'Gross interest:          {gross_interest_usd:,.2f} TRY')
print(f'Stopaj tax ({stopaj_rate*100:.1f}%):       {stopaj_usd:,.2f} TRY')
print(f'Net interest:            {net_interest_usd:,.2f} TRY')
print(f'Final TRY:               {final_try_usd:,.2f} TRY')

print('\n--- STEP 3: BASELINE (Keep USD in T-bill) ---')
usd0_exec_usd = principal_usd  # CRITICAL: baseline is original USD
usd_rf_end_usd = usd0_exec_usd * (1 + tbill_rate_annual * T)
print(f'USD principal:           ${usd0_exec_usd:,.2f} (what you started with)')
print(f'T-bill rate:             {tbill_rate_annual*100:.1f}% annual')
print(f'Baseline (USD + T-bill): ${usd_rf_end_usd:,.2f}')

print('\n--- STEP 4: EXIT (Convert TRY->USD at exit) ---')
print(f'Spot at exit (assume):   {spot_entry:.4f} (unchanged)')
exit_bank_rate_usd = spot_entry * (1 + exit_spread_pct)
print(f'Exit spread:             {exit_spread_pct*100:.2f}% (ONLY spread paid)')
print(f'Exit bank rate:          {exit_bank_rate_usd:.4f} (spot x {1+exit_spread_pct:.4f})')
usd_gross_usd = final_try_usd / exit_bank_rate_usd
usd_net_usd = usd_gross_usd - swift_fee_usd
print(f'USD from TRY conversion: ${usd_gross_usd:,.2f}')
print(f'Less SWIFT fee:          ${swift_fee_usd:.2f}')
print(f'Net USD:                 ${usd_net_usd:,.2f}')

print('\n--- STEP 5: RETURNS ---')
pnl_vs_convert_usd = usd_net_usd - usd0_exec_usd
ret_vs_convert_usd = (usd_net_usd / usd0_exec_usd - 1) * 100
pnl_vs_tbill_usd = usd_net_usd - usd_rf_end_usd
excess_ret_usd = (usd_net_usd / usd_rf_end_usd - 1) * 100

print(f'P&L vs principal:        ${pnl_vs_convert_usd:,.2f}')
print(f'Return vs principal:     {ret_vs_convert_usd:+.3f}%')
print(f'P&L vs T-bill:           ${pnl_vs_tbill_usd:,.2f}')
print(f'Excess vs T-bill:        {excess_ret_usd:+.3f}%')

print('\n--- STEP 6: BREAK-EVEN ---')
bank_rate_be_usd = final_try_usd / (usd_rf_end_usd + swift_fee_usd)
spot_be_usd = bank_rate_be_usd / (1 + exit_spread_pct)
be_move_usd = (spot_be_usd / spot_entry - 1) * 100

print(f'Bank rate at BE:         {bank_rate_be_usd:.4f}')
print(f'Spot at BE:              {spot_be_usd:.4f}')
print(f'BE move from entry:      {be_move_usd:+.2f}%')

# Verify
usd_at_be_usd = final_try_usd / bank_rate_be_usd - swift_fee_usd
print(f'Verify USD at BE:        ${usd_at_be_usd:,.2f} (should = ${usd_rf_end_usd:,.2f})')

print('\n\n' + '='*80)
print('COMPARISON SUMMARY')
print('='*80)
print('')
print(f'{"Metric":<30} {"TRY Holder":>15} {"USD Holder":>15}')
print('-'*80)
print(f'{"Principal (TRY)":<30} {principal_try:>15,.0f} {try0:>15,.0f}')
print(f'{"Final TRY":<30} {final_try:>15,.2f} {final_try_usd:>15,.2f}')
print(f'{"Baseline USD":<30} {usd_rf_end_try:>15,.2f} {usd_rf_end_usd:>15,.2f}')
print(f'{"Exit USD (net)":<30} {usd_net:>15,.2f} {usd_net_usd:>15,.2f}')
print(f'{"Excess vs T-bill (%)":<30} {excess_ret:>15.2f} {excess_ret_usd:>15.2f}')
print(f'{"Break-even spot":<30} {spot_be_try:>15.4f} {spot_be_usd:>15.4f}')
print(f'{"BE move (%)":<30} {be_move_try:>15.2f} {be_move_usd:>15.2f}')
print('')
print('WHY DIFFERENT?')
print(f'- TRY holder has MORE TRY to work with ({principal_try:,.0f} vs {try0:,.0f})')
print('- TRY holder baseline includes entry spread cost already')
print('- USD holder baseline is pure T-bill (no FX cost baked in)')
print(f'- USD holder gets less TRY at entry ({try0:,.0f} vs {principal_try:,.0f})')
print(f'- Entry rate difference: {entry_bank_rate_usd_to_try:.4f} vs {entry_bank_rate_try_to_usd:.4f}')
