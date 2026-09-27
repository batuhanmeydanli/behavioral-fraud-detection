"""Generate fully fictional transactions; no company data or derived statistics."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate():
    rng = np.random.default_rng(42)
    rows = []
    start = pd.Timestamp('2025-01-01T00:00:00Z')
    for card in range(8):
        for step in range(60):
            unusual = step in (48, 49, 50) and card % 2 == 0
            rows.append({
                'CardToken': f'SYNTHETIC_CARD_{card:03d}',
                'CardHolderIp': f'192.0.2.{200+card if unusual else card+1}',
                'Amount': round(float(rng.uniform(1500, 3000) if unusual else rng.uniform(40, 140)), 2),
                'CurrencyCode': 'USD' if step == 40 else 'TRY',
                'ResponseCode': '05' if unusual else '00',
                'TransactionDate': (start+pd.Timedelta(minutes=step*20+card)).isoformat(),
                'externalCustomerId': f'SYNTHETIC_CUSTOMER_{card:03d}',
            })
    return pd.DataFrame(rows).sort_values('TransactionDate').reset_index(drop=True)


if __name__ == '__main__':
    target = Path(__file__).with_name('synthetic_transactions.csv')
    generate().to_csv(target, index=False)
    print(f'Generated {len(generate())} fictional transactions: {target}')
