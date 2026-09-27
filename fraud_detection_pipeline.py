#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavioral fraud analysis: feature engineering, models, reports and simulator.

Developed during a machine learning internship. Public demonstration with
synthetic data; see README for scope and known limitations.
"""
import argparse
import ast
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class Config:
    DATA_PATHS = {
        'multi_data': 'examples/synthetic_transactions.csv',
        'output_features': 'outputs/transactions_with_features.csv',
    }

    USER_COL = 'CardToken'
    IP_COL = 'CardHolderIp'
    AMOUNT_COL = 'Amount'
    CURRENCY_COL = 'CurrencyCode'
    RESPONSE_COL = 'ResponseCode'
    DATE_COL = 'TransactionDate'

    # Model parametreleri
    ISOLATION_FOREST_PARAMS = {
        'n_estimators': 150,
        'contamination': 0.01,
        'random_state': 42
    }


    OneClassSVM_PARAMS = {
        'kernel': 'rbf',      # veya 'linear', 'sigmoid', 'poly'
        'gamma': 'auto',      # veya float (örneğin: 0.01)
        'nu': 0.05            # anomali oranı, 0 ile 1 arasında olmalı
    }

    # Z-score eşik değerleri
    Z_SCORE_THRESHOLDS = [2.5, 2, 1.5, 1]


MODEL_FEATURES = [
    'last_5_amount_avg', 'amount_zscore', 'amount_daily_zscore', 'daily_mean',
    'last_10min_tx_count', 'last_1h_tx_count', 'daily_tx_count', 'daily_amount_sum',
    'max_unique_ip_count_1h', 'is_new_ip', 'consecutive_decline_count',
    'time_since_last_tx', 'velocity', 'daily_currency_variety', 'daily_card_variety',
    'unique_user_per_card', 'same_amount_tx_count_last_1h', 'currency_change',
    'tx_hour_pattern'
]


def validate_data(data):
    """Validate schema, preserve response strings, normalize timestamps to UTC."""
    required = [Config.USER_COL, Config.IP_COL, Config.AMOUNT_COL,
                Config.CURRENCY_COL, Config.RESPONSE_COL, Config.DATE_COL]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if data.empty:
        raise ValueError('No transactions supplied.')
    data = data.copy().reset_index(drop=True)
    for col in required:
        if data[col].isna().any() or data[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{col} contains missing values.')
    for col in [Config.USER_COL, Config.IP_COL, Config.CURRENCY_COL, Config.RESPONSE_COL]:
        data[col] = data[col].astype(str).str.strip()
    data[Config.RESPONSE_COL] = data[Config.RESPONSE_COL].str.zfill(2)
    data[Config.AMOUNT_COL] = pd.to_numeric(data[Config.AMOUNT_COL], errors='raise')
    if not np.isfinite(data[Config.AMOUNT_COL].to_numpy(dtype=float)).all():
        raise ValueError('Amounts must be finite numbers.')
    data[Config.DATE_COL] = pd.to_datetime(data[Config.DATE_COL], errors='raise', utc=True)
    if data[Config.DATE_COL].isna().any():
        raise ValueError('Invalid transaction timestamps.')
    return data


def load_data(file_path):
    string_columns = [Config.USER_COL, Config.IP_COL, Config.CURRENCY_COL,
                      Config.RESPONSE_COL, 'externalCustomerId']
    return validate_data(pd.read_csv(file_path, dtype={c: str for c in string_columns},
                                     keep_default_na=False))


def save_data(data, file_path):
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(target, index=False)
    print(f'Veri kaydedildi: {target}')


def _per_transaction(df, function):
    """Assign one returned value per original transaction, never per card."""
    result = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby(Config.USER_COL, sort=False):
        values = np.asarray(function(group))
        if values.ndim != 1 or len(values) != len(group):
            raise ValueError('Feature function must return one value per transaction.')
        result.loc[group.index] = values
    return result


def force_scalar(x):
    """Değeri skaler hale getirme fonksiyonu"""
    if isinstance(x, (int, float, np.integer, np.floating)):
        return x
    if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list) and len(val) > 0:
                return val[0]
        except:
            return 0
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    return 0


def rolling_zscore(series, window=50):
    """Hareketli Z-score hesaplama"""
    return (series - series.rolling(window, min_periods=2).mean()) / series.rolling(window, min_periods=2).std()


def dynamic_flag(series, k=1.5, window=10):
    """Dinamik eşik değeri ile anomali tespiti"""
    rolling_mean = series.rolling(window=window, min_periods=2).mean()
    rolling_std = series.rolling(window=window, min_periods=2).std()
    upper = rolling_mean + k * rolling_std
    lower = rolling_mean - k * rolling_std
    return (series > upper) | (series < lower)


def calculate_time_based_features(group, time_window_seconds):
    """Zaman bazlı özellikler hesaplama"""
    timestamps = group[Config.DATE_COL].values.astype('datetime64[s]')
    counts = np.zeros(len(timestamps), dtype=int)

    for i in range(len(timestamps)):
        time_mask = (timestamps[:i] >= timestamps[i] - np.timedelta64(time_window_seconds, 's')) & (timestamps[:i] < timestamps[i])
        counts[i] = np.sum(time_mask)

    return counts


def max_unique_ips_in_1h_window(group):
    """Ttüm işlemlerinde, herhangi bir 1 saatlik pencerede en fazla farklı IP sayısı"""
    group = group.sort_values(Config.DATE_COL, kind='stable')
    times = group[Config.DATE_COL].values.astype('datetime64[s]')
    ips = group[Config.IP_COL].astype(str).values
    max_unique = 0
    for i in range(len(times)):
        window_start = times[i]
        window_end = window_start + np.timedelta64(3600, 's')
        mask = (times >= window_start) & (times < window_end)
        unique_ips = len(set(ips[mask]))
        if unique_ips > max_unique:
            max_unique = unique_ips
    return max_unique


def engineer_features(transactions_df):
    """Özellik mühendisliği ana fonksiyonu"""
    # Veri yükleme ve hazırlama
    transactions_df = validate_data(transactions_df)
    transactions_df = transactions_df.sort_values([Config.USER_COL, Config.DATE_COL], kind="stable")

    # 1. Son 5 işlemin ortalaması
    transactions_df['last_5_amount_avg'] = (
        transactions_df.groupby(Config.USER_COL)[Config.AMOUNT_COL]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # 2. Amount z-score (rolling, window=10)
    window_size = 10
    k = 1.0

    transactions_df['amount_zscore'] = (
        transactions_df.groupby(Config.USER_COL)[Config.AMOUNT_COL]
        .transform(lambda x: (x - x.rolling(window=window_size, min_periods=2).mean()) / x.rolling(window=window_size, min_periods=2).std())
    )

    transactions_df['zscore_anomaly_flag'] = (transactions_df['amount_zscore'].abs() > k).astype(int)


    # 3. Son 10 dakikadaki işlem sayısı
    transactions_df['last_10min_tx_count'] = (
        _per_transaction(transactions_df, lambda x: calculate_time_based_features(x, 600))
    )

    # 4. Son 1 saatteki işlem sayısı
    transactions_df['last_1h_tx_count'] = (
        _per_transaction(transactions_df, lambda x: calculate_time_based_features(x, 3600))
    )

    # 5. Günlük işlem sayısı ve toplamı
    transactions_df['date'] = transactions_df[Config.DATE_COL].dt.date
    transactions_df['daily_tx_count'] = transactions_df.groupby(
        [Config.USER_COL, 'date'])[Config.AMOUNT_COL].transform('count')
    transactions_df['daily_amount_sum'] = transactions_df.groupby(
        [Config.USER_COL, 'date'])[Config.AMOUNT_COL].transform('sum')

    # 6. Herhangi bir 1 saatlik dilimde maksimum farklı IP sayısı (user/card bazında)
    max_ips = pd.Series({key: max_unique_ips_in_1h_window(group) for key, group in transactions_df.groupby(Config.USER_COL)})
    transactions_df['max_unique_ip_count_1h'] = transactions_df[Config.USER_COL].map(max_ips)


    # 7. Yeni IP flagi
    def is_new_ip(group):
        seen_ips = set()
        results = []
        for ip in group[Config.IP_COL]:
            results.append(0 if ip in seen_ips else 1)
            seen_ips.add(ip)
        return results

    transactions_df['is_new_ip'] = _per_transaction(transactions_df, is_new_ip)

    # 8. Peş peşe decline sayısı
    def consecutive_decline_count(group):
        result = []
        count = 0
        for code in group[Config.RESPONSE_COL]:
            if code != '00':
                count += 1
            else:
                count = 0
            result.append(count)
        return result

    transactions_df['consecutive_decline_count'] = _per_transaction(transactions_df, consecutive_decline_count)

    # 9. Son işlemden bu yana geçen süre (saniye)
    def time_since_last_tx(group):
        times = group[Config.DATE_COL]
        return times.diff().dt.total_seconds().fillna(0)

    transactions_df['time_since_last_tx'] = _per_transaction(transactions_df, time_since_last_tx)

    # 10. Velocity (son 5 işlem arası ortalama dakika)
    def velocity(group):
        times = group[Config.DATE_COL].sort_values()
        if len(times) < 2:
            return pd.Series([np.nan] * len(times), index=times.index)
        return times.diff().dt.total_seconds().rolling(window=5, min_periods=1).mean() / 60

    transactions_df['velocity'] = _per_transaction(transactions_df, velocity)

    # Dinamik threshold (rolling mean + k*rolling std) flag'i
    k = 1.5
    window_size = 10

    transactions_df['dynamic_amount_flag'] = (
        transactions_df.groupby(Config.USER_COL)[Config.AMOUNT_COL]
        .apply(lambda x: dynamic_flag(x, k=k, window=window_size))
        .reset_index(level=0, drop=True)
        .astype(int)
    )

    # 11. Günlük currency çeşitliliği
    transactions_df['daily_currency_variety'] = transactions_df.groupby(
        [Config.USER_COL, 'date'])[Config.CURRENCY_COL].transform('nunique')

    # 12. Günlük kullanılan farklı kart sayısı (user_id varsa)
    if 'externalCustomerId' in transactions_df.columns:
        transactions_df['daily_card_variety'] = transactions_df.groupby(
            ['externalCustomerId', 'date'])[Config.USER_COL].transform('nunique')
    else:
        transactions_df['daily_card_variety'] = 1

    # 13. Aynı kartı kullanan farklı kullanıcı sayısı (user_id varsa)
    if 'externalCustomerId' in transactions_df.columns:
        transactions_df['unique_user_per_card'] = transactions_df.groupby(
            Config.USER_COL)['externalCustomerId'].transform('nunique')
    else:
        transactions_df['unique_user_per_card'] = 1

    # 14. Son 1 saatte aynı tutarda işlem sayısı
    def same_amount_tx_count_last_1h(group):
        timestamps = group[Config.DATE_COL].values.astype('datetime64[s]')
        amounts = group[Config.AMOUNT_COL].values
        counts = np.zeros(len(timestamps), dtype=int)

        for i in range(len(timestamps)):
            time_mask = (timestamps[:i] >= timestamps[i] - np.timedelta64(3600, 's')) & (timestamps[:i] < timestamps[i])
            counts[i] = np.sum(amounts[:i][time_mask] == amounts[i])
        return counts

    transactions_df['same_amount_tx_count_last_1h'] = (
        _per_transaction(transactions_df, same_amount_tx_count_last_1h)
    )

    # 15. Currency değişimi flag'i
    def currency_change(group):
        if group[Config.CURRENCY_COL].empty:
            return pd.Series([0] * len(group), index=group.index)

        main_currency = group[Config.CURRENCY_COL].mode()
        if not main_currency.empty:
            main_currency = main_currency[0]
        else:
            return pd.Series([0] * len(group), index=group.index)

        return (group[Config.CURRENCY_COL] != main_currency).astype(int)

    transactions_df['currency_change'] = _per_transaction(transactions_df, currency_change)

    # 16. Saat bazlı davranışsal pattern
    transactions_df['tx_hour'] = transactions_df[Config.DATE_COL].dt.hour
    transactions_df['tx_hour_pattern'] = transactions_df.groupby(Config.USER_COL)['tx_hour'].transform(
        lambda x: x.mode()[0] if not x.mode().empty else -1)

    # Günlük yapılan işlem ortalaması ve Z-score hesabı
    transactions_df['daily_mean'] = transactions_df.groupby(
        [Config.USER_COL, 'date'])[Config.AMOUNT_COL].transform('mean')
    transactions_df['amount_daily_zscore'] = (
        transactions_df[Config.AMOUNT_COL] - transactions_df['daily_mean']
    ) / transactions_df.groupby([Config.USER_COL, 'date'])[Config.AMOUNT_COL].transform('std')

    # Tek elemanlı liste olarak kalan kolonları düzelt
    feature_list_columns = [
         'is_new_ip', 'consecutive_decline_count',
        'same_amount_tx_count_last_1h', 'last_10min_tx_count', 'last_1h_tx_count'
    ]

    for column in feature_list_columns:
        transactions_df[column] = transactions_df[column].apply(force_scalar)

    # Kaydet
    print("Özellik mühendisliği tamamlandı.")

    return transactions_df.reset_index(drop=True)



def perform_feature_engineering(input_path, output_path):
    transactions = engineer_features(load_data(input_path))
    save_data(transactions, output_path)
    return transactions


def prepare_model_data(data_path, feature_columns):
    """Model için veri hazırlama"""
    transactions_data = load_data(data_path)

    # Özellikleri seç
    model_data = transactions_data[feature_columns].copy()

    # Liste olarak kalmış kolonları düzelt
    list_columns = [
         'is_new_ip', 'consecutive_decline_count',
        'same_amount_tx_count_last_1h', 'last_10min_tx_count', 'last_1h_tx_count'
    ]

    for column in list_columns:
        if column in model_data.columns:
            model_data[column] = model_data[column].apply(force_scalar)

    # NaN değerleri doldur
    # Undefined rolling z-scores (cold start / zero variance) use neutral zero.
    # This is not a repair for broken row alignment: all count features are checked.
    numeric = model_data.replace('', np.nan).apply(pd.to_numeric, errors='raise')
    transactions_data[feature_columns] = numeric
    allowed_missing = ['amount_zscore', 'amount_daily_zscore', 'velocity']
    count_cols = [c for c in feature_columns if c not in allowed_missing]
    if numeric[count_cols].isna().any().any():
        raise ValueError('Unexpected missing feature values.')
    model_data = numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

    return model_data, transactions_data


def train_isolation_forest(model_data, feature_columns):
    isolation_forest = IsolationForest(**Config.ISOLATION_FOREST_PARAMS)
    model_data['isoforest_prediction'] = isolation_forest.fit_predict(model_data[feature_columns])
    model_data['isoforest_flag'] = model_data['isoforest_prediction'].apply(lambda x: 1 if x == -1 else 0)
    model_data['isoforest_score'] = isolation_forest.decision_function(model_data[feature_columns])
    return model_data, isolation_forest


def train_oneclass_svm(model_data, feature_columns):
    svm = OneClassSVM(**Config.OneClassSVM_PARAMS)
    model_data['ocsvm_prediction'] = svm.fit_predict(model_data[feature_columns])
    model_data['ocsvm_flag'] = model_data['ocsvm_prediction'].apply(lambda x: 1 if x == -1 else 0)
    model_data['ocsvm_score'] = svm.decision_function(model_data[feature_columns])
    return model_data, svm


def analyze_results(transactions_data, model_data):
    print("Dynamic Amount Flag dağılımı:")
    print(transactions_data['dynamic_amount_flag'].value_counts())
    print("\nIsolation Forest Flag dağılımı:")
    print(model_data['isoforest_flag'].value_counts())
    print("\nOne-Class SVM Flag dağılımı:")
    print(model_data['ocsvm_flag'].value_counts())
    print("\nZ-score anomali oranları:")
    for threshold in Config.Z_SCORE_THRESHOLDS:
        amount_zscore_rate = (model_data['amount_zscore'].abs() > threshold).mean()
        daily_zscore_rate = (model_data['amount_daily_zscore'].abs() > threshold).mean()
        print(f"|Z-score| > {threshold}: {amount_zscore_rate:.4f}, |Günlük Z-score| > {threshold}: {daily_zscore_rate:.4f}")
    transactions_data['isoforest_flag'] = model_data['isoforest_flag']
    transactions_data['ocsvm_flag'] = model_data['ocsvm_flag']
    print("\nEn çok anomali içeren kartlar (Isolation Forest):")
    print(transactions_data.groupby(Config.USER_COL)['isoforest_flag'].sum().sort_values(ascending=False).head(10))
    print("\nEn çok anomali içeren kartlar (OCSVM):")
    print(transactions_data.groupby(Config.USER_COL)['ocsvm_flag'].sum().sort_values(ascending=False).head(10))
    print("\nÜç yöntemle birlikte işaretlenen anomaliler:")
    print(transactions_data[(transactions_data['dynamic_amount_flag'] == 1) & (transactions_data['isoforest_flag'] == 1) & (transactions_data['ocsvm_flag'] == 1)][[
        Config.USER_COL, Config.DATE_COL, Config.AMOUNT_COL
    ]].head())


def enhanced_results_analysis(transactions_data, model_data, config):


    """Geliştirilmiş sonuç analizi"""

    print("=" * 80)
    print("             FRAUD DETECTION MODEL SONUÇLARI")
    print("=" * 80)

    # Temel istatistikler
    total_transactions = len(transactions_data)
    print(f"\n📊 TEMEL İSTATİSTİKLER")
    print("-" * 50)
    print(f"Toplam İşlem Sayısı        : {total_transactions:,}")
    print(f"Benzersiz Kart Sayısı      : {transactions_data[config.USER_COL].nunique():,}")
    print(f"Tarih Aralığı             : {transactions_data[config.DATE_COL].min().date()} - {transactions_data[config.DATE_COL].max().date()}")
    print(f"Ortalama Günlük İşlem     : {total_transactions / (transactions_data[config.DATE_COL].dt.date.nunique()):,.0f}")

    # Anomali oranları
    print(f"\n🚨 ANOMALİ TESPİT SONUÇLARI")
    print("-" * 50)

    # Dynamic Amount Flag
    dynamic_anomalies = transactions_data['dynamic_amount_flag'].sum()
    dynamic_ratio = dynamic_anomalies / total_transactions * 100
    print(f"Dinamik Threshold Anomalileri  : {dynamic_anomalies:,} (%{dynamic_ratio:.2f})")

    # OCSVM
    svm_anomalies = model_data['ocsvm_flag'].sum()
    svm_ratio = svm_anomalies / total_transactions * 100
    print(f"One-Class SVM Anomalileri   : {svm_anomalies:,} (%{svm_ratio:.2f})")

    # Z-score anomalileri
    zscore_anomalies = (model_data['amount_zscore'].abs() > 1.0).sum()
    zscore_ratio = zscore_anomalies / total_transactions * 100
    print(f"Z-Score Anomalileri (|z|>1.0)  : {zscore_anomalies:,} (%{zscore_ratio:.2f})")

    # Overlap analizi
    both_anomalies = transactions_data[(transactions_data['dynamic_amount_flag'] == 1) &
                                     (transactions_data['isoforest_flag'] == 1)]
    overlap_count = len(both_anomalies)
    print(f"Dynamic + Isolation Forest       : {overlap_count:,} (Dynamic'in %{(overlap_count/dynamic_anomalies*100 if dynamic_anomalies else 0):.1f})")

    # Z-score detay analizi
    print(f"\n📈 Z-SCORE DETAY ANALİZİ")
    print("-" * 50)
    print("Threshold    Anomali Sayısı    Oran")
    print("-" * 40)
    for threshold in [0.5, 1.0, 1.5, 2.0, 2.5]:
        count = (model_data['amount_zscore'].abs() > threshold).sum()
        ratio = count / total_transactions * 100
        print(f"|z| > {threshold:<6} {count:>8,} işlem    %{ratio:>5.2f}")

    # En riskli kartlar
    print(f"\n⚠️  EN ÇOK ANOMALİ İŞARETLENEN KARTLAR (Isolation Forest)")
    print("-" * 50)
    top_risky_cards = (transactions_data.groupby(config.USER_COL)['isoforest_flag']
                      .sum().sort_values(ascending=False).head(10))

    for i, (card, anomaly_count) in enumerate(top_risky_cards.items(), 1):
        total_tx = len(transactions_data[transactions_data[config.USER_COL] == card])
        anomaly_rate = anomaly_count / total_tx * 100
        print(f"{i:2d}. {card:<15} : {anomaly_count:2d}/{total_tx:2d} işlem (%{anomaly_rate:4.1f})")

    return {
        'total_transactions': total_transactions,
        'dynamic_anomalies': dynamic_anomalies,
        'svm_anomalies': svm_anomalies,
        'zscore_anomalies': zscore_anomalies,
        'overlap_count': overlap_count,
        'top_risky_cards': top_risky_cards
    }


def detailed_anomaly_examples(transactions_data, config, n_examples=10):
    """Detaylı anomali örnekleri"""

    print(f"\n🔍 DETAYLI ANOMALİ ÖRNEKLERİ")
    print("=" * 80)

    # Anomali işlemleri al
    anomaly_transactions = transactions_data[transactions_data['ocsvm_flag'] == 1].copy()

    # Z-score'a göre sırala (en yüksekten en düşüğe)
    anomaly_transactions['abs_zscore'] = anomaly_transactions['amount_zscore'].abs()
    top_anomalies = anomaly_transactions.nlargest(n_examples, 'abs_zscore')

    print(f"En yüksek Z-score'lu {n_examples} anomali işlem:")
    print("-" * 80)

    display_columns = [
        config.USER_COL, config.DATE_COL, config.AMOUNT_COL,
        'amount_zscore', 'daily_tx_count', 'velocity', 'dynamic_amount_flag'
    ]

    for idx, row in top_anomalies.iterrows():
        print(f"\n{idx+1:2d}. ANOMALİ İŞLEM:")
        print(f"    Kart           : {row[config.USER_COL]}")
        print(f"    Tarih/Saat     : {row[config.DATE_COL]}")
        print(f"    Tutar          : {row[config.AMOUNT_COL]:,.2f}")
        print(f"    Z-Score        : {row['amount_zscore']:.3f}")
        print(f"    Günlük İş. Say.: {row['daily_tx_count']:.0f}")
        print(f"    Velocity       : {row['velocity']:.1f} dk" if pd.notna(row['velocity']) else "    Velocity       : N/A")
        print(f"    Dinamik Flag   : {'✓' if row['dynamic_amount_flag'] == 1 else '✗'}")
        print(f"    {'─' * 50}")



def model_performance_summary(model_data):
    """Report observed flag rates without claiming accuracy or optimal tuning."""
    print('\nMODEL ÇIKTI ÖZETİ (etiketsiz, eğitim verisi üzerinde)')
    for name in ['isoforest', 'ocsvm']:
        print(f"{name}: {int(model_data[name + '_flag'].sum())}/{len(model_data)} anomali")
    agreement = (model_data['isoforest_flag'] == model_data['ocsvm_flag']).mean()
    print(f'İki modelin etiket uyumu: {agreement:.2%} (doğruluk metriği değildir)')
    print('Grid search veya etiketli fraud değerlendirmesi bu akışta yapılmaz.')
    print('Yöntemler tamamlayıcıdır; tek bir yöntemin işaretlememesi birleşik sonucu göstermez.')


def _finish_plot(output_path=None):
    fig = plt.gcf()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return fig


def create_comprehensive_visualizations(transactions_data, model_data, config, output_path=None):
    """Kapsamlı görselleştirme"""

    print(f"\n📊 GÖRSELLEŞTİRME HAZIRLANIYOR...")

    # 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Fraud Detection Model - Kapsamlı Analiz', fontsize=16, fontweight='bold')

    # 1. Anomali oranları pasta grafiği
    ax1 = axes[0, 0]
    normal_count = len(transactions_data) - model_data['isoforest_flag'].sum()
    anomaly_count = model_data['isoforest_flag'].sum()

    ax1.pie([normal_count, anomaly_count],
            labels=['Normal İşlemler', 'Anomali İşlemler'],
            autopct='%1.2f%%', startangle=90, colors=['lightblue', 'red'])
    ax1.set_title('Isolation Forest - Anomali Oranları')

    # 2. Z-score dağılımı
    ax2 = axes[0, 1]
    zscore_clean = model_data['amount_zscore'].dropna()
    ax2.hist(zscore_clean, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Threshold (k=1.0)')
    ax2.axvline(x=-1.0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('Z-Score Dağılımı')
    ax2.set_xlabel('Z-Score')
    ax2.set_ylabel('Frekans')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Günlük anomali sayısı
    ax3 = axes[0, 2]
    daily_anomalies = (transactions_data[transactions_data['ocsvm_flag'] == 1]
                      .groupby(transactions_data[config.DATE_COL].dt.date)
                      .size())

    ax3.plot(daily_anomalies.index, daily_anomalies.values, marker='o', linewidth=2)
    if len(daily_anomalies):
        ax3.set_xlim(pd.Timestamp(daily_anomalies.index.min()) - pd.Timedelta(hours=12),
                     pd.Timestamp(daily_anomalies.index.max()) + pd.Timedelta(hours=12))
    ax3.set_title('OCSVM - Günlük Anomali Sayısı')
    ax3.set_xlabel('Tarih')
    ax3.set_ylabel('Anomali Sayısı')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. Anomali tutar dağılımı
    ax4 = axes[1, 0]
    normal_amounts = transactions_data[transactions_data['ocsvm_flag'] == 0][config.AMOUNT_COL]
    anomaly_amounts = transactions_data[transactions_data['ocsvm_flag'] == 1][config.AMOUNT_COL]

    ax4.hist(normal_amounts if len(normal_amounts) else [], bins=50, alpha=0.5, label='Normal', color='blue', density=False)
    ax4.hist(anomaly_amounts, bins=30, alpha=0.7, label='Anomali', color='red', density=False)
    ax4.set_title('OCSVM - İşlem Tutarı Dağılımı')
    ax4.set_xlabel('Tutar')
    ax4.set_ylabel('İşlem sayısı')
    ax4.legend()
    if transactions_data[config.AMOUNT_COL].nunique() > 1:
        ax4.set_xlim(min(0, transactions_data[config.AMOUNT_COL].min()), np.percentile(transactions_data[config.AMOUNT_COL], 95))

    # 5. Saatlik anomali dağılımı
    ax5 = axes[1, 1]
    hourly_anomalies = (transactions_data[transactions_data['ocsvm_flag'] == 1]
                       .groupby(transactions_data[config.DATE_COL].dt.hour)
                       .size())

    bars = ax5.bar(hourly_anomalies.index, hourly_anomalies.values, alpha=0.7, color='orange')
    ax5.set_title('OCSVM - Saatlik Anomali Dağılımı')
    ax5.set_xlabel('Saat')
    ax5.set_ylabel('Anomali Sayısı')
    ax5.set_xticks(range(0, 24, 2))

    # En yüksek değerleri vurgula
    if len(hourly_anomalies):
        bars[int(hourly_anomalies.values.argmax())].set_color('red')

    # 6. İki yöntem karşılaştırması
    ax6 = axes[1, 2]

    # Confusion matrix benzeri
    both_normal = len(transactions_data[(transactions_data['dynamic_amount_flag'] == 0) &
                                       (transactions_data['ocsvm_flag'] == 0)])
    both_anomaly = len(transactions_data[(transactions_data['dynamic_amount_flag'] == 1) &
                                        (transactions_data['ocsvm_flag'] == 1)])
    dynamic_only = len(transactions_data[(transactions_data['dynamic_amount_flag'] == 1) &
                                        (transactions_data['ocsvm_flag'] == 0)])
    iso_only = len(transactions_data[(transactions_data['dynamic_amount_flag'] == 0) &
                                    (transactions_data['ocsvm_flag'] == 1)])

    categories = ['Her İkisi\nNormal', 'Sadece\nDynamic', 'Sadece\nOCSVM', 'Her İkisi\nAnomali']
    values = [both_normal, dynamic_only, iso_only, both_anomaly]
    colors = ['lightgreen', 'yellow', 'orange', 'red']

    ax6.bar(categories, values, color=colors, alpha=0.7)
    ax6.set_title('Yöntem Karşılaştırması')
    ax6.set_ylabel('İşlem Sayısı')

    # Değerleri çubukların üzerine yaz
    for i, v in enumerate(values):
        ax6.text(i, v + max(values)*0.01, f'{v:,}', ha='center', va='bottom')

    plt.tight_layout()
    _finish_plot(output_path)
    return fig


def visualize_one_class_svm_results(transactions_data, model_data, card_token, output_path=None):
    """One-Class SVM sonuçlarını görselleştirme"""
    # Kart verisini filtrele
    card_mask = transactions_data[Config.USER_COL] == card_token
    card_data = transactions_data[card_mask].copy()
    card_model_data = model_data[card_mask].copy()

    # Veriyi tarihe göre sırala
    ordered = card_data.sort_values(Config.DATE_COL, kind='stable').index
    card_data = card_data.loc[ordered].reset_index(drop=True)
    card_model_data = card_model_data.loc[ordered].reset_index(drop=True)
    if card_data.empty:
        raise ValueError('Selected card does not exist.')

    # Görselleştirme
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Üst grafik: İşlem tutarları ve anomaliler
    ax1.plot(card_data[Config.DATE_COL], card_data[Config.AMOUNT_COL],
             marker='o', color='darkblue', label='Tutar', linewidth=2)
    ax1.set_xlabel('Tarih')
    ax1.set_ylabel('Tutar', color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Anomali noktalarını göster
    anomaly_idx = card_data.index[card_model_data['ocsvm_flag'] == 1]
    ax1.scatter(card_data.loc[anomaly_idx, Config.DATE_COL],
                card_data.loc[anomaly_idx, Config.AMOUNT_COL],
                color='red', edgecolor='black', s=180,
                label='One-Class SVM Anomalileri', zorder=5)

    ax1.legend(loc='upper left')
    ax1.set_title(f'Kart {card_token} - İşlem Tutarları ve Anomaliler')
    ax1.grid(True, axis='x')

    # Alt grafik: Anomali skorları
    ax2.plot(card_data[Config.DATE_COL], card_model_data['ocsvm_score'],
             marker='s', color='purple', alpha=0.7, label='Anomali Skoru')
    ax2.set_xlabel('Tarih')
    ax2.set_ylabel('Anomali Skoru', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    # Eşik çizgisi
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, label='Normal/Anomali Eşiği')

    # Anomali noktalarını işaretle
    ax2.scatter(card_data.loc[anomaly_idx, Config.DATE_COL],
                card_model_data.loc[anomaly_idx, 'ocsvm_score'],
                color='red', edgecolor='black', s=100, zorder=5)

    ax2.legend(loc='upper left')
    ax2.set_title(f'Kart {card_token} - Anomali Skorları')
    ax2.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    _finish_plot(output_path)

    # İstatistikleri göster
    anomaly_count = card_model_data['ocsvm_flag'].sum()
    total_count = len(card_model_data)
    print(f"Kart {card_token} için istatistikler:")
    print(f"Toplam işlem: {total_count}")
    print(f"Tespit edilen anomali sayısı: {anomaly_count}")
    print(f"Anomali oranı: {anomaly_count/total_count:.2%}")

    # Anomali işlemlerin detaylarını göster
    if anomaly_count > 0:
        print("\nAnomali işlem detayları:")
        anomaly_details = card_data[card_model_data['ocsvm_flag'] == 1].copy()
        anomaly_details['anomaly_score'] = card_model_data.loc[anomaly_idx, 'ocsvm_score']

        display_cols = [Config.DATE_COL, Config.AMOUNT_COL, 'anomaly_score',
                       'amount_zscore', 'last_5_amount_avg', 'last_1h_tx_count']
        display_cols = [col for col in display_cols if col in anomaly_details.columns]

        print(anomaly_details[display_cols])
    return fig


def visualize_isolation_forest_results(transactions_data, model_data, card_token, output_path=None):
    """Isolation Forest sonuçlarını görselleştirme"""
    # Kart verisini filtrele
    card_mask = transactions_data[Config.USER_COL] == card_token
    card_data = transactions_data[card_mask].copy()
    card_model_data = model_data[card_mask].copy()

    # Veriyi tarihe göre sırala
    ordered = card_data.sort_values(Config.DATE_COL, kind='stable').index
    card_data = card_data.loc[ordered].reset_index(drop=True)
    card_model_data = card_model_data.loc[ordered].reset_index(drop=True)
    if card_data.empty:
        raise ValueError('Selected card does not exist.')

    # Görselleştirme
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Üst grafik: İşlem tutarları ve anomaliler
    ax1.plot(card_data[Config.DATE_COL], card_data[Config.AMOUNT_COL],
             marker='o', color='darkblue', label='Tutar', linewidth=2)
    ax1.set_xlabel('Tarih')
    ax1.set_ylabel('Tutar', color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Anomali noktalarını göster
    anomaly_idx = card_data.index[card_model_data['isoforest_flag'] == 1]
    ax1.scatter(card_data.loc[anomaly_idx, Config.DATE_COL],
                card_data.loc[anomaly_idx, Config.AMOUNT_COL],
                color='red', edgecolor='black', s=180,
                label='Isolation Forest Anomalileri', zorder=5)

    ax1.legend(loc='upper left')
    ax1.set_title(f'Kart {card_token} - İşlem Tutarları ve Anomaliler')
    ax1.grid(True, axis='x')

    # Alt grafik: Anomali skorları
    ax2.plot(card_data[Config.DATE_COL], card_model_data['isoforest_score'],
             marker='s', color='purple', alpha=0.7, label='Anomali Skoru')
    ax2.set_xlabel('Tarih')
    ax2.set_ylabel('Anomali Skoru', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    # Eşik çizgisi
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, label='Normal/Anomali Eşiği')

    # Anomali noktalarını işaretle
    ax2.scatter(card_data.loc[anomaly_idx, Config.DATE_COL],
                card_model_data.loc[anomaly_idx, 'isoforest_score'],
                color='red', edgecolor='black', s=100, zorder=5)

    ax2.legend(loc='upper left')
    ax2.set_title(f'Kart {card_token} - Anomali Skorları')
    ax2.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    _finish_plot(output_path)

    # İstatistikleri göster
    anomaly_count = card_model_data['isoforest_flag'].sum()
    total_count = len(card_model_data)
    print(f"Kart {card_token} için istatistikler:")
    print(f"Toplam işlem: {total_count}")
    print(f"Tespit edilen anomali sayısı: {anomaly_count}")
    print(f"Anomali oranı: {anomaly_count/total_count:.2%}")

    # Anomali işlemlerin detaylarını göster
    if anomaly_count > 0:
        print("\nAnomali işlem detayları:")
        anomaly_details = card_data[card_model_data['isoforest_flag'] == 1].copy()
        anomaly_details['anomaly_score'] = card_model_data.loc[anomaly_idx, 'isoforest_score']

        display_cols = [Config.DATE_COL, Config.AMOUNT_COL, 'anomaly_score',
                       'amount_zscore', 'last_5_amount_avg', 'last_1h_tx_count']
        display_cols = [col for col in display_cols if col in anomaly_details.columns]

        print(anomaly_details[display_cols])
    return fig


def visualize_card_transactions(transactions_data, card_token, output_path=None):
    """Kart işlemlerini görselleştirme"""
    card_data = transactions_data[transactions_data[Config.USER_COL] == card_token].copy()
    card_data = card_data.sort_values(Config.DATE_COL, kind='stable').reset_index(drop=True)
    if card_data.empty:
        raise ValueError('Selected card does not exist.')

    plt.figure(figsize=(12, 6))
    plt.plot(card_data[Config.DATE_COL], card_data[Config.AMOUNT_COL], marker='o', linestyle='-')
    plt.scatter(
        card_data[card_data['isoforest_flag'] == 1][Config.DATE_COL],
        card_data[card_data['isoforest_flag'] == 1][Config.AMOUNT_COL],
        color='red', s=100, label='Anomali'
    )
    plt.xlabel('Tarih')
    plt.ylabel('Tutar')
    plt.title(f'Kart: {card_token} - İşlem Tutarları (Kırmızı: Anomali)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig = _finish_plot(output_path)

    # Aynı kartta tespit edilen anomaliler
    print(f"\n{card_token} kartındaki anomali işlemler:")
    print(card_data[card_data['isoforest_flag'] == 1][[Config.DATE_COL, Config.AMOUNT_COL]])

    # Tüm işlemleri göster
    print(f"\n{card_token} kartının tüm işlemleri:")
    print(card_data[[Config.DATE_COL, Config.AMOUNT_COL, 'isoforest_flag']])

    # Anomali işlemlerin detayları
    if any(card_data['isoforest_flag'] == 1):
        print(f"\n{card_token} kartındaki anomali işlem detayları:")
        print(card_data[card_data['isoforest_flag'] == 1][[
            Config.DATE_COL, Config.AMOUNT_COL, 'daily_mean',
            'amount_zscore', 'amount_daily_zscore', 'daily_tx_count'
        ]])
    return fig


def visualize_card_transactions_svm(transactions_data, card_token, output_path=None):
    """Kart işlemlerini görselleştirme"""
    card_data = transactions_data[transactions_data[Config.USER_COL] == card_token].copy()
    card_data = card_data.sort_values(Config.DATE_COL, kind='stable').reset_index(drop=True)
    if card_data.empty:
        raise ValueError('Selected card does not exist.')

    plt.figure(figsize=(12, 6))
    plt.plot(card_data[Config.DATE_COL], card_data[Config.AMOUNT_COL], marker='o', linestyle='-')
    plt.scatter(
        card_data[card_data['isoforest_flag'] == 1][Config.DATE_COL],
        card_data[card_data['isoforest_flag'] == 1][Config.AMOUNT_COL],
        color='red', s=100, label='Anomali'
    )
    plt.xlabel('Tarih')
    plt.ylabel('Tutar')
    plt.title(f'Kart: {card_token} - İşlem Tutarları (Kırmızı: Anomali)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig = _finish_plot(output_path)

    # Aynı kartta tespit edilen anomaliler
    print(f"\n{card_token} kartındaki anomali işlemler:")
    print(card_data[card_data['isoforest_flag'] == 1][[Config.DATE_COL, Config.AMOUNT_COL]])

    # Tüm işlemleri göster
    print(f"\n{card_token} kartının tüm işlemleri:")
    print(card_data[[Config.DATE_COL, Config.AMOUNT_COL, 'isoforest_flag']])

    # Anomali işlemlerin detayları
    if any(card_data['isoforest_flag'] == 1):
        print(f"\n{card_token} kartındaki anomali işlem detayları:")
        print(card_data[card_data['isoforest_flag'] == 1][[
            Config.DATE_COL, Config.AMOUNT_COL, 'daily_mean',
            'amount_zscore', 'amount_daily_zscore', 'daily_tx_count'
        ]])
    return fig


def visualize_rolling_zscore(transactions_data, card_token, window=5, output_path=None):
    card_data = transactions_data[transactions_data[Config.USER_COL] == card_token].copy()
    if card_data.empty:
        raise ValueError('Selected card does not exist.')
    card_data = card_data.sort_values(Config.DATE_COL).reset_index(drop=True)


    # Rolling mean/std ve z-score hesapla (window=5 veya 7, sen karar ver)
    window_size = window
    card_data['rolling_mean'] = card_data[Config.AMOUNT_COL].rolling(window=window_size, min_periods=2).mean()
    card_data['rolling_std']  = card_data[Config.AMOUNT_COL].rolling(window=window_size, min_periods=2).std()
    card_data['z_score'] = (card_data[Config.AMOUNT_COL] - card_data['rolling_mean']) / card_data['rolling_std']


    #Birleştirilmiş hali
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Sol eksen: Amount ve Rolling Mean
    ax1.plot(card_data[Config.DATE_COL], card_data[Config.AMOUNT_COL], marker='o', color='darkblue', label='Tutar')
    ax1.plot(card_data[Config.DATE_COL], card_data['rolling_mean'], linestyle='--', color='darkorange', label=f'Rolling Mean ({window_size})')
    ax1.set_xlabel('Tarih')
    ax1.set_ylabel('Tutar', color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Z-score anomali noktalarını büyük kırmızı ile göster
    anomaly_idx = card_data[card_data['z_score'].abs() > 1.5].index
    ax1.scatter(card_data.loc[anomaly_idx, Config.DATE_COL], card_data.loc[anomaly_idx, Config.AMOUNT_COL],
                color='red', edgecolor='black', s=180, label='Z-score Anomali (>1.5)', zorder=5)

    # Sağ eksen: Z-score
    ax2 = ax1.twinx()
    ax2.plot(card_data[Config.DATE_COL], card_data['z_score'], marker='s', color='crimson', label='Z-Score')
    ax2.set_ylabel('Z-Score', color='green')
    ax2.axhline(1.5, color='red', linestyle='--', linewidth=1, label='Z-score > 1.5')
    ax2.axhline(-1.5, color='red', linestyle='--', linewidth=1)
    ax2.axhline(0, color='gray', linestyle=':')

    ax2.tick_params(axis='y', labelcolor='green')

    # Legend'ları birleştir
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title(f'Kart {card_token} - Amount, Rolling Mean ve Z-Score')
    plt.xticks(rotation=45)
    plt.grid(True, axis='x')
    plt.tight_layout()
    return _finish_plot(output_path)


class SimulationRules:
    """Explicit demonstration policy; thresholds are not fraud calibration."""
    REPEAT_WINDOW_SECONDS = 600
    REPEAT_COUNT = 2
    RAPID_COUNT = 3
    SHORT_GAP_SECONDS = 30


class FraudDetectionTester:
    """Interactive, in-memory what-if analysis with the original rule checks.

    Uses the same retrospective feature formulas as the batch pipeline.
    Session transactions are simulations and never modify the input CSV.
    Rule points explain hand-written heuristics, not the model's reasoning.
    """
    def __init__(self, transactions_df, model_features, iso_model, ocsvm_model=None):
        self.df = validate_data(transactions_df)
        self.model_features = model_features
        self.iso_model = iso_model
        self.ocsvm_model = ocsvm_model
        self.session_transactions = {}
        self.session_start_times = {}
        self._pending = {}
        self._last_real_time = {}

    def get_user_historical_data(self, card_token):
        user_data = self.df[self.df[Config.USER_COL] == card_token].copy()
        if user_data.empty:
            print(f'{card_token} kartı için geçmiş veri bulunamadı.')
            return None
        session = self.session_transactions.get(card_token, [])
        if session:
            user_data = pd.concat([user_data, pd.DataFrame(session)], ignore_index=True)
        return user_data.sort_values(Config.DATE_COL, kind='stable')

    def calculate_test_features(self, card_token, new_amount, new_ip=None, new_currency='TRY'):
        user_data = self.get_user_historical_data(card_token)
        if user_data is None:
            return None, None
        if not np.isfinite(new_amount) or new_amount <= 0:
            raise ValueError('Test amount must be positive and finite.')
        # Start after the entire supplied history; never insert before future
        # historical records, which the old random-today clock could do.
        all_sessions = [tx for values in self.session_transactions.values() for tx in values]
        last_timestamp = max([self.df[Config.DATE_COL].max()] +
                             [tx[Config.DATE_COL] for tx in all_sessions])
        now = datetime.now()
        elapsed = max(0.001, (now-self._last_real_time[card_token]).total_seconds()) if card_token in self._last_real_time else 30.0
        current_time = last_timestamp + timedelta(seconds=elapsed)
        ip = new_ip or user_data[Config.IP_COL].iloc[-1]
        candidate = {
            Config.USER_COL: card_token, Config.AMOUNT_COL: float(new_amount),
            Config.IP_COL: ip, Config.CURRENCY_COL: new_currency,
            Config.RESPONSE_COL: '00', Config.DATE_COL: current_time,
        }
        if 'externalCustomerId' in self.df.columns:
            customer = user_data['externalCustomerId'].dropna()
            candidate['externalCustomerId'] = customer.iloc[-1] if len(customer) else ''
        # Include other cards too: customer/card sharing features must use the
        # same context as training. Select the candidate through its row index.
        combined = pd.concat([self.df, pd.DataFrame(all_sessions), pd.DataFrame([candidate])], ignore_index=True)
        marker = '__simulator_candidate__'
        combined[marker] = False
        combined.loc[combined.index[-1], marker] = True
        with redirect_stdout(io.StringIO()):
            engineered = engineer_features(combined)
        candidate_features = engineered.loc[engineered[marker]].iloc[0]
        features = candidate_features[self.model_features].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict()
        features['zscore_anomaly_flag'] = int(candidate_features['zscore_anomaly_flag'])
        features['dynamic_amount_flag'] = int(candidate_features['dynamic_amount_flag'])
        session = self.session_transactions.get(card_token, [])
        # True consecutive suffix, not a count of all matches in the last five.
        consecutive = 1
        for tx in reversed(session[-4:]):
            age = (current_time - tx[Config.DATE_COL]).total_seconds()
            if (tx[Config.AMOUNT_COL] != new_amount
                    or tx[Config.CURRENCY_COL] != new_currency
                    or not 0 <= age <= SimulationRules.REPEAT_WINDOW_SECONDS):
                break
            consecutive += 1
        features['consecutive_same_amount'] = consecutive
        features['rapid_session_transactions'] = 1 + sum(
            0 <= (current_time-tx[Config.DATE_COL]).total_seconds() <= 600 for tx in session)
        features['seconds_since_last_test'] = (current_time-session[-1][Config.DATE_COL]).total_seconds() if session else 0
        self._pending[card_token] = candidate
        self.session_start_times.setdefault(card_token, current_time)
        print(f'\n{card_token} - test zamanı: {current_time}')
        print(f'Orijinal işlem: {len(self.df[self.df[Config.USER_COL] == card_token])}; oturum: {len(session)}')
        print('FEATURE ANALİZLERİ (batch akışıyla aynı hesaplama):')
        for name, value in features.items():
            print(f'  {name}: {value:.4f}')
        return features, user_data

    def add_session_transaction(self, card_token, amount, ip=None, currency='TRY'):
        if card_token not in self._pending:
            raise ValueError('Calculate and score a candidate before committing it.')
        candidate = self._pending[card_token]
        if float(amount) != candidate[Config.AMOUNT_COL] or currency != candidate[Config.CURRENCY_COL] or (ip is not None and ip != candidate[Config.IP_COL]):
            raise ValueError('Session transaction differs from the scored candidate.')
        self.session_transactions.setdefault(card_token, []).append(self._pending.pop(card_token))
        self._last_real_time[card_token] = datetime.now()

    def predict_anomaly(self, features):
        feature_df = pd.DataFrame([features])
        prediction = self.iso_model.predict(feature_df[self.model_features])
        score = self.iso_model.decision_function(feature_df[self.model_features])
        return bool(prediction[0] == -1), float(score[0]), feature_df
    def evaluate_rules(self, features, user_data, new_amount, currency='TRY'):
        """Session rules are reported separately from the four detection methods."""
        reasons = []
        if features.get('consecutive_same_amount', 0) >= SimulationRules.REPEAT_COUNT:
            reasons.append('10 dakika içinde peş peşe aynı tutar ve para birimi')
        if features.get('rapid_session_transactions', 0) >= SimulationRules.RAPID_COUNT:
            reasons.append('10 dakika içinde en az üç oturum işlemi')
        gap = features.get('seconds_since_last_test', 0)
        if 0 < gap < SimulationRules.SHORT_GAP_SECONDS:
            reasons.append('Oturum işlemleri arasında 30 saniyeden kısa süre')
        return bool(reasons), reasons

    def evaluate_candidate(self, features, user_data, new_amount, currency='TRY'):
        """Default review policy: union of available methods OR session rules.

        This is an explicit public-demo policy, not a reconstructed historical
        business decision rule. Statistical flags use the original batch code.
        """
        model_flag, score, matrix = self.predict_anomaly(features)
        ocsvm_flag, ocsvm_score = None, None
        if self.ocsvm_model is not None:
            ocsvm_flag = bool(self.ocsvm_model.predict(matrix[self.model_features])[0] == -1)
            ocsvm_score = float(self.ocsvm_model.decision_function(matrix[self.model_features])[0])
        zscore_flag = bool(features.get('zscore_anomaly_flag', abs(features['amount_zscore']) > 1.0))
        dynamic_flag = bool(features.get('dynamic_amount_flag', False))
        methods = {'isolation_forest': bool(model_flag), 'one_class_svm': ocsvm_flag,
                   'zscore': zscore_flag, 'dynamic_threshold': dynamic_flag}
        method_union = any(value is True for value in methods.values())
        rule_flag, reasons = self.evaluate_rules(features, user_data, new_amount, currency)
        return {
            'model_flag': model_flag, 'model_score': score,
            'ocsvm_flag': ocsvm_flag, 'ocsvm_score': ocsvm_score,
            'zscore_flag': zscore_flag, 'dynamic_flag': dynamic_flag,
            'triggered_methods': [name for name, value in methods.items() if value is True],
            'method_union_flag': method_union,
            'rule_flag': rule_flag, 'rule_reasons': reasons,
            'review_flag': bool(method_union or rule_flag),
        }

    def analyze_anomaly_reasons(self, features, user_data, new_amount):
        """Anomali nedenlerini analiz et"""
        print("│")
        print("│")
        print(f"└─ ANOMALİ ANALİZİ:")
        reasons = []
        risk_score = 0

        # Z-score kontrolü
        if abs(features['amount_zscore']) > 2:
            reasons.append(f"Yüksek Z-score ({features['amount_zscore']:.3f})")
            risk_score += 3

        # Son 5 işlem ortalamasından sapma
        if features['last_5_amount_avg'] > 0:
            avg_deviation = abs(new_amount - features['last_5_amount_avg']) / features['last_5_amount_avg']
            if avg_deviation > 0.5:  # %50'den fazla sapma
                reasons.append(f"Son 5 işlem ortalamasından büyük sapma (%{avg_deviation*100:.1f})")
                risk_score += 2

        # Varyans analizi - yeni risk faktörü
        last_10 = user_data[Config.AMOUNT_COL].tail(10)
        mean_10 = last_10.mean()
        std_10 = last_10.std()
        if std_10 < 300 and abs(new_amount - mean_10) > 0.7 * mean_10:
            reasons.append(f"Düşük varyanslı kullanıcı: Standart sapma {std_10:.2f}, ama yeni işlem {new_amount:.2f} (ortalama {mean_10:.2f}'den %70'ten fazla sapma!)")
            risk_score += 3
        elif std_10 > 0 and abs(new_amount - mean_10) > 2 * std_10:
            reasons.append(f"İşlem, son 10 işlem ortalamasının {2} katı standart sapmadan daha fazla sapma!")
            risk_score += 2


        # Yüksek işlem frekansı
        if features['last_10min_tx_count'] > 3:
            reasons.append(f"Son 10 dakikada yüksek işlem sayısı ({features['last_10min_tx_count']})")
            risk_score += 2

        # TEST OTURUMU RİSKLERİ
        if features.get('consecutive_same_amount', 0) >= 3:
            reasons.append(f"Peş peşe aynı tutarda işlem ({features['consecutive_same_amount']} kez)")
            risk_score += 4  # Yüksek risk

        if features.get('rapid_session_transactions', 0) >= 3:
            reasons.append(f"10 dakika içinde çok fazla işlem ({features['rapid_session_transactions']} adet)")
            risk_score += 3

        # Yeni IP
        if features['is_new_ip']:
            reasons.append("Yeni IP adresi kullanımı")
            risk_score += 1

        # Çok hızlı işlem
        if features['time_since_last_tx'] < 60:  # 1 dakikadan az
            reasons.append(f"Çok hızlı ardışık işlem ({features['time_since_last_tx']:.0f} saniye)")
            risk_score += 2

        # Günlük limite yaklaşma
        daily_avg = user_data[Config.AMOUNT_COL].mean()
        if features['daily_amount_sum'] > daily_avg * 5:  # Günlük ortalamadan 5 kat fazla
            reasons.append("Günlük toplamın geçmiş işlem ortalamasının 5 katını aşması")
            risk_score += 3

        # Kullanıcı max tutarını aşma
        user_max = user_data[Config.AMOUNT_COL].max()
        if new_amount > user_max * 1.2:  # %20 fazla
            reasons.append(f"Kullanıcı max tutarının aşılması (Max: {user_max:.2f})")
            risk_score += 2

        if reasons:
            print("├── Kural bazlı inceleme sinyalleri (model açıklaması değildir):")
            for reason in reasons:
                print(f"│   • {reason}")

            # Risk seviyesi belirleme
            if risk_score >= 6:
                risk_level = "🔴 YÜKSEK HEURİSTİK PUAN - İNCELEME ADAYI"
            elif risk_score >= 4:
                risk_level = "🟠 YÜKSEK RİSK - MANUEL İNCELEME GEREKLİ"
            elif risk_score >= 2:
                risk_level = "🟡 ORTA RİSK - DİKKAT"
            else:
                risk_level = "🟢 DÜŞÜK RİSK"

            print(f"├── Heuristik Puan: {risk_score} (kalibre edilmemiş)")
            print(f"└── Simülasyon değerlendirmesi: {risk_level}")
        else:
            print("├── Belirgin anomali nedeni tespit edilemedi")
            print(f"└── Heuristik Puan: 0")

        return reasons, risk_score

    def run_interactive_test(self):
        """İnteraktif test başlat"""
        print("-> FRAUD DETECTION FEATURE TEST SİMÜLATÖRÜ")
        print("─" * 50)
        print("-> Kontrol Edilen Özellikler:")
        print("├── Peş peşe aynı tutar kontrolü")
        print("├── Hızlı ardışık işlem tespiti")
        print("├── Risk puanlama sistemi")
        print("└── Test oturumu takibi")
        print("─" * 50)

        while True:
            try:
                # Kullanıcıdan girdi al
                card_token = input("\n-> Kart Token giriniz (çıkmak için 'q'): ").strip()
                if card_token.lower() == 'q':
                    break

                # Kart kontrolü
                if card_token not in self.df[Config.USER_COL].values:
                    print(f"❌ {card_token} kartı sistemde bulunamadı!")
                    available_cards = self.df[Config.USER_COL].unique()[:10]
                    print(f"Mevcut kartlardan bazıları: {list(available_cards)}")
                    continue

                # Tutar girişi
                amount_str = input("-> İşlem tutarını giriniz: ").strip()
                try:
                    new_amount = float(amount_str)
                    if not np.isfinite(new_amount) or new_amount <= 0:
                        print("❌ Tutar pozitif olmalıdır!")
                        continue
                except ValueError:
                    print("❌ Geçerli bir tutar giriniz!")
                    continue

                # IP adresi (opsiyonel)
                new_ip = input("🌐 IP adresi (opsiyonel, Enter ile geç): ").strip()
                if not new_ip:
                    new_ip = None

                print("\n" + "─"*60)

                # Feature hesaplama
                features, user_data = self.calculate_test_features(card_token, new_amount, new_ip)
                if features is None:
                    continue

                decision = self.evaluate_candidate(features, user_data, new_amount)
                is_anomaly = decision['model_flag']
                anomaly_score = decision['model_score']
                rule_based_anomaly = decision['rule_flag']
                rule_reasons = decision['rule_reasons']
                final_anomaly = decision['review_flag']

                # Sonuçları göster

                print('\n🚨 ANOMALİ TESPİT EDİLDİ!' if final_anomaly else '✅ MODEL/KURALLAR İŞARETLEMEDİ')
                print(f"├── Model Anomali Skoru: {anomaly_score:.4f}")
                print(f"├── Isolation Forest: {'İŞARETLEDİ' if is_anomaly else 'İŞARETLEMEDİ'}")
                print(f"├── OCSVM: {decision['ocsvm_flag']} | Z-score: {decision['zscore_flag']} | Dynamic: {decision['dynamic_flag']}")
                print(f"├── İşaretleyen yöntemler: {', '.join(decision['triggered_methods']) or 'Yok'}")
                if rule_based_anomaly:
                    print(f"├── Kural Bazlı Tespit: ANOMALİ ({', '.join(rule_reasons)})")
                print(f"└── Karar: {'ANOMALİ' if final_anomaly else 'NORMAL'}")

                if final_anomaly:
                    reasons, risk_score = self.analyze_anomaly_reasons(features, user_data, new_amount)

                    # Risk seviyeleri (güncellendi)
                    if risk_score >= 6:
                        risk_level = "🔴 YÜKSEK HEURİSTİK PUAN - İNCELEME ADAYI"
                    elif risk_score >= 4:
                        risk_level = "🟠 YÜKSEK RİSK - MANUEL İNCELEME"
                    elif risk_score >= 2:
                        risk_level = "🟡 ORTA RİSK - DİKKAT"
                    else:
                        risk_level = "🟢 DÜŞÜK RİSK"
                else:
                    if anomaly_score < -0.3:
                        risk_level = "🟡 ORTA RİSK (Model skoru yüksek)"
                    elif anomaly_score < -0.1:
                        risk_level = "🟢 DÜŞÜK RİSK"
                    else:
                        risk_level = "🟢 DÜŞÜK RİSK"

                # İşlemi test oturumuna ekle
                self.add_session_transaction(card_token, new_amount, new_ip)
                print(f"\n─> İşlem test oturumuna kaydedildi")

                print(f"\n─> KARŞILAŞTIRMALI ANALİZ:")
                original_data_count = len(self.df[self.df[Config.USER_COL] == card_token])
                print(f"├── Orijinal kullanıcı ortalaması: {self.df[self.df[Config.USER_COL] == card_token][Config.AMOUNT_COL].mean():.2f}")
                print(f"├── Kullanıcı medyanı: {user_data[Config.AMOUNT_COL].median():.2f}")
                print(f"├── Kullanıcı max işlemi: {user_data[Config.AMOUNT_COL].max():.2f}")
                print(f"├── Yeni işlem / Ortalama: {(new_amount / user_data[Config.AMOUNT_COL].mean() if user_data[Config.AMOUNT_COL].mean() else float('nan')):.2f}x")
                print(f"├── Orijinal işlem sayısı: {original_data_count}")
                print(f"└── Bu oturumda test edilen: {len(self.session_transactions.get(card_token, []))}")

                print("\n" + "─"*60)

            except (KeyboardInterrupt, EOFError):
                print("\n\n─> Test sonlandırıldı!")
                break
            except Exception as e:
                print(f"❌ Hata oluştu: {str(e)}")
                continue


def start_fraud_test(transactions_data, model_features, iso_model, ocsvm_model=None):
    tester = FraudDetectionTester(transactions_data, model_features, iso_model, ocsvm_model)
    tester.run_interactive_test()
    return tester


def run_enhanced_analysis(transactions_data, model_data, config, output_dir=None, skip_plots=False):
    results = enhanced_results_analysis(transactions_data, model_data, config)
    if not skip_plots:
        path = Path(output_dir) / 'dashboard.png' if output_dir else None
        create_comprehensive_visualizations(transactions_data, model_data, config, path)
    detailed_anomaly_examples(transactions_data, config, n_examples=5)
    model_performance_summary(model_data)
    return results


def pick_example_card(transactions_data):
    # Preserve specific-card analysis without original hardcoded identifiers.
    ranking = transactions_data.groupby(Config.USER_COL)['isoforest_flag'].sum()
    return ranking.sort_values(ascending=False, kind='stable').index[0]


def run_pipeline(input_path, output_dir='outputs', skip_plots=False, card_token=None):
    output_dir = Path(output_dir)
    feature_path = output_dir / 'transactions_with_features.csv'
    results_path = output_dir / 'transactions_scored.csv'
    if Path(input_path).resolve() in {feature_path.resolve(), results_path.resolve()}:
        raise ValueError('Input must not be one of the output files.')
    feature_df = perform_feature_engineering(input_path, feature_path)
    X, df = prepare_model_data(feature_path, MODEL_FEATURES)
    X, iso_model = train_isolation_forest(X, MODEL_FEATURES)
    X, svm_model = train_oneclass_svm(X, MODEL_FEATURES)
    for col in ['isoforest_prediction', 'isoforest_score', 'isoforest_flag',
                'ocsvm_prediction', 'ocsvm_score', 'ocsvm_flag']:
        df[col] = X[col]
    method_columns = ['isoforest_flag', 'ocsvm_flag', 'zscore_anomaly_flag', 'dynamic_amount_flag']
    df['method_union_flag'] = df[method_columns].any(axis=1).astype(int)
    df['method_flag_count'] = df[method_columns].sum(axis=1)
    save_data(df, results_path)
    card = card_token or pick_example_card(df)
    if card not in df[Config.USER_COL].values:
        raise ValueError(f'Unknown selected card: {card}')
    plot_dir = output_dir / 'plots'
    with (output_dir/'analysis_report.txt').open('w', encoding='utf-8') as report, redirect_stdout(report):
        analyze_results(df, X)
        results = run_enhanced_analysis(df, X, Config, plot_dir, skip_plots)
        if not skip_plots:
            visualize_card_transactions(df, card, plot_dir/'card_transactions_if.png')
            visualize_card_transactions_svm(df, card, plot_dir/'card_transactions_svm.png')
            visualize_rolling_zscore(df, card, output_path=plot_dir/'card_rolling_zscore.png')
            visualize_isolation_forest_results(df, X, card, plot_dir/'card_isolation_forest.png')
            visualize_one_class_svm_results(df, X, card, plot_dir/'card_one_class_svm.png')
    summary = {
        'transactions': len(df), 'cards': int(df[Config.USER_COL].nunique()),
        'analysis_mode': 'retrospective; fit and score on the same dataset',
        'score_direction': 'lower decision_function = more anomalous; not a probability',
        'isoforest_anomalies': int(X.isoforest_flag.sum()),
        'ocsvm_anomalies': int(X.ocsvm_flag.sum()),
        'dynamic_anomalies': int(df.dynamic_amount_flag.sum()),
        'zscore_threshold_counts': {str(t): int((X.amount_zscore.abs() > t).sum()) for t in Config.Z_SCORE_THRESHOLDS},
        'method_union_count': int(df.method_union_flag.sum()),
        'combination_policy': 'any of IF, OCSVM, rolling Z-score or dynamic threshold; review candidate only',
        'all_three_overlap': int(((df.dynamic_amount_flag == 1) & (X.isoforest_flag == 1) & (X.ocsvm_flag == 1)).sum()),
    }
    (output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Ayrıntılı rapor: {output_dir / "analysis_report.txt"}')
    return df, X, iso_model, svm_model


def main(argv=None):
    parser = argparse.ArgumentParser(description='Behavioral fraud analysis and interactive simulator')
    parser.add_argument('--input', default=Config.DATA_PATHS['multi_data'])
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--skip-plots', '--skip_plots', action='store_true')
    parser.add_argument('--card', help='Select a card for detailed plots')
    parser.add_argument('--interactive', action='store_true', help='Start the original what-if simulator after analysis')
    args = parser.parse_args(argv)
    try:
        df, X, iso_model, svm_model = run_pipeline(args.input, args.output_dir, args.skip_plots, args.card)
        if args.interactive:
            start_fraud_test(df, MODEL_FEATURES, iso_model, svm_model)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
