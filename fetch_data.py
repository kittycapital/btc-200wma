#!/usr/bin/env python3
"""
BTC 200-Week Moving Average Dashboard - Data Processor
Calculates 200WMA, detects touch events, computes forward returns.
Data source: BTC_USD.csv (daily OHLCV)
"""

import csv
import json
import os
import math
from datetime import datetime, timedelta
from collections import OrderedDict

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'BTC_USD.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'btc_200wma.json')

TOUCH_THRESHOLD = 0.05      # 5% above or below 200WMA
TOUCH_MIN_GAP_WEEKS = 20    # Minimum weeks between separate touch events
TOUCH_EXIT_THRESHOLD = 0.20 # 20% above 200WMA = event ended

FORWARD_PERIODS = {
    '1mo': 4,
    '3mo': 13,
    '6mo': 26,
    '12mo': 52
}

SIMULATION_AMOUNT = 10000  # $10K

# ============================================================
# READ CSV
# ============================================================
def read_btc_csv(path):
    """Read BTC daily CSV -> list of (date, close)"""
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                date_str = row['Date'].strip()
                close = float(row['Close'].strip())
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                rows.append((dt, close))
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda x: x[0])
    return rows

# ============================================================
# RESAMPLE TO WEEKLY
# ============================================================
def resample_weekly(daily_data):
    """
    Resample daily data to weekly (Sunday close).
    Returns list of (week_end_date, close_price)
    """
    weekly = OrderedDict()
    for dt, close in daily_data:
        # Find the Sunday of this week
        days_until_sunday = (6 - dt.weekday()) % 7
        week_end = dt + timedelta(days=days_until_sunday)
        week_key = week_end.strftime('%Y-%m-%d')
        # Keep the latest daily close for each week
        weekly[week_key] = (week_end, close)

    result = []
    for key in weekly:
        result.append(weekly[key])
    result.sort(key=lambda x: x[0])
    return result

# ============================================================
# CALCULATE 200WMA
# ============================================================
def calculate_200wma(weekly_data):
    """
    Calculate 200-week simple moving average.
    Returns list of (date, close, wma200) starting from week 200.
    """
    result = []
    closes = [w[1] for w in weekly_data]

    for i in range(len(weekly_data)):
        if i < 199:  # Need 200 data points
            result.append((weekly_data[i][0], weekly_data[i][1], None))
        else:
            window = closes[i - 199:i + 1]
            wma = sum(window) / 200
            result.append((weekly_data[i][0], weekly_data[i][1], wma))

    return result

# ============================================================
# DETECT TOUCH EVENTS
# ============================================================
def detect_touches(wma_data):
    """
    Detect touch events where BTC price comes within TOUCH_THRESHOLD of 200WMA.
    Groups consecutive weeks into single events.
    Returns list of touch events.
    """
    # Filter to only weeks with valid 200WMA
    valid = [(dt, close, wma) for dt, close, wma in wma_data if wma is not None]

    events = []
    in_event = False
    current_event = []

    for i, (dt, close, wma) in enumerate(valid):
        pct_from_wma = (close - wma) / wma  # positive = above, negative = below

        if not in_event:
            # Start event if price is within threshold
            if pct_from_wma <= TOUCH_THRESHOLD:
                in_event = True
                current_event = [(dt, close, wma, pct_from_wma)]
        else:
            # Continue event or end it
            if pct_from_wma <= TOUCH_EXIT_THRESHOLD:
                current_event.append((dt, close, wma, pct_from_wma))
            else:
                # Event ended - process it
                if current_event:
                    events.append(current_event)
                current_event = []
                in_event = False

    # Handle event still in progress at end of data
    if current_event:
        events.append(current_event)

    # For each event, use the FIRST week as the touch date (first entry into zone)
    touch_points = []
    last_touch_date = None

    for event in events:
        # Use first week of event as touch point
        touch_date, touch_price, touch_wma, touch_pct = event[0]

        # Find deepest point too for reference
        min_week = min(event, key=lambda x: x[3])
        min_date, min_price, min_wma, min_pct = min_week

        # Check minimum gap between events
        if last_touch_date and (touch_date - last_touch_date).days < TOUCH_MIN_GAP_WEEKS * 7:
            continue

        # Duration at or below 200WMA
        below_weeks = sum(1 for w in event if w[3] <= 0)

        # Check if still ongoing (last data point is still in the event)
        still_active = event[-1] == event[-1]  # always true initially
        # Really check: is the last event data point the last in the full dataset?
        last_event_date = event[-1][0]

        touch_points.append({
            'date': touch_date,
            'price': touch_price,
            'wma200': touch_wma,
            'pct_from_wma': touch_pct,
            'min_date': min_date,
            'min_price': min_price,
            'min_pct': min_pct,
            'event_start': event[0][0],
            'event_end': event[-1][0],
            'duration_weeks': len(event),
            'below_weeks': below_weeks,
            'still_active': False  # will be updated below
        })

        last_touch_date = touch_date

    # Mark the last event as active if it includes recent data
    if touch_points:
        last_event_end = touch_points[-1]['event_end']
        latest_data_date = valid[-1][0]
        if (latest_data_date - last_event_end).days <= 14:
            touch_points[-1]['still_active'] = True

    return touch_points

# ============================================================
# CALCULATE FORWARD RETURNS
# ============================================================
def calculate_forward_returns(touch_points, wma_data):
    """
    For each touch point, calculate returns at 1mo, 3mo, 6mo, 12mo.
    """
    # Build lookup: date -> close
    date_close = {dt: close for dt, close, wma in wma_data}
    all_dates = sorted(date_close.keys())

    def find_closest_date(target_date):
        """Find closest available date to target"""
        closest = None
        min_diff = timedelta(days=999)
        for d in all_dates:
            diff = abs(d - target_date)
            if diff < min_diff:
                min_diff = diff
                closest = d
        return closest if min_diff.days <= 14 else None

    for tp in touch_points:
        tp['returns'] = {}
        tp['sim_values'] = {}
        touch_date = tp['date']
        touch_price = tp['price']

        for label, weeks in FORWARD_PERIODS.items():
            future_date = touch_date + timedelta(weeks=weeks)
            closest = find_closest_date(future_date)

            if closest and closest <= all_dates[-1]:
                future_price = date_close[closest]
                ret = (future_price - touch_price) / touch_price
                sim_value = SIMULATION_AMOUNT * (1 + ret)
                tp['returns'][label] = round(ret * 100, 1)  # percentage
                tp['sim_values'][label] = round(sim_value)
            else:
                tp['returns'][label] = None
                tp['sim_values'][label] = None

    return touch_points

# ============================================================
# BUILD OUTPUT JSON
# ============================================================
def build_output(wma_data, touch_points):
    """Build the final JSON output."""
    # Weekly chart data (date, close, 200wma)
    chart_data = OrderedDict()
    for dt, close, wma in wma_data:
        key = dt.strftime('%Y-%m-%d')
        chart_data[key] = {
            'close': round(close, 2),
            'wma200': round(wma, 2) if wma else None
        }

    # Current values
    latest = wma_data[-1]
    current_price = latest[1]
    current_wma = latest[2]
    pct_above = ((current_price - current_wma) / current_wma * 100) if current_wma else 0

    # Touch events for table
    touches = []
    for tp in touch_points:
        touches.append({
            'date': tp['date'].strftime('%Y-%m-%d'),
            'date_label': tp['date'].strftime('%b %Y'),
            'price': round(tp['price'], 2),
            'wma200': round(tp['wma200'], 2),
            'pct_from_wma': round(tp['pct_from_wma'] * 100, 1),
            'min_date': tp['min_date'].strftime('%Y-%m-%d'),
            'min_price': round(tp['min_price'], 2),
            'min_pct': round(tp['min_pct'] * 100, 1),
            'duration_weeks': tp['duration_weeks'],
            'below_weeks': tp['below_weeks'],
            'returns': tp['returns'],
            'sim_values': tp['sim_values'],
            'is_current': tp['still_active']
        })

    # Statistics
    completed_touches = [t for t in touches if not t['is_current']]
    total_touches = len(touches)

    stats = {
        'total_touches': total_touches,
        'completed_touches': len(completed_touches),
        'hit_rate_12mo': None,
        'avg_return_12mo': None,
        'avg_return_6mo': None,
        'avg_return_3mo': None,
        'avg_return_1mo': None,
    }

    for period in ['1mo', '3mo', '6mo', '12mo']:
        returns = [t['returns'][period] for t in completed_touches if t['returns'].get(period) is not None]
        if returns:
            stats[f'avg_return_{period}'] = round(sum(returns) / len(returns), 1)
            positive = sum(1 for r in returns if r > 0)
            stats[f'hit_rate_{period}'] = round(positive / len(returns) * 100)

    output = {
        'metadata': {
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
            'source': 'BTC_USD.csv',
            'wma_period': 200,
            'touch_threshold_pct': TOUCH_THRESHOLD * 100,
            'simulation_amount': SIMULATION_AMOUNT,
            'total_weeks': len(wma_data),
            'first_wma_date': next((dt.strftime('%Y-%m-%d') for dt, c, w in wma_data if w), None),
        },
        'current': {
            'price': round(current_price, 2),
            'wma200': round(current_wma, 2) if current_wma else None,
            'pct_above_wma': round(pct_above, 1),
            'date': latest[0].strftime('%Y-%m-%d'),
        },
        'chart_data': chart_data,
        'touches': touches,
        'stats': stats,
    }

    return output

# ============================================================
# MAIN
# ============================================================
def main():
    print("📊 BTC 200WMA Dashboard - Data Processor")
    print("=" * 50)

    # 1. Read CSV
    print(f"Reading {CSV_PATH}...")
    daily = read_btc_csv(CSV_PATH)
    print(f"  → {len(daily)} daily records ({daily[0][0].date()} ~ {daily[-1][0].date()})")

    # 2. Resample to weekly
    weekly = resample_weekly(daily)
    print(f"  → {len(weekly)} weekly records")

    # 3. Calculate 200WMA
    wma_data = calculate_200wma(weekly)
    valid_wma = [(d, c, w) for d, c, w in wma_data if w is not None]
    print(f"  → 200WMA available from {valid_wma[0][0].date()} ({len(valid_wma)} weeks)")

    # 4. Detect touch events
    touch_points = detect_touches(wma_data)
    print(f"\n🔍 Touch Events Detected: {len(touch_points)}")
    for tp in touch_points:
        pct = tp['pct_from_wma'] * 100
        print(f"  {tp['date'].strftime('%b %Y'):>10} | Price: ${tp['price']:>10,.0f} | 200WMA: ${tp['wma200']:>10,.0f} | {pct:>+.1f}% | {tp['duration_weeks']}w")

    # 5. Calculate forward returns
    touch_points = calculate_forward_returns(touch_points, wma_data)

    # 6. Build and save output
    output = build_output(wma_data, touch_points)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Output saved to {OUTPUT_PATH}")
    print(f"   Current BTC: ${output['current']['price']:,.0f}")
    print(f"   Current 200WMA: ${output['current']['wma200']:,.0f}")
    print(f"   % Above 200WMA: {output['current']['pct_above_wma']:+.1f}%")

if __name__ == '__main__':
    main()
