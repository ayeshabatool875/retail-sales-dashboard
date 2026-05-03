import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ── 1. Generate realistic retail data ──────────────────────────────────────
np.random.seed(42)
days = 180
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(days)]

products = ['Electronics', 'Clothing', 'Food & Grocery', 'Home & Living']
data = []

for date in dates:
    weekend_boost = 1.3 if date.weekday() >= 5 else 1.0
    for product in products:
        base = {'Electronics': 85000, 'Clothing': 45000,
                'Food & Grocery': 120000, 'Home & Living': 35000}
        sales = base[product] * weekend_boost * np.random.uniform(0.8, 1.2)
        data.append({
            'date': date,
            'product': product,
            'sales': round(sales),
            'units': round(sales / (base[product] / 100))
        })

df = pd.DataFrame(data)
df['month'] = df['date'].dt.strftime('%b %Y')
df['week'] = df['date'].dt.isocalendar().week

# ── 2. KPI Calculations ────────────────────────────────────────────────────
total_revenue   = df['sales'].sum()
avg_daily       = df.groupby('date')['sales'].sum().mean()
best_product    = df.groupby('product')['sales'].sum().idxmax()
growth          = ((df[df['date'] >= dates[90]]['sales'].sum() /
                    df[df['date'] <  dates[90]]['sales'].sum()) - 1) * 100

# ── 3. Forecasting (next 30 days) ─────────────────────────────────────────
daily = df.groupby('date')['sales'].sum().reset_index()
daily['day_num'] = range(len(daily))

X = daily[['day_num']]
y = daily['sales']
model = LinearRegression().fit(X, y)

future_days = [{'day_num': len(daily) + i} for i in range(30)]
future_df   = pd.DataFrame(future_days)
forecast    = model.predict(future_df)
future_dates = [dates[-1] + timedelta(days=i+1) for i in range(30)]

# ── 4. Dashboard Plot ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Retail Sales Dashboard — 6 Month Analysis',
             fontsize=20, fontweight='bold', y=0.98)

colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

# -- KPI Cards (top row) --
kpi_labels = ['Total Revenue', 'Avg Daily Sales',
               'Top Product',  'Growth (90 days)']
kpi_values = [f"Rs. {total_revenue/1e6:.1f}M",
               f"Rs. {avg_daily/1000:.0f}K",
               best_product,
               f"+{growth:.1f}%"]
kpi_colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

for i, (label, value, color) in enumerate(
        zip(kpi_labels, kpi_values, kpi_colors)):
    ax = fig.add_axes([0.05 + i*0.235, 0.82, 0.21, 0.12])
    ax.set_facecolor(color)
    ax.text(0.5, 0.65, value, transform=ax.transAxes,
            ha='center', va='center', fontsize=14,
            fontweight='bold', color='white')
    ax.text(0.5, 0.25, label, transform=ax.transAxes,
            ha='center', va='center', fontsize=9, color='white')
    ax.set_xticks([]); ax.set_yticks([])

# -- Daily Revenue Trend --
ax1 = fig.add_subplot(3, 3, 4)
daily_total = df.groupby('date')['sales'].sum()
ax1.plot(daily_total.index, daily_total.values / 1000,
         color='#3498db', linewidth=1.5, alpha=0.8)
ax1.fill_between(daily_total.index, daily_total.values / 1000,
                  alpha=0.2, color='#3498db')
ax1.set_title('Daily Revenue Trend (Rs. 000s)', fontweight='bold')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.set_ylabel('Revenue (Rs. 000s)')
ax1.grid(True, alpha=0.3)

# -- Product Sales Breakdown --
ax2 = fig.add_subplot(3, 3, 5)
product_sales = df.groupby('product')['sales'].sum() / 1e6
bars = ax2.bar(product_sales.index, product_sales.values,
               color=colors, edgecolor='white', linewidth=0.5)
ax2.set_title('Revenue by Product (Rs. M)', fontweight='bold')
ax2.set_ylabel('Revenue (Rs. Million)')
for bar, val in zip(bars, product_sales.values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             f'Rs.{val:.1f}M', ha='center', fontsize=8)
plt.setp(ax2.get_xticklabels(), rotation=15, ha='right', fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# -- Market Share Pie --
ax3 = fig.add_subplot(3, 3, 6)
ax3.pie(product_sales.values, labels=product_sales.index,
        colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 8})
ax3.set_title('Market Share by Product', fontweight='bold')

# -- 30-Day Forecast --
ax4 = fig.add_subplot(3, 1, 3)
ax4.plot(daily['date'], daily['sales'] / 1000,
         color='#3498db', linewidth=1.5, label='Actual Sales')
ax4.plot(future_dates, forecast / 1000,
         color='#e74c3c', linewidth=2,
         linestyle='--', label='30-Day Forecast')
ax4.fill_between(future_dates,
                  (forecast * 0.9) / 1000,
                  (forecast * 1.1) / 1000,
                  alpha=0.2, color='#e74c3c',
                  label='Confidence Interval')
ax4.axvline(x=dates[-1], color='gray',
            linestyle=':', alpha=0.7, label='Forecast Start')
ax4.set_title('Sales Forecast — Next 30 Days', fontweight='bold')
ax4.set_ylabel('Daily Revenue (Rs. 000s)')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout(rect=[0, 0, 1, 0.82])
plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("Dashboard saved as dashboard.png")
print(f"\nKEY INSIGHTS:")
print(f"  Total Revenue   : Rs. {total_revenue/1e6:.2f}M")
print(f"  Avg Daily Sales : Rs. {avg_daily/1000:.1f}K")
print(f"  Top Product     : {best_product}")
print(f"  90-day Growth   : +{growth:.1f}%")
