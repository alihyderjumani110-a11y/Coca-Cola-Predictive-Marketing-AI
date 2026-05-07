import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. LOAD TIME SERIES DATA (Quarterly Revenue 2020-2025)
quarters = pd.date_range(start='2020-01-01', periods=24, freq='Q')
sales = [8.25, 9.50, 10.10, 8.16, 9.66, 10.90, 11.20, 9.90,
         10.75, 12.00, 12.50, 10.75, 11.43, 12.80, 13.00, 11.52,
         11.76, 13.20, 13.40, 11.70, 11.98, 13.50, 13.80, 11.96]

df = pd.Series(sales, index=quarters, name='Revenue_Billion')

# 2. FIT ARIMA(2,1,1) MODEL
# d=1 (one differencing to make data stationary)
# p=2 (use 2 past values)
# q=1 (use 1 past forecast error)
model = ARIMA(df, order=(2,1,1))
fitted_model = model.fit()

# 3. FORECAST NEXT 4 QUARTERS (2026)
forecast_result = fitted_model.forecast(steps=4)
forecast_index = pd.date_range(start='2026-01-01', periods=4, freq='Q')

print("=== ARIMA(2,1,1) FORECAST RESULTS ===")
print(f"Model Summary:\n{fitted_model.summary()}")
print("\n--- Future Forecast (2026) ---")
for i, val in enumerate(forecast_result):
    print(f"{forecast_index[i].strftime('%Y Q%q')}: ${val:.2f} Billion")

# Validate against your report Section 8.3
q1_2026 = forecast_result[0]
print(f"\n-> Q1 2026 Forecast: ${q1_2026:.2f}B (Report says $11.83B)")

# 4. PLOT (Matches your Figure in Section 8.4)
plt.figure(figsize=(10,4))
plt.plot(df.index, df, label='Historical Revenue', marker='o')
plt.plot(forecast_index, forecast_result, label='ARIMA Forecast', marker='o', linestyle='--', color='red')
plt.title('Coca-Cola Quarterly Revenue Forecast (ARIMA)')
plt.ylabel('Revenue (Billion USD)')
plt.legend()
plt.grid(True)
plt.savefig('arima_forecast_plot.png') # Save it to include in GitHub
plt.show()
