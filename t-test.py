import scipy.stats as stats

rmse_cslstm = [0.7115, 1.5052, 2.4360, 3.5530, 4.9117]  # CS-LSTM(M)
rmse_weather3 = [0.6954, 1.4869, 2.4022, 3.4955, 4.8337]  # CS-LSTM-Weather(M)(3)
rmse_weather5 = [0.6904, 1.4672, 2.3686, 3.4654, 4.7684]  # CS-LSTM-Weather(M)(5)

t_stat_1_2, p_value_1_2 = stats.ttest_rel(rmse_cslstm, rmse_weather3)
t_stat_2_3, p_value_2_3 = stats.ttest_rel(rmse_weather3, rmse_weather5)
t_stat_1_3, p_value_1_3 = stats.ttest_rel(rmse_cslstm, rmse_weather5)

print("Comparison 1: CS-LSTM(M) vs CS-LSTM-Weather(M)(3)")
print(f"t-statistic: {t_stat_1_2:.4f}")
print(f"p-value: {p_value_1_2:.4f}")
if p_value_1_2 < 0.05:
    print("The difference is statistically significant.\n")
else:
    print("The difference is not statistically significant.\n")

print("Comparison 2: CS-LSTM-Weather(M)(3) vs CS-LSTM-Weather(M)(5)")
print(f"t-statistic: {t_stat_2_3:.4f}")
print(f"p-value: {p_value_2_3:.4f}")
if p_value_2_3 < 0.05:
    print("The difference is statistically significant.\n")
else:
    print("The difference is not statistically significant.\n")

print("Comparison 3: CS-LSTM(M) vs CS-LSTM-Weather(M)(5)")
print(f"t-statistic: {t_stat_1_3:.4f}")
print(f"p-value: {p_value_1_3:.4f}")
if p_value_1_3 < 0.05:
    print("The difference is statistically significant.\n")
else:
    print("The difference is not statistically significant.\n")
