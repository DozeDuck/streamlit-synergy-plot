#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:04:27 2024

@author: b8048283
"""
import getopt
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
# for scatter plot's non-linear regression
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser(description='Version: 1.6  \n'
                                 'Usage: read excel file and plot synergy plots\n' \
                                  './vc-5.py -f sample-data-synergy-finder-VC.xlsx' ,
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-f', '--files', required=True, help='excel file, e.g: sample-data-synergy-finder-VC.xlsx')   
parser.add_argument('-l', '--label', required=False, default='Inhibition', help='type of response, e.g: Inhibition or Activation')   
args = parser.parse_args()
# define the parameters
file_path = str(args.files)
response_label = str(args.label)


# 从Excel文件中读取数据
df = pd.read_excel(file_path)

# 获取药物名称
drug1 = df["Drug1"].iloc[0]
drug2 = df["Drug2"].iloc[0]
unit_label = df["ConcUnit"].iloc[0]

# 创建数据透视表
pivot_table = df.pivot("Conc1", "Conc2", "Response")

# 定义颜色映射
colors = [(0, "green"), (0.5, "white"), (1, "red")]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# 绘制heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, cmap=cmap, center=0, annot=True, fmt=".2f", cbar_kws={'label': f'{response_label} (%)'})
# plt.title(f'Heatmap of {response_label} by Concentrations of {drug2} and {drug1}')
plt.title(f'Dose-response matrix ({response_label})')
plt.xlabel(f'{drug2} Conc. ({unit_label})')
plt.ylabel(f'{drug1} Conc. ({unit_label})')
plt.gca().invert_yaxis()  # 反转y轴使其从0开始
plt.show()
############################################################################################################################################

# 定义非线性回归函数
def logistic(x, a, b, c, d):
    return a / (1.0 + np.exp(-c * (x - d))) + b

# 定义指数回归函数
def exponential(x, a, b, c):
#     return a * np.exp(b * x) + c
    return a * np.exp(-b * x) + c

# 定义x^2多项式回归回归函数
def polynomial_2(x, a, b, c):
    return a * x**2 + b * x + c

# 定义x^3多项式回归回归函数
def polynomial_3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# 安全地计算拟合
def safe_curve_fit(model, x, y):
    try:
        popt, _ = curve_fit(model, x, y, maxfev=10000)
        y_pred = model(x, *popt)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return None, -np.inf
        return popt, r2_score(y, y_pred)
    except (RuntimeError, OverflowError, ValueError):
        return None, -np.inf

# 计算IC50或y=50时的x值
def calc_ic50(popt, model_name):
    y_target = 50
    if model_name == 'Logistic':
        a, b, c, d = popt
        return d + np.log(a / (y_target - b) - 1) / -c
    elif model_name == 'Polynomial_2':
        a, b, c = popt
        # 求解二次方程 ax^2 + bx + (c - y_target) = 0
        c -= y_target
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        x1 = (-b + np.sqrt(discriminant)) / (2*a)
        x2 = (-b - np.sqrt(discriminant)) / (2*a)
        return x1 if x1 > 0 else x2
    elif model_name == 'Exponential':
        a, b, c = popt
        if 50 - c <= 0:
            return None
        return -np.log((y_target - c) / a) / b

# 函数来绘制剂量-响应曲线并计算IC50
def plot_dose_response_curve(df, conc_column, response_column, drug_label, response_label, unit_label):
    x_data = df[conc_column]
    y_data = df[response_column]

    # 检查抑制力最大值是否小于50
    max_response = y_data.max()
    if max_response < 50:
        # 仅使用指数模型
        best_model_name = 'Exponential'
        best_model = exponential
        best_popt, r2 = safe_curve_fit(exponential, x_data, y_data)
        print(best_popt)
        # # 拟合不同的模型
        # models = {
        #     'Exponential': (exponential, 3)
        # }

        # best_model = None
        # best_r2 = -np.inf
        # best_popt = None
        # best_model_name = None

        # for name, (model, param_count) in models.items():
        #     popt, r2 = safe_curve_fit(model, x_data, y_data)
        #     if r2 > best_r2:
        #         best_r2 = r2
        #         best_model = model
        #         best_popt = popt
        #         best_model_name = name
        
    else:
        # 拟合不同的模型
        models = {
            'Logistic': (logistic, 4),
            'Exponential': (exponential, 3),
            'Polynomial_2': (polynomial_2, 3)
        }

        best_model = None
        best_r2 = -np.inf
        best_popt = None
        best_model_name = None

        for name, (model, param_count) in models.items():
            popt, r2 = safe_curve_fit(model, x_data, y_data)
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_popt = popt
                best_model_name = name

    if best_model_name == 'Logistic' or best_model_name == 'Exponential' or best_model_name == 'Polynomial_2':
        ic50_value = calc_ic50(best_popt, best_model_name)
    else:
        ic50_value = None

    # 绘制散点图和最佳拟合曲线
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='red', label='Data')
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = best_model(x_fit, *best_popt)
    plt.plot(x_fit, y_fit, color='black', label=f'Best fit ({best_model.__name__})')
    plt.xscale('log')
    plt.title(f'Dose-response curve for {drug_label}')
    plt.xlabel(f'Concentration ({unit_label})')
    plt.ylabel(f'{response_label} (%)')
    # 设置y轴范围
    y_max = max(100, y_data.max())
    y_min = min(0, y_data.min())
    
    plt.ylim(y_min, y_max)

    # 在图中打印出参数信息和IC50（如果适用）
    if ic50_value:
        plt.text(0.05, 0.95, f'IC50: {ic50_value:.2f}', 
                 transform=plt.gca().transAxes, verticalalignment='top')
    else:
        plt.text(0.05, 0.95, f'Best R²: {best_r2:.2f}', 
                 transform=plt.gca().transAxes, verticalalignment='top')

    plt.show()
    
# 为Drug1绘制剂量-响应曲线
df_drug1 = df[df['Conc2'] == 0]
plot_dose_response_curve(df_drug1, 'Conc1', 'Response', df_drug1['Drug1'].iloc[0], response_label, unit_label)

# 为Drug2绘制剂量-响应曲线
df_drug2 = df[df['Conc1'] == 0]
plot_dose_response_curve(df_drug2, 'Conc2', 'Response', df_drug2['Drug2'].iloc[0], response_label, unit_label)
