#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:04:27 2024

@author: dozeduck
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import mimetypes
import streamlit as st
from tempfile import NamedTemporaryFile

st.set_page_config(layout="wide")

class SynergyPlotter:
    def __init__(self, file_path, response_label):
        self.df = self.read_excel(file_path)
        self.drug1_name, self.drug2_name, self.unit_label = self.extract_labels(self.df)
        self.response_label = response_label
        self.plot_heatmap()
        self.plot_dose_response_curves()
      
    def read_excel(self, file_path):
        """从Excel文件中读取数据"""
        df = pd.read_excel(file_path)
        st.text(df)
        return df
      
    def extract_labels(self, df):
        """提取药物名称和单位标签"""
        drug1_name = df["Drug1"].iloc[0]
        drug2_name = df["Drug2"].iloc[0]
        unit_label = df["ConcUnit"].iloc[0]
        return drug1_name, drug2_name, unit_label
      
    def plot_heatmap(self):
        """绘制热图并保存为文件"""
        pivot_table = self.df.pivot(index="Conc1", columns="Conc2", values="Response")
        colors = [(0, "green"), (0.5, "white"), (1, "red")]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, cmap=cmap, center=0, annot=True, fmt=".2f", cbar_kws={'label': f'{self.response_label} (%)'})
        plt.title(f'Dose-response matrix ({self.response_label})')
        plt.xlabel(f'{self.drug2_name} Conc. ({self.unit_label})')
        plt.ylabel(f'{self.drug1_name} Conc. ({self.unit_label})')
        plt.gca().invert_yaxis()
        plt.savefig("/tmp/response-matrix.png", dpi=600, bbox_inches='tight')
        st.image("/tmp/response-matrix.png")  # 展示热图
        self.add_download_button('response-matrix.png', "/tmp/response-matrix.png")
      
    def add_download_button(self, download_name, file_path):
        """创建文件下载按钮"""
        with open(file_path, "rb") as file:
            file_content = file.read()
        
        mime_type, _ = mimetypes.guess_type(file_path)
        
        st.download_button(
            label=f"Download {download_name}",
            data=file_content,
            file_name=download_name,
            mime=mime_type
        )

    def plot_dose_response_curves(self):
        """绘制剂量-响应曲线并保存为文件"""
        def logistic(x, a, b, c, d):
            return a / (1.0 + np.exp(-c * (x - d))) + b
        
        def exponential(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        def polynomial_2(x, a, b, c):
            return a * x**2 + b * x + c
        
        def safe_curve_fit(model, x, y):
            try:
                popt, _ = curve_fit(model, x, y, maxfev=10000)
                y_pred = model(x, *popt)
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    return None, -np.inf
                return popt, r2_score(y, y_pred)
            except (RuntimeError, OverflowError, ValueError):
                return None, -np.inf
        
        def calc_ic50(popt, model_name):
            y_target = 50
            if model_name == 'Logistic':
                a, b, c, d = popt
                return d + np.log(a / (y_target - b) - 1) / -c
            elif model_name == 'Polynomial_2':
                a, b, c = popt
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
        
        def plot_curve(df, conc_column, response_column, drug_label):
            x_data = df[conc_column]
            y_data = df[response_column]
        
            max_response = y_data.max()
            if max_response < 50:
                best_model_name = 'Exponential'
                best_model = exponential
                best_popt, _ = safe_curve_fit(exponential, x_data, y_data)
            else:
                models = {
                    'Logistic': (logistic, 4),
                    'Exponential': (exponential, 3),
                    'Polynomial_2': (polynomial_2, 3)
                }
        
                best_model = None
                best_r2 = -np.inf
                best_popt = None
                best_model_name = None
        
                for name, (model, _) in models.items():
                    popt, r2 = safe_curve_fit(model, x_data, y_data)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model
                        best_popt = popt
                        best_model_name = name
        
            if best_model_name:
                ic50_value = calc_ic50(best_popt, best_model_name)
            else:
                ic50_value = None
        
            plt.figure(figsize=(10, 6))
            plt.scatter(x_data, y_data, color='red', label='Data')
            x_fit = np.linspace(min(x_data), max(x_data), 100)
            y_fit = best_model(x_fit, *best_popt)
            plt.plot(x_fit, y_fit, color='black', label=f'Best fit ({best_model_name})')
            plt.xscale('log')
            plt.title(f'Dose-response curve for {drug_label}')
            plt.xlabel(f'Concentration ({self.unit_label})')
            plt.ylabel(f'{self.response_label} (%)')
            y_max = max(100, y_data.max())
            y_min = min(0, y_data.min())
            plt.ylim(y_min, y_max)
        
            if ic50_value:
                plt.text(0.05, 0.95, f'IC50: {ic50_value:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
            else:
                plt.text(0.05, 0.95, f'Best R²: {best_r2:.2f}', transform=plt.gca().transAxes, verticalalignment='top')

            scatter_path = f"/tmp/dose-response-scatter-{drug_label}.png"
            plt.savefig(scatter_path, dpi=600, bbox_inches='tight')
            st.image(scatter_path)  # 展示散点图
            
            # plt.savefig(f"/tmp/dose-response-scatter-{drug_label}.png", dpi=600, bbox_inches='tight')
            self.add_download_button(f"dose-response-scatter-{drug_label}.png", f"/tmp/dose-response-scatter-{drug_label}.png")
        
        df_drug1 = self.df[self.df['Conc2'] == 0]
        # plot_curve(df_drug1, 'Conc1', 'Response', df_drug1['Drug1'].iloc[0])
        df_drug2 = self.df[self.df['Conc1'] == 0]
        # plot_curve(df_drug2, 'Conc2', 'Response', df_drug2['Drug2'].iloc[0])

        col1, col2 = st.columns(2)
        plot_curve(df_drug1, 'Conc1', 'Response', df_drug1['Drug1'].iloc[0], col1)
        plot_curve(df_drug2, 'Conc2', 'Response', df_drug2['Drug2'].iloc[0], col2)

# Streamlit 主程序
def main():
    st.title("Welcome to synergy plotter v1.1")

    plot = st.columns(1)

    def save_uploaded_file(uploaded_file):
        try:
            with NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                return temp_file.name
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return None

    with plot[0]:
        st.header("Plot figures")
        multi_files = st.file_uploader("Upload files", accept_multiple_files=True, type=['xlsx'])
        uploaded_filenames = [uploaded_file.name for uploaded_file in multi_files]
        tmp_path = [save_uploaded_file(multi_files[i]) for i in range(len(multi_files))]
        response_label = st.selectbox("Type of response", ['Inhibition', 'Activation'])

        if st.button('Plotting') and multi_files:
            for path in tmp_path:
                SynergyPlotter(path, response_label)

if __name__ == "__main__":
    main()
