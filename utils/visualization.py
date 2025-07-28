import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DataVisualizer:
    @staticmethod
    def plot_price_and_volume(df, figsize=(14, 10)):
        """Plot Bitcoin price and volume over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Price plot
        ax1.plot(df.index, df['close'], label='Closing Price', color='tab:blue')
        ax1.set_title('Bitcoin Closing Price Over Time')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend()
        ax1.grid(True)
        
        # Volume plot
        ax2.plot(df.index, df['volume'], label='Volume Traded', color='tab:red')
        ax2.set_title('Bitcoin Trading Volume Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_outliers(df):
        """Plot boxplots for outlier detection"""
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        n_cols = 3
        n_rows = len(numeric_columns) // n_cols + (1 if len(numeric_columns) % n_cols else 0)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, column in enumerate(numeric_columns):
            sns.boxplot(y=df[column], ax=axes[i], color='skyblue')
            axes[i].set_title(f'Boxplot of {column}', fontsize=12)
            axes[i].set_xlabel(column, fontsize=10)
        
        # Remove unused subplots
        for j in range(len(numeric_columns), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_predictions(y_true, y_pred, title="Predictions vs Actual", sample_size=200):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(12, 6))
        
        # Take a sample for better visualization
        indices = np.random.choice(len(y_true), min(sample_size, len(y_true)), replace=False)
        indices = np.sort(indices)
        
        plt.plot(indices, y_true[indices], label='Actual', alpha=0.7)
        plt.plot(indices, y_pred[indices], label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True)
        plt.show()