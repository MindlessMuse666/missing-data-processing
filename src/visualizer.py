import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self, df):
        self.df = df


    def plot_missing_values_heatmap(self):
        plt.figure(figsize=(12, 8), num='Тепловая карта пропущенных значений')

        # Транспонируем матрицу пропусков
        missing_data = self.df.isnull().transpose()
        
        sns.heatmap(missing_data, cbar=False, cmap='Reds', linewidths=.5)  
        
        plt.title('Тепловая карта пропущенных значений', fontsize=16, y=1.02)
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_interpolation_comparison(self, initial_data, interpolated_data, column_name):
        plt.figure(figsize=(10, 6), num=f'Интерполяция {column_name}')
        plt.plot(initial_data.index, initial_data.values, label='Исходные данные', marker='o', linestyle='-', markersize=4)
        plt.plot(interpolated_data.index, interpolated_data.values, label='Интерполированные данные', marker='x', linestyle='--', markersize=4)
        plt.xlabel('Индекс')
        plt.ylabel(column_name)
        plt.title('Сравнение данных до и после интерполяции', fontsize=14, y=1.02)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()