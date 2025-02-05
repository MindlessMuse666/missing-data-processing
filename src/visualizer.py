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


    def plot_comparison_of_distributions(self, original_data, imputed_data, original_label, imputed_label, title):
        '''Сравнение распределений до и после заполнения пропусков'''
        plt.figure(figsize=(12, 6), num=title)
        sns.histplot(original_data, kde=True, label=original_label)
        sns.histplot(imputed_data, kde=True, color='purple', alpha=0.6, label=imputed_label)  # Изменены цвет и прозрачность
        plt.title(title, fontsize=14, y=1.02)
        plt.xlabel('Значение')
        plt.ylabel('Плотность')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_knn_imputation_results(self, original_df, knn_df):
        '''Визуализация результатов KNN Imputation (scatter plot age vs fare до и после)'''
        plt.figure(figsize=(14, 7), num='Результаты KNN Imputation')
        
        # График до KNN Imputation
        plt.subplot(1, 2, 1)
        plt.scatter(original_df['age'], original_df['fare'], alpha=0.5)
        plt.title('Исходные данные (age vs fare)')
        plt.xlabel('Age')
        plt.ylabel('Fare')
        plt.grid(True)

        # График после KNN Imputation
        plt.subplot(1, 2, 2)
        plt.scatter(knn_df['age'], knn_df['fare'], alpha=0.5, color='green')
        plt.title('После KNN Imputation (age vs fare)')
        plt.xlabel('Age')
        plt.ylabel('Fare')
        plt.grid(True)

        plt.tight_layout()
        plt.show()