from data_loader import DataLoader
from missing_data_handler import MissingDataHandler
from feature_engineer import FeatureEngineer
from visualizer import Visualizer
from utils import print_results
import warnings
warnings.filterwarnings('ignore')


DATA_URL = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'

def main():
    # 1. Загрузка данных
    data_loader = DataLoader(DATA_URL)
    df = data_loader.load_data()

    if df is None:
        print('Не удалось загрузить данные. Завершение работы.')
        return

    # 2. Изучение пропущенных значений
    missing_handler = MissingDataHandler(df)
    missing_values = missing_handler.calculate_missing_values()
    print_results('Количество пропущенных значений в каждом признаке:', missing_values)

    # 3. Визуализация пропусков
    visualizer = Visualizer(df)
    visualizer.plot_missing_values_heatmap()
    
    # 4. Удаление пропущенных значений
    dropped_df = missing_handler.drop_rows_with_missing_values()
    print_results('Размер датасета после удаления пропусков', dropped_df.shape)
    
    # Возвращаем исходный df
    missing_handler = MissingDataHandler(df)

    # 5. Заполнение пропущенных значений (возраст) средним
    filled_df_mean = missing_handler.fill_missing_with_mean_median('age', method='mean')
    print_results('Датасет после заполнения пропусков в age средним:', filled_df_mean.head())
    
    # Возвращаем исходный df
    missing_handler = MissingDataHandler(df)

    # 6. Заполнение пропущенных значений (возраст) медианным
    filled_df_median = missing_handler.fill_missing_with_mean_median('age', method='median')
    print_results('Датасет после заполнения пропусков в age медианным:', filled_df_median.head())

    # Возвращаем исходный df
    missing_handler = MissingDataHandler(df)
    
    # 7. Интерполяция (fare)
    initial_fare, interpolated_fare = missing_handler.interpolate_missing_values('fare')
    visualizer.plot_interpolation_comparison(initial_fare, interpolated_fare, 'fare')
    print_results('Датасет после интерполяции fare:', missing_handler.df.head())
    
    # 8. KNN Imputation (age и fare)
    columns_for_knn = ['age', 'fare']
    knn_df = missing_handler.knn_imputation(columns_for_knn)
    print_results('Датасет после KNN Imputation:', knn_df.head())

    # 9. Создание дополнительных признаков
    feature_engineer = FeatureEngineer(df)
    df_with_missing_indicator = feature_engineer.create_missing_indicator('age')
    print_results('Датасет с новыми признаками:', df_with_missing_indicator.head())

    #10. Предсказание пропущенных значений в 'age'
    missing_handler = MissingDataHandler(df)
    predictors = ['pclass','sex','sibsp','parch','fare','embarked'] #Важно выбрать подходящие предикторы
    predicted_df = missing_handler.predict_missing_values_with_linear_regression('age', predictors) #Изменено
    print_results('Датасет после предсказания пропусков:', predicted_df.head())


if __name__ == '__main__':
    main()