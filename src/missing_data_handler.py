import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import LabelEncoder


class MissingDataHandler:
    def __init__(self, df):
        self.df = df.copy()
        
        
    def calculate_missing_values(self):
        '''Подсчет количества пропущенных значений в каждом признаке'''
        missing_values = self.df.isnull().sum()
        
        return missing_values


    def drop_rows_with_missing_values(self):
        '''Удаление строк с пропущенными значениями'''
        initial_shape = self.df.shape
        self.df.dropna(inplace=True)
        dropped_shape = self.df.shape
        
        print(f'Размер датасета до удаления: {initial_shape}')
        print(f'Размер датасета после удаления: {dropped_shape}')
        
        return self.df

        
    def fill_missing_with_mean_median(self, column, method='mean'):
        '''Заполнение пропущенных значений средним или медианным значением'''
        initial_stats = self.df[column].describe()
        
        if method == 'mean':
            fill_value = self.df[column].mean()
        elif method == 'median':
            fill_value = self.df[column].median()
        else:
            raise ValueError('Неверный метод. Должен быть "mean" или "median"')

        self.df[column].fillna(fill_value, inplace=True)
        filled_stats = self.df[column].describe()
        
        print(f'Статистика до заполнения: \n{initial_stats}')
        print(f'Статистика после заполнения: \n{filled_stats}')
        
        return self.df

    
    def interpolate_missing_values(self, column, method='linear', num_missing=50):
        '''Заполнение пропущенных значений методом интерполяции'''
        initial_data = self.df[column].copy()
        missing_indices = np.random.choice(self.df.index, size=num_missing, replace=False)
        self.df.loc[missing_indices, column] = np.nan
        self.df[column].interpolate(method=method, inplace=True)
        interpolated_data = self.df[column].copy()
        self.df.loc[missing_indices, column] = initial_data.loc[missing_indices]
        
        return initial_data, interpolated_data

      
    def knn_imputation(self, columns, n_neighbors=5):
        '''Заполнение пропущенных значений методом KNN'''
        imputer = KNNImputer(n_neighbors=n_neighbors)
        self.df[columns] = imputer.fit_transform(self.df[columns])
        
        return self.df


    def predict_missing_values_with_linear_regression(self, column, predictors):
        '''Предсказание пропущенных значений в столбце с использованием линейной регрессии.'''
        df_copy = self.df.copy()

        # 1. Подготовка данных: разделение на обучающую и тестовую выборки
        train_df = df_copy[df_copy[column].notnull()].copy()
        test_df = df_copy[df_copy[column].isnull()].copy()

        if train_df.empty or test_df.empty:
            print(f'Недостаточно данных для предсказания пропусков в столбце {column}')
            return self.df

        # 2. Кодирование категориальных признаков и масштабирование
        for predictor in predictors:
            if train_df[predictor].dtype == 'object':
                label_encoder = LabelEncoder()
                train_df[predictor] = label_encoder.fit_transform(train_df[predictor].astype(str))
                test_df[predictor] = label_encoder.transform(test_df[predictor].astype(str))

        scaler = StandardScaler()
        X_train = train_df[predictors].copy()
        X_test = test_df[predictors].copy()

        # Определяем числовые колонки для масштабирования
        numerical_cols = X_train.select_dtypes(include=np.number).columns
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        y_train = train_df[column].copy()

        # 3. Обучение модели
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 4. Предсказание пропущенных значений
        predicted_ages = model.predict(X_test)

        # 5. Заполнение пропущенных значений в исходном DataFrame
        self.df.loc[self.df[column].isnull(), column] = predicted_ages

        print(f'Предсказаны пропущенные значения в столбце "{column}" с использованием линейной регрессии.')

        return self.df