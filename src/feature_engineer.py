class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()


    def create_missing_indicator(self, column):
        '''Создание нового признака, указывающего на наличие пропущенных значений'''
        self.df[f'missing_{column}'] = self.df[column].isnull().astype(int)
        return self.df