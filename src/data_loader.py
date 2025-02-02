import pandas as pd


class DataLoader:
    def __init__(self, url):
        self.url = url


    def load_data(self):
        try:
            df = pd.read_csv(self.url)
            return df
        except Exception as e:
            print(f'Ошибка загрузки данных: {e}')
            return None
