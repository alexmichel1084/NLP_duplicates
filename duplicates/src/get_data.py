from sklearn.model_selection import train_test_split


class PrepareData:
    def __init__(self, train):
        self.name1 = train['name_1'].values
        self.name2 = train['name_2'].values
        self.labels = train['is_duplicate'].values

    def get_data(self):
        return train_test_split(self.name1, self.name2, self.labels, test_size=0.2, random_state=42)
