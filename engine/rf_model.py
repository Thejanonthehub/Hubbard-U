from sklearn.ensemble import RandomForestRegressor

class RFModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
