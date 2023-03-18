from sklearn.impute import  SimpleImputer  
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor (BaseEstimator, TransformerMixin):
    #train our custom preprocessor
    def fit(self, X, y = None):
        
        #create a fit imputer
        self.imputer = SimpleImputer()
        self.imputer.fit(X[['HouseAge', 'DistanceToStation', 'NumberOfPubs']])
        
        #create and fit Standard Scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X[['HouseAge', 'DistanceToStation', 'NumberOfPubs']])
        
        #Create and fit OneHotEncoder
        self.onehot = OneHotEncoder(handle_unknown = 'ignore')
        self.onehot.fit(X[['PostCode']])
        
        return self
        
        
        
    # Apply our custom Preprocessor
    def transform(self,X):
             
        #Apply simple imputer
        imputed_cols =  self.imputer.transform(X[['HouseAge', 'DistanceToStation', 'NumberOfPubs']])
        onehot_cols = self.onehot.transform(X[['PostCode']])
        
        
        # Copy the df
        transformed_df = X.copy()
        
        #Apply year and month transforms
        
        transformed_df['Year'] = transformed_df['TransactionDate'].apply(lambda x: x[:4]).astype(int)
        transformed_df['Month'] = transformed_df['TransactionDate'].apply(lambda x: x[5:]).astype(int)
        transformed_df = transformed_df.drop('TransactionDate', axis = 1)
                                                                         
        
        
        
        # Apply transformed columns
        
        transformed_df[['HouseAge', 'DistanceToStation', 'NumberOfPubs']] = imputed_cols
        transformed_df[['HouseAge', 'DistanceToStation', 'NumberOfPubs']] = self.scaler.transform(transformed_df[['HouseAge', 'DistanceToStation', 'NumberOfPubs']])
        
        #Drop Exisiting Post coloumn and replace with one hot equiv
        transformed_df = transformed_df.drop('PostCode', axis = 1)
        transformed_df[self.onehot.get_feature_names_out()] = onehot_cols.toarray().astype(int)
        
        return transformed_df