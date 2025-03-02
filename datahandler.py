import pandas as pd
import numpy as np
import os
from visualizer import Visualizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)

class DataHandler:
    """
    
    """

    def __init__(self, train_data_path: str = 'data/Train.csv', test_data_path: str = 'data/Test.csv'):
        
        if(not os.path.exists(train_data_path)):
            raise FileNotFoundError(f"File not found: {train_data_path}")
        if(not os.path.exists(test_data_path)):
            raise FileNotFoundError(f"File not found: {test_data_path}")
        
        self.train_df = pd.read_csv(train_data_path)
        self.test_df = pd.read_csv(test_data_path)
        self.__calculate_missing_values__()
        self.encoder = OneHotEncoder(sparse_output=False)
        self.target_encoder = LabelEncoder()
        

    def get_train_data(self):
        return self.train_df
    
    def get_test_data(self):
        return self.test_df
    
    def print_testdata_with_missing_values(self):
        print(f"Columns with Missing values: \n{self.test_missing_data}\n\n")
    
    def print_traindata_with_missing_values(self):
        print(f"Columns with Missing values: \n{self.train_missing_data}\n\n") 
    
    def inspect_train_data(self):
        self.__inspect_data__(self.train_df)

    def inspect_test_data(self):
        self.__inspect_data__(self.test_df)
    
    def encode_data(self):
        self.train_df = self.__encode_categorical_features__(self.train_df)
        self.test_df = self.__encode_categorical_features__(self.test_df)
        self.__calculate_missing_values__()

    def clean_data(self):
        self.train_df.drop(['ID'], axis=1, inplace=True)
        self.test_df.drop(['ID'], axis=1, inplace=True)
        self.__fill_missing_values__(self.train_df)
        self.__fill_missing_values__(self.test_df)
        self.train_df.drop_duplicates(inplace=True)
        self.test_df.drop_duplicates(inplace=True)

        self.__calculate_missing_values__()
    
    def balance_data(self):
        
        max_count = self.train_df['Segmentation'].value_counts().min()
        new_train_df = pd.DataFrame()
        for segment in self.train_df['Segmentation'].unique():
            segment_df = self.train_df[self.train_df['Segmentation'] == segment]
            new_train_df = pd.concat([new_train_df, segment_df.sample(n=max_count, random_state=42)])
        self.train_df = new_train_df
        self.__calculate_missing_values__()


    def __inspect_data__(self, data, name: str = "Dataset"):
        print(f"Inspecting {name} Data: \n")
        print(f"Shape: {data.shape}\n\n")
        print(f"Top 5 Rows: \n{data.head()}\n\n")
        print(f"Data Info: \n{data.info()}\n\n")
        print(f"Data Description: \n{data.describe()}\n\n")
        for column in data.columns:
            print(f"{column} Value Counts: \n{data[column].value_counts()}\n\n")
        print("\n\n")
        #v = Visualizer(data)
        #v.plot_histograms()

    def __calculate_missing_values__(self):
        test_missing_values = pd.DataFrame(self.test_df.isnull().sum())
        test_missing_values.reset_index(inplace=True)
        test_missing_values.columns = ['Feature', 'Missing_Values']
        test_missing_values = test_missing_values[test_missing_values['Missing_Values'] > 0]
        self.test_missing_data = test_missing_values

        train_missing_values = pd.DataFrame(self.train_df.isnull().sum())
        train_missing_values.reset_index(inplace=True)
        train_missing_values.columns = ['Feature', 'Missing_Values']
        train_missing_values = train_missing_values[train_missing_values['Missing_Values'] > 0]
        self.train_missing_data = train_missing_values

    def __fill_missing_values__(self, df: pd.DataFrame):
        df.fillna({'Ever_Married': df['Ever_Married'].mode()[0]
                   ,'Graduated': df['Graduated'].mode()[0]
                   ,'Profession': df['Profession'].mode()[0]
                   ,'Var_1': df['Var_1'].mode()[0]
                   # Histograms show that the data is skewed left, so we will use median
                   ,'Work_Experience': df['Work_Experience'].median()
                   ,'Family_Size': df['Family_Size'].median()
                   }, inplace=True)
        
        self.__calculate_missing_values__()
        
        
    def __encode_categorical_features__(self, df: pd.DataFrame):
        
        # Map Age to Age Groups
        df['Age'] = df['Age'].apply(lambda x: '0-20' if x <= 20 else '21-40' if x <= 40 else '41-60' if x <= 60 else '60+')
        df['Family_Size'] = df['Family_Size'].apply(lambda x: 'Small' if x <= 2 else 'Medium' if x <= 4 else 'Large')
        df['Work_Experience'] = df['Work_Experience'].apply(lambda x: 'Entry' if x <= 2 else 'Mid' if x <= 8 else 'Senior')     
        print(df['Segmentation'].value_counts())   
        #df['Segmentation'] =  self.target_encoder.fit_transform(df['Segmentation']) # df['Segmentation'].apply(lambda x: 1 if x == 'A' else 2 if x == 'B' else 3 if x == 'C' else 4)
        # One Hot Encode Categorical Features
        categorical_attributes = df.select_dtypes(include=['object']).columns.tolist()
        if 'Segmentation' in categorical_attributes:
            categorical_attributes.remove('Segmentation')

        
        #encoded = self.encoder.fit_transform(df[categorical_attributes])
        #encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(categorical_attributes))                
        for attribute in categorical_attributes:
            df[attribute] = self.target_encoder.fit_transform(df[attribute])
        #encoded_df['Segmentation'] = df['Segmentation']
        #encoded_df.dropna(inplace=True)
        return df#encoded_df
        
        
        #print(df)
        
        #df.dropna(inplace=True)

        #print(encoded_df)
        
        #df.dropna(inplace=True)


        #for attribute in categorical_attributes:
        #    print(f"Encoding {attribute}")
        #    df[attribute] = self.target_encoder.fit_transform(df[attribute])
        #df.drop_duplicates(inplace=True)
        #
        ##remove target column
        #categorical_attributes.remove('Segmentation')
#
        #encoded = self.encoder.fit_transform(df[categorical_attributes])
        #encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(categorical_attributes))        
        #df.drop(categorical_attributes, axis=1, inplace=True)
        #df['Segmentation'] = self.target_encoder.fit_transform(df['Segmentation'])
        #df = pd.concat([df, encoded_df], axis=1)
        #df.dropna(inplace=True)    

        



if __name__ == "__main__":
    dh = DataHandler()
    #dh.inspect_train_data()
    #dh.inspect_test_data()
    #dh.print_traindata_with_missing_values()
    #dh.print_testdata_with_missing_values()
    dh.clean_data()
    dh.balance_data()
    dh.encode_data()
    
    print(dh.get_train_data())
    print(dh.get_test_data())
    dh.print_testdata_with_missing_values()
    dh.print_traindata_with_missing_values()
    print(dh.get_test_data()['Segmentation'].value_counts())
    print(dh.get_train_data().isnull().sum())
    print(dh.get_test_data().isnull().sum())
    #dh.get_test_data()
    #dh.get_train_data()
    #print(f"Train Data: \n{dh.inspect_train_data()}\n\n")
    #print(f"Test Data: \n{dh.inspect_test_data()}\n\n")