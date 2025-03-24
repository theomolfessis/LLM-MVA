import glob
import pandas as pd
import json

class DataLoaderECHR:
    def __init__(self, train_path:str, test_path:str, val_path:str):
        json_files_train = glob.glob(train_path+'/*.json')
        json_files_dev = glob.glob(val_path + '/*.json')
        json_files_test = glob.glob(test_path+'/*.json')
        
        df_train = self._concatenate_files_in_df(json_files_train)
        df_test = self._concatenate_files_in_df(json_files_test)
        df_val = self._concatenate_files_in_df(json_files_dev)
        
        self.df_train = self._add_binary_violation(df_train)
        self.df_test = self._add_binary_violation(df_test)
        self.df_val = self._add_binary_violation(df_val)
        self.df_train_dev = pd.concat([df_train,df_val], ignore_index=True)

    def load_data(self):
    
        return {'train':self.df_train, 'test': self.df_test, 'dev': self.df_val, 'train_dev': self.df_train_dev}


    def _concatenate_files_in_df(self, json_files)->pd.DataFrame:
        rows = []

        for file in json_files:
            with open(file, 'r') as f:
                data  = json.load(f)
                # If needed, process certain fields (e.g., join lists) here:
                # data['TEXT'] = ' '.join(data['TEXT'])
                rows.append(data)

        # Create the global DataFrame
        df = pd.DataFrame(rows)
        return df
    
    def _add_binary_violation(self, df):
        columns_to_check = [
            'VIOLATED_ARTICLES', 'VIOLATED_PARAGRAPHS', 'VIOLATED_BULLETPOINTS'
        ]

        # Create the new binary column
        df['VIOLATED'] = df[columns_to_check].apply(
            lambda row: 1 if any(len(item) > 0 for item in row) else 0,
            axis=1
        )
        
        return df


