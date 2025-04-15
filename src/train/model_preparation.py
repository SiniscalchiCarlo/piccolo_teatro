import pandas as pd
from pydantic import BaseModel
from typing import Any, Literal
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from typing import List

from config import Features, TrainConfig
import pickle


class ModelData(BaseModel):
    
    df_path: str
    features_config: Features = Features()
    train_config: TrainConfig =TrainConfig()
    df: pd.DataFrame = None
    X: pd.DataFrame = None
    Y: pd.DataFrame = None

    class Config:
        arbitrary_types_allowed = True

    def load_df(self):
        self.df = pd.read_parquet(self.df_path)
        
        #if offset is not None:
        #    self.df =self.df.head(int(len(self.df)*offset))



    def keep_enabled_columns(self):
        """
        Keeps only the enabled features and the target column
        """
        # Removing not needed features
        features = self.features_config.enabled_features
        target_col = f"TARGET_{self.train_config.target}_{self.train_config.prediction_period}"
        columns_to_keep = ["date", target_col]
        for feature in features:
            if feature.enabled:
                columns_to_keep += feature.columns

        self.df = self.df[columns_to_keep]

    def separete_features_targets(self, sort=False, shuffle=False):
        """
        splits the df into features and target
        sort: if true sorts the df, if false shuffles it
        """
        if self.df is None:
            raise Exception("Please load the df first")

        if sort and shuffle:
            raise Exception("You can't both shuffle and sort, please choose one")
        
        
        self.keep_enabled_columns()

        # Convert date column to datetime"
        self.df.loc[:, "date"] = pd.to_datetime(self.df["date"], format='%d/%m/%Y')
        self.df = self.df.set_index("date")

        
        if sort:
            self.df = self.df.sort_values(by="date")
        if shuffle:
            self.df = self.df.sample(frac=1)

        target_col = f"TARGET_{self.train_config.target}_{self.train_config.prediction_period}"
        if target_col in self.df:
            self.Y = self.df[target_col]
        self.X = self.df.drop(columns=[target_col])



class TrainModel(BaseModel):
    train_data: ModelData
    validation_data: ModelData = None
    test_data_: ModelData = None
    train_prediciton: List[float] = None
    validation_prediciton: List[float] = None
    test_prediciton: List[float] = None
    model: Any = None



    def train_XGBRegressor(self,parameters):
        # Check that we are using only enabled features and that are separeted from target
        if self.train_data.X is None and self.train_data.Y is None:
            self.train_data.separete_features_targets(sort=False) #when sort=False shuffles the data

        self.model = XGBRegressor(n_estimators=parameters["n_estimators"], 
                                  learning_rate=parameters["learning_rate"], 
                                  objective=parameters["objective"])
        self.model.fit(self.train_data.X, self.train_data.Y)
    
    def save_model(self,path):
        print("FEATURES", self.model.feature_names_in_)
        pickle.dump(self.model, open(path, "wb"))

    def abs_error(self, df_type: Literal["train", "validation", "test"]):
        if df_type == "train":
            self.train_prediciton = self.model.predict(self.train_data.X)
            train_mae = mean_absolute_error(self.train_data.Y, self.train_prediciton)
            print(f"Training Mean Absolute Error: {train_mae}")

        if df_type == "validation":
            if self.validation_data!=None:
                self.validation_prediciton = self.model.predict(self.validation_data.X)
                validation_mae = mean_absolute_error(self.validation_data.Y, self.validation_prediciton)
                print(f"Validation Mean Absolute Error: {validation_mae}")
            else:
                raise Exception("Please provide validation data")

        if df_type == "test":
            if self.test_data!=None:
                self.__test_prediction = self.model.predict(self.test_data.X)
                test_mae = mean_absolute_error(self.test_data.Y, self.__test_prediction)
                print(f"Test Mean Absolute Error: {test_mae}")
            else:
                raise Exception("Please provide test data")
   