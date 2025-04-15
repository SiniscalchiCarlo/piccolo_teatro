from pydantic import BaseModel
from typing import List, Dict, Callable
from utils import add_moving_avarages, add_shifted_values
import pandas as pd

class TrainConfig(BaseModel):
    periods: List[int] = [2, 4, 6, 8, 10, 15, 20, 30]
    prediction_period: int = 1
    target: str = "percentage_bought"

class FeatEngConf(BaseModel):
    encoding_dict: Dict[str, List[str]] = {
        "performance_type": ['Internazionale', 'Ospitalità', 'Collaborazione', 'Produzione', 'Festival'],
    #    "performance_day": ["lun", "mar", "mer", "gio", "ven", "sab", "dom"],
    }

    targets_dict: Dict[str, List[int]] = {
        "gain_cum_sum": [1],
        "tickets_cum_sum": [1],
        "gain": [1],
        "tickets": [1],
        "percentage_bought": [1]
        }

class Feature(BaseModel):
    columns: List[str]
    const: bool
    enabled: bool
    update: Callable = None

class Features:
    def __init__(self):
        self.train_config = TrainConfig()
        self.feat_eng_conf = FeatEngConf()
        self.input_df = None
        self.prediction = None

        # Initialization of constant features
        self.performance_type: Feature = Feature(columns=self.feat_eng_conf.encoding_dict["performance_type"],
                                                        const=True,
                                                        enabled=False,
                                                        update=self.__update_const)
        
        #self.performance_day: Feature = Feature(columns=self.feat_eng_conf.encoding_dict["performance_day"],
        #                                            const=True,
        #                                            enabled=False,
        #                                            update=self.__update_const)
        
        self.performance_capacity: Feature = Feature(columns=["performance_capacity"],
                                                            const=True,
                                                            enabled=False,
                                                            update=self.__update_const)
        
        #self.performance_hour: Feature = Feature(columns=["performance_hour"],
        #                                                    const=True,
        #                                                    enabled=False,
        #                                                    update=self.__update_const)
        
        self.num_performances: Feature = Feature(columns=["num_performances"],
                                                            const=True,
                                                            enabled=False,
                                                            update=self.__update_const)
        
        #self.performance_number: Feature = Feature(columns=["performance_number"],
        #                                                    const=True,
        #                                                    enabled=False,
        #                                                    update=self.__update_const)
        
        self.sales_duration: Feature = Feature(columns=["sales_duration"],
                                                            const=True,
                                                            enabled=False,
                                                            update=self.__update_const)
        

        # Variables features
        self.start_sales_distance: Feature = Feature(columns=["start_sales_distance"],
                                                            const=False,
                                                            enabled=False,
                                                            update=self.__update_start_sales_distance)
        
        self.end_sales_distance: Feature = Feature(columns=["end_sales_distance"],
                                                            const=False,
                                                            enabled=False,
                                                            update=self.__update_end_sales_distance)
        
        self.end_season_distance: Feature = Feature(columns=["end_season_distance"],
                                                            const=False,
                                                            enabled=False)
        self.percentage_sales_day: Feature = Feature(columns=["percentage_sales_day"],
                                                     cont=False,
                                                     enabled=True)
    
        self.remaining_tickets: Feature = Feature(columns=["remaining_tickets"],
                                                        const=False,
                                                        enabled=False)
        
        self.tickets_cum_sum: Feature = Feature(columns=["tickets_cum_sum"],
                                                    const=False,
                                                    enabled=False)
        
        self.tickets: Feature = Feature(columns=["tickets"],
                                                const=False,
                                                enabled=False)
        
        self.percentage_bought = Feature(columns=["percentage_bought"],
                                                const=False,
                                                enabled=True,
                                                update=self.__update_percentage_bought)
        
        # Calling functions that generate multiple features (es. moving avg of multiple periods)
        self.__get_percentage_bought_avg(enabled=True),
        self.__get_percentage_bought_shifted(enabled=True)

        self.__get_features()

        
    def update_features(self, input_df, prediction):
        self.input_df = input_df
        self.prediction = prediction
        self.updated_df = pd.DataFrame()

        for feature in self.variable_features:
            update_function = feature.update
            update_function()

        for feature in self.const_features:
            update_function = feature.update
            update_function(feature.columns[0])

    def __get_features(self):
        """
        Get a list of all the features, constant features, variable features
        """
        self.enabled_features = []
        self.const_features = []
        self.variable_features = []
        for name, value in vars(self).items():
            if isinstance(value, Feature):
                self.enabled_features.append(value)
                if value.enabled:
                    if value.const:
                        self.const_features.append(value)
                    else:
                        self.variable_features.append(value)

    def __create_period_names(self, col_name: str) -> List[str]:
        return [col_name+"_"+str(period) for period in self.train_config.periods]
        columns = self.__create_period_names("tickets_shifted")
        self.tickets_shifted = Feature(columns=columns,
                                    const=False,
                                    enabled=enabled)
      
    def __get_percentage_bought_avg(self, enabled) -> List[str]:
        columns = self.__create_period_names("percentage_bought_avg")
        self.percentage_bought_avg = Feature(columns=columns,
                                    const=False,
                                    enabled=enabled,
                                    update=self.__update_percentage_bought_avg
                                    )
        
    def __get_percentage_bought_shifted(self, enabled) -> List[str]:
        columns = self.__create_period_names("percentage_bought_shifted")
        self.percentage_bought_shifted = Feature(columns=columns,
                                    const=False,
                                    enabled=enabled,
                                    update=self.__update_percentage_bought_shifted)

    def __update_percentage_bought(self):
        targets_and_predictions = self.input_df["percentage_bought"].tolist() + [self.prediction]
        self.updated_df["percentage_bought"] = targets_and_predictions

    def __update_percentage_bought_avg(self):
        self.updated_df = add_moving_avarages(self.updated_df, ["percentage_bought"], self.train_config.periods)

    def __update_percentage_bought_shifted(self):
        self.updated_df = add_shifted_values(self.updated_df, ["percentage_bought"], self.train_config.periods)

    def __update_start_sales_distance(self):
        last_row = self.input_df.iloc[[-1]]
        new_value = float(last_row["start_sales_distance"].iloc[0]) + 1
        values = self.input_df["start_sales_distance"].tolist()
        values.append(new_value)
        self.updated_df["start_sales_distance"] = values

    def __update_end_sales_distance(self):
        last_row = self.input_df.iloc[[-1]]
        new_value = float(last_row["end_sales_distance"].iloc[0]) -1
        values = self.input_df["end_sales_distance"].tolist()
        values.append(new_value)
        self.updated_df["end_sales_distance"] = values

    def __update_const(self, col_name):
        const_val = self.input_df[col_name].iloc[-1]
        self.updated_df["end_sales_distance"] = const_val
        