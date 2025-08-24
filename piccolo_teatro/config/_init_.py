from pydantic import BaseModel, confloat
from typing import List, Dict, Callable, Literal
import pandas as pd
from skopt.space import Real, Integer

class XGBConfig(BaseModel):
    parameters: dict = {
        "eta": 0.3,               # learning rate
        "max_depth": 6,           # maximum tree depth
        "min_child_weight": 1,    # minimum sum Hessian in a leaf
        "gamma": 0,               # minimum loss reduction for a split
        "subsample": 1,           # row subsampling ratio
        "colsample_bytree": 1,    # feature subsampling ratio per tree
        "lambda": 1,              # L2 regularization term
        "alpha": 0,               # L1 regularization term
    }

    param_space: dict = {
        'n_estimators': Integer(50, 500),
        'max_depth': Integer(3, 12),
        'learning_rate': Real(1e-3, 1e-1, prior='log-uniform'),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'gamma': Real(0, 5),
        'reg_alpha': Real(1e-8, 1.0, prior='log-uniform'),
        'reg_lambda': Real(1e-8, 1.0, prior='log-uniform')
    }
    
    ic_dim: confloat(ge=0.0, le=1.0) = 0.9


    file_name: str = "xgb_trend_perc"

class Feature(BaseModel):
    columns: List[str]
    const: bool
    enabled: bool
    update: Callable = None

class ProblemConfig(BaseModel):
    periods: List[int] = [2, 4, 6, 8, 10, 15, 20, 30]
    target: Literal["percentage_bought", "percentage_bought_delta", "percentage_bought_log1p"]

problem_config = ProblemConfig(target = "percentage_bought")
class TimeSeriesEngine:
    def __init__(self, df=None):
        self.df = df
        self.new_prediction = None
        self.periods = problem_config.periods
        self.target = problem_config.target

        self.encoding_dict = {
            "show_type": ['Internazionale', 'Ospitalità', 'Collaborazione', 'Produzione', 'Festival'],
            #    "performance_day": ["lun", "mar", "mer", "gio", "ven", "sab", "dom"],
        }


        # Initialization of constant features
        #self.performance_day: Feature = Feature(columns=self.encoding_dict["performance_day"],
        #                                            const=True,
        #                                            enabled=False,
        #                                            update=self.__update_const)
        #self.performance_hour: Feature = Feature(columns=["performance_hour"],
        #                                                    const=True,
        #                                                    enabled=False,
        #                                                    update=self.__update_const)
        #self.performance_number: Feature = Feature(columns=["performance_number"],
        #                                                    const=True,
        #                                                    enabled=False,
        #                                                    update=self.__update_const)

        self.show_type: Feature = Feature(columns=self.encoding_dict["show_type"],
                                                        const=True,
                                                        enabled=False,
                                                        update=self.__update_const)
        
        self.show_capacity: Feature = Feature(columns=["show_capacity"],
                                                            const=True,
                                                            enabled=False,
                                                            update=self.__update_const)
        
        self.num_performances: Feature = Feature(columns=["num_performances"],
                                                            const=True,
                                                            enabled=False,
                                                            update=self.__update_const)
        
        self.sales_duration: Feature = Feature(columns=["sales_duration"],
                                                            const=True,
                                                            enabled=True,
                                                            update=self.__update_const)
        

        # Variables features
        self.start_sales_distance: Feature = Feature(columns=["start_sales_distance"],
                                                            const=False,
                                                            enabled=True,
                                                            update=self.__update_start_sales_distance)
        
        self.end_sales_distance: Feature = Feature(columns=["end_sales_distance"],
                                                            const=False,
                                                            enabled=True,
                                                            update=self.__update_end_sales_distance)
        
        self.end_season_distance: Feature = Feature(columns=["end_season_distance"],
                                                            const=False,
                                                            enabled=False)

        self.percentage_sales_day: Feature = Feature(columns=["percentage_sales_day"],
                                                     const=False,
                                                     enabled=True,
                                                     update=self.__update_percentage_sales_day)
    
        self.remaining_tickets: Feature = Feature(columns=["remaining_tickets"],
                                                        const=False,
                                                        enabled=False)
        
        # self.tickets_cum_sum: Feature = Feature(columns=["tickets_cum_sum"],
        #                                             const=False,
        #                                             enabled=False)
        # 
        # self.tickets: Feature = Feature(columns=["tickets"],
        #                                         const=False,
        #                                         enabled=False)
        
        self.percentage_bought = Feature(columns=["percentage_bought"],
                                                const=False,
                                                enabled=self.target=="percentage_bought",
                                                update=self.__update_percentage_bought)
        
        # Calling functions that generate multiple features (es. moving avg of multiple periods)
        self.__init_percentage_bought_avg(enabled=self.target=="percentage_bought")

        self.__init_percentage_bought_delta(enabled=self.target=="percentage_bought")

        self.__init_percentage_bought_shifted(enabled=self.target=="percentage_bought")

        self.__get_features()
        
    def update_features(self, prediction):
        if self.df is None:
            raise Exception("Please add a input df to Feature class before calling update_features")
        self.new_prediction = prediction
        self.new_row = {}

        for feature in self.variable_features:
            update_function = feature.update
            update_function()

        for feature in self.const_features:
            update_function = feature.update
            update_function(feature.columns[0])

        self.new_row = pd.DataFrame(self.new_row)
        self.df = pd.concat([self.df, self.new_row], ignore_index=True)

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
        return [col_name+"_"+str(period) for period in self.periods]

    # Init methods 
    def __init_percentage_bought_avg(self, enabled):
        columns = self.__create_period_names("percentage_bought_avg")
        self.percentage_bought_avg = Feature(columns=columns,
                                    const=False,
                                    enabled=enabled,
                                    update=self.__update_percentage_bought_avg
                                    )

    def __init_percentage_bought_delta(self, enabled):
        columns = self.__create_period_names("percentage_bought_delta")
        self.percentage_bought_avg = Feature(columns=columns,
                                    const=False,
                                    enabled=enabled,
                                    update=self.__update_percentage_bought_delta
                                    )
        
    def __init_percentage_bought_shifted(self, enabled):
        columns = self.__create_period_names("percentage_bought_shifted")
        self.percentage_bought_shifted = Feature(columns=columns,
                                    const=False,
                                    enabled=enabled,
                                    update=self.__update_percentage_bought_shifted)

    # Update methods
    def __update_const(self, col_name):
        const_val = self.df[col_name].iloc[-1]
        self.new_row[col_name] = [const_val]

    def __update_percentage_bought(self):
        self.new_row["percentage_bought"] = [self.new_prediction]

    def __update_percentage_bought_avg(self):
        # concateno storico + nuova predizione
        values = self.df["percentage_bought"].tolist() + [self.new_prediction]
        s = pd.Series(values)
        for period in self.periods:
            # rolling mean e prendo solo l'ultimo
            avg_val = s.rolling(period).mean().iloc[-1]
            self.new_row[f"percentage_bought_avg_{period}"] = avg_val

    def __update_percentage_bought_delta(self):
        values = self.df["percentage_bought"].tolist() + [self.new_prediction]
        s = pd.Series(values)
        for period in self.periods:
            # differenza tra t e t-period
            delta_val = s.diff(periods=period).iloc[-1]
            self.new_row[f"percentage_bought_delta_{period}"] = delta_val

    def __update_percentage_bought_shifted(self):
        values = self.df["percentage_bought"].tolist() + [self.new_prediction]
        s = pd.Series(values)
        first_val = s.iloc[0]
        for period in self.periods:
            # shift e fill dei NaN con il primo valore
            shifted_val = s.shift(period).fillna(first_val).iloc[-1]
            self.new_row[f"percentage_bought_shifted_{period}"] = shifted_val

    def __update_start_sales_distance(self):
        last_row = self.df.iloc[[-1]]
        new_value = float(last_row["start_sales_distance"].iloc[0]) + 1

        self.new_row["start_sales_distance"] = [new_value]

    def __update_end_sales_distance(self):
        last_row = self.df.iloc[[-1]]
        new_value = int(last_row["end_sales_distance"].iloc[0]) -1
        self.new_row["end_sales_distance"] = [new_value]
    
    def __update_percentage_sales_day(self):
        last_row = self.df.iloc[[-1]]
        increased_day = int(last_row["start_sales_distance"].iloc[0]) + 1
        new_value = increased_day/self.df["sales_duration"].iloc[0]
        self.new_row["percentage_sales_day"] = [new_value]
       

    def config_recap(self):
        const_feat = self.const_features
        variable_feat = self.variable_features

        print("Training Config\n")
        print("Target:", self.target)
        print("Constant Features:")
        for feat in const_feat:
            if len(feat.columns)==1:
                print(f" {feat.columns[0]}")
            else:
                print(f" {feat.columns[0]}, ... , {feat.columns[-1]}")


        print("Variable Features:")
        for feat in variable_feat:
            if len(feat.columns)==1:
                print(f" {feat.columns[0]}")
            else:
                print(f" {feat.columns[0]}, ... , {feat.columns[-1]}")

     
        print("MA e Lag periods:", self.periods)
        print("")

