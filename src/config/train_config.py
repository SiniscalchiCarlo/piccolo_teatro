from pydantic import BaseModel
from typing import List

encoding_dict = {
    "performance_type": ['Internazionale', 'Ospitalità', 'Collaborazione', 'Produzione', 'Festival'],
    "performance_day": ["lun", "mar", "mer", "gio", "ven", "sab", "dom"],
}


class DfColumns(BaseModel):
    columns: List[str]
    const: bool
    enabled: bool


class TrainConfig:

    def __init__(self):
        self.periods: List[int] = [2, 4, 6, 8, 10, 15, 20, 30]
        self.prediction_period: int = 1
        self.target: str = "percentage_bought"

        # Contanst features
        self.performance_type: DfColumns = DfColumns(columns=['internazionale', 'ospitalità', 'collaborazione', 'produzione', 'festival'],
                                                     const=True,
                                                     enabled=False)
        self.performance_day: DfColumns = DfColumns(columns=["lun", "mar", "mer", "gio", "ven", "sab", "dom"],
                                                    const=True,
                                                    enabled=False)
        
        self.performance_capacity: DfColumns = DfColumns(columns=["performance_capacity"],
                                                         const=True,
                                                         enabled=False)
        
        self.performance_hour: DfColumns = DfColumns(columns=["performance_hour"],
                                                         const=True,
                                                         enabled=True)
        
        self.num_performances: DfColumns = DfColumns(columns=["num_performances"],
                                                         const=True,
                                                         enabled=True)
        
        self.performance_number: DfColumns = DfColumns(columns=["performance_number"],
                                                         const=True,
                                                         enabled=True)
        
        self.sales_duration: DfColumns = DfColumns(columns=["sales_duration"],
                                                         const=True,
                                                         enabled=True)
        

        # Variables features
        self.start_sales_distance: DfColumns = DfColumns(columns=["start_sales_distance"],
                                                         const=False,
                                                         enabled=True)
        
        self.end_sales_distance: DfColumns = DfColumns(columns=["end_sales_distance"],
                                                         const=False,
                                                         enabled=True)
        self.end_season_distance: DfColumns = DfColumns(columns=["end_season_distance"],
                                                         const=False,
                                                         enabled=False)
        
        #self.avg_ticket_price: DfColumns = DfColumns(columns=["avg_ticket_price"],
        #                                             const=False,
        #                                             enabled=False)
        self.remaining_tickets: DfColumns = DfColumns(columns=["remaining_tickets"],
                                                      const=False,
                                                      enabled=False)
        
        self.tickets_cum_sum: DfColumns = DfColumns(columns=["tickets_cum_sum"],
                                                    const=False,
                                                    enabled=False)
        
        #self.gain_cum_sum: DfColumns = DfColumns(columns=["gain_cum_sum"],
        #                                        const=False,
        #                                        enabled=False)
        
        self.tickets: DfColumns = DfColumns(columns=["tickets"],
                                                const=False,
                                                enabled=False)
        
        self.percentage_bought = DfColumns(columns=["percentage_bought"],
                                                const=False,
                                                enabled=True)
        
        #self.gain: DfColumns = DfColumns(columns=["gain"],
        #                                        const=False,
        #                                        enabled=False)
        
        #self.gain_cum_sum(enabled=False)
        #self.gain_cum_sum_avg(enabled=False)
        #self.gain_cum_sum_shifted(enabled=False)
#
        #self.gain_avg(enabled=False)
        #self.gain_shifted(enabled=False)

        self.get_tickets_cum_sum_avg(enabled=False),
        self.get_tickets_cum_sum_shifted(enabled=False)

        self.get_tickets_avg(enabled=False),
        self.get_tickets_shifted(enabled=False)


        self.get_percentage_bought_avg(enabled=True),
        self.get_percentage_bought_shifted(enabled=True)

        self.features: List[DfColumns] = [self.performance_type, self.performance_day,
                                          self.performance_capacity, 
                                          self.start_sales_distance, self.end_sales_distance,self.end_season_distance,
                                          #self.avg_ticket_price,
                                          #self.gain_cum_sum_avg, self.gain_cum_sum_shifted,
                                          self.remaining_tickets, 

                                          self.tickets, self.tickets_cum_sum, 
                                          self.tickets_cum_sum_avg, self.tickets_cum_sum_shifted,

                                          self.percentage_bought, self.percentage_bought_avg, self.percentage_bought_shifted
                                          ]

            
    def create_period_names(self, col_name: str) -> List[str]:
        return [col_name+"_"+str(period) for period in self.periods]

    def get_tickets_cum_sum_avg(self, enabled) -> List[str]:
        columns = self.create_period_names("tickets_cum_sum_avg")
        self.tickets_cum_sum_avg = DfColumns(columns=columns,
                                             const=False,
                                             enabled=enabled)


    def get_tickets_cum_sum_shifted(self, enabled) -> List[str]:
        columns = self.create_period_names("tickets_cum_sum_shifted")
        self.tickets_cum_sum_shifted = DfColumns(columns=columns,
                                                 const=False,
                                                 enabled=enabled)
    
    def get_tickets_avg(self, enabled) -> List[str]:
        columns = self.create_period_names("tickets_avg")
        self.tickets_avg = DfColumns(columns=columns,
                                    const=False,
                                    enabled=enabled)
        
    def get_tickets_shifted(self, enabled) -> List[str]:
        columns = self.create_period_names("tickets_shifted")
        self.tickets_shifted = DfColumns(columns=columns,
                                    const=False,
                                    enabled=enabled)
    
    
    def get_percentage_bought_avg(self, enabled) -> List[str]:
        columns = self.create_period_names("percentage_bought_avg")
        self.percentage_bought_avg = DfColumns(columns=columns,
                                    const=False,
                                    enabled=enabled)
        
    def get_percentage_bought_shifted(self, enabled) -> List[str]:
        columns = self.create_period_names("percentage_bought_shifted")
        self.percentage_bought_shifted = DfColumns(columns=columns,
                                    const=False,
                                    enabled=enabled)
        



train_config = TrainConfig()
