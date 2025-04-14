


#self.avg_ticket_price: DfColumns = DfColumns(columns=["avg_ticket_price"],
#                                             const=False,
#                                             enabled=False)
#self.gain_cum_sum: DfColumns = DfColumns(columns=["gain_cum_sum"],
#                                        const=False,
#                                        enabled=False)
#self.gain: DfColumns = DfColumns(columns=["gain"],
#                                        const=False,
#                                        enabled=False)
#self.gain_cum_sum(enabled=False)
#self.gain_cum_sum_avg(enabled=False)
#self.gain_cum_sum_shifted(enabled=False)
#self.gain_avg(enabled=False)
#self.gain_shifted(enabled=False)

def __get_tickets_cum_sum_avg(self, enabled) -> List[str]:
        columns = self.__create_period_names("tickets_cum_sum_avg")
        self.tickets_cum_sum_avg = DfColumns(columns=columns,
                                             const=False,
                                             enabled=enabled)

def __get_tickets_cum_sum_shifted(self, enabled) -> List[str]:
    columns = self.__create_period_names("tickets_cum_sum_shifted")
    self.tickets_cum_sum_shifted = DfColumns(columns=columns,
                                                const=False,
                                                enabled=enabled)

def __get_tickets_avg(self, enabled) -> List[str]:
    columns = self.__create_period_names("tickets_avg")
    self.tickets_avg = DfColumns(columns=columns,
                                const=False,
                                enabled=enabled)
    
def __get_tickets_shifted(self, enabled) -> List[str]: