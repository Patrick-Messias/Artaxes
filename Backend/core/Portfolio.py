# Holds >1 models, doesn't define Assets, Server uniquely to Manage Positions between multiple models has to dominate over all MMM and MMA

from dataclasses import dataclass, field
from typing import Optional
from PortfolioSystemManager import PortfolioSystemManager
from PortfolioMoneyManager import PortfolioMoneyManager
from Storage import Storage
import polars as pl, uuid

"""
portfolio_data = {
    "operation_test": {
        "MA Trend Following": {
            "AT15": {
                "EURUSD"
            }
        }
    }
}
"""

@dataclass
class PortfolioParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    portfolio_data: dict=None
    portfolio_money_manager: Optional['PortfolioMoneyManager'] = None
    portfolio_system_manager: Optional['PortfolioSystemManager'] = None

    data_storage_base_path: str="Backend/results"
    datetime_timeline: set=field(default_factory=set)
    portfolio_returns: dict=None

class Portfolio(): 
    def __init__(self, portfolio_params: PortfolioParams):
        self.name = portfolio_params.name
        self.portfolio_data = portfolio_params.portfolio_data
        self.portfolio_money_manager = portfolio_params.portfolio_money_manager
        self.portfolio_system_manager = portfolio_params.portfolio_system_manager

        self.data_storage_base_path = portfolio_params.data_storage_base_path
        self.datetime_timeline = portfolio_params.datetime_timeline
        self.portfolio_returns = portfolio_params.portfolio_returns

    def _get_all_portfolio_data(self) -> dict:
        return self.portfolio_data if self.portfolio_data else {}
    
    def _run(self):
        # Data Init - Uses already uploaded data or loads from drive with Storage.py
        print("> Populating Portfolio Data from Database")
        self._load_saved_data()

        # Maps all unique datetimes to use as simulator timeline
        #print("> Mapping all unique datetimes from Parset, Money and System Manager")
        #self._map_all_unique_datetimes()




        # Runs Portfolio Simulation
        #print("> Executing Portfolio Simulation")
        #self._simulation()
            


        return True








    def _simulation(self):

        self.portfolio_returns = {}
        return True








    def _map_all_unique_datetimes(self):
        unique_dts = set()

        # From Portfolio Data
        for *_, a_name, a_obj in self._iter_portfolio_data(): 
            if "pnl" in a_obj and isinstance(a_obj["pnl"], dict):
                unique_dts.update(a_obj["pnl"]["datetime"])
            elif "wf" in a_obj and isinstance(a_obj["wf"], dict):
                unique_dts.update(a_obj["wf"]["datetime"])
            else: 
                print(f"< [Error] No PnL or Walkforward data found for Asset: {a_name}")

        # From System/Money Manager Assets

        self.datetime_timeline = sorted(list(unique_dts))
        return True

    def _load_saved_data(self):
        storage = Storage(base_path=self.data_storage_base_path)

        for op_name, _, m_name, _, s_name, _, a_name, a_obj in self._iter_portfolio_data():

            Abandonando JSON, ficar apenas em parquet
            Walkforward, pnl_matrix, lot_matrix e cada wf_{is}_{os}_{st} individual
            serão salvos em
            wfm/
            matrix/
            trades/

            assets_trade_matrix = storage.load(op_name, m_name, s_name, a_name)
            a_obj.update(assets_trade_matrix)
            for parset in assets_trade_matrix:
                print(parset)
        return True

    def _iter_portfolio_data(self):
        for op_name, op_obj in self.portfolio_data.items():
            for m_name, m_obj in op_obj.items():
                for s_name, s_obj in m_obj.items():
                    for a_name, a_obj in s_obj.items():
                        yield op_name, op_obj, m_name, m_obj, s_name, s_obj, a_name, a_obj



if __name__ == "__main__":

    portfolio_data = {
        "operation_test": {
            "MA Trend Following": {
                "AT15": {
                    "EURUSD": {}
                }
            }
        }
    }
    
    portfolio = Portfolio(PortfolioParams(name="Portfolio_Test", portfolio_data=portfolio_data))
    portfolio._run()


































