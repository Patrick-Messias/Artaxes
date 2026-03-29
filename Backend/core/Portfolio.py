# Holds >1 models, doesn't define Assets, Server uniquely to Manage Positions between multiple models has to dominate over all MMM and MMA

from dataclasses import dataclass, field
from typing import Optional
from PortfolioSystemManager import PortfolioSystemManager
from PortfolioMoneyManager import PortfolioMoneyManager
from Storage import Storage
from Asset import Asset
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

    use_portfolio_asset_data: bool=True

class Portfolio(): 
    def __init__(self, portfolio_params: PortfolioParams):
        self.name = portfolio_params.name
        self.portfolio_data = portfolio_params.portfolio_data
        self.portfolio_money_manager = portfolio_params.portfolio_money_manager
        self.portfolio_system_manager = portfolio_params.portfolio_system_manager

        self.data_storage_base_path = portfolio_params.data_storage_base_path
        self.datetime_timeline = portfolio_params.datetime_timeline
        self.portfolio_returns = portfolio_params.portfolio_returns

        self.use_portfolio_asset_data = portfolio_params.use_portfolio_asset_data
    
    def _run(self):
        # Data Init - Uses already uploaded data or loads from drive with Storage.py
        print("     > Populating Portfolio Data from Database")
        self._load_selected_saved_returns_data()

        # Maps all unique datetimes to use as simulator timeline
        print("     > Mapping all unique datetimes from Portfolio Data")
        self._map_all_unique_datetimes()

        # Runs Portfolio Simulation
        print("     > Executing Portfolio Simulation")
        self._simulation()
            
        return True

    def _simulation(self):

        self.portfolio_returns = {}
        return True








    def _map_all_unique_datetimes(self):
        unique_dts = set()

        if self.use_portfolio_asset_data: # From Portfolio Assets
            unique_dts.update(self._load_all_op_asset_datetimes())

        else: # From Portfolio Operation results data
            for *_, a_name, a_obj in self._iter_portfolio_data():        
                if "pnl_matrix" in a_obj:
                    unique_dts.update(a_obj["pnl_matrix"]["ts"])
                elif "wf" in a_obj:
                    unique_dts.update(a_obj["wf"]["datetime"])
                else: 
                    print(f"< [Error] No PnL or Walkforward data found for Asset: {a_name}")

        # From System/Money Manager Assets



        self.datetime_timeline = sorted(list(unique_dts))

        print("> Datetime sample")
        for dt in self.datetime_timeline[:5] + self.datetime_timeline[-5:]:
            print(dt.strftime("%d-%m-%y %H:%M:%S"))
        return True
    
    def _load_all_op_asset_datetimes(self):
        unique_assets_dts = set()

        # NOTE After updating Asset storage, update below

        for *_, a_name in self._iter_portfolio_data(): 
            # Loads assets from Operation map
            assets_class = Asset(date_start=None, date_end=None)
            asset_df = assets_class.data_get(a_name) # pl.DataFrame with column "ts"

            #Adds to unique
            unique_assets_dts.update(asset_df["datetime"])

        return unique_assets_dts

    def _load_selected_saved_returns_data(self): # Loads all data from selected map
        storage = Storage(base_path=self.data_storage_base_path)

        for op_name, _, m_name, _, s_name, _, a_name, a_obj in self._iter_portfolio_data():
            assets_trade_matrix = storage.load(op_name, m_name, s_name, a_name)
            a_obj.update(assets_trade_matrix)
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

    """
    

    # XXX -> Resolver problema minutos wf
    # -> IMPORTANTE -> AO INVÉS DE PEGAR DATETIMES DOS WF/PNL, USAR DOS ASSETS DOS MODELOS, JÁ QUE PROVAVELMENTE VAI USAR NO SYSTEM/MONEY
    # -> Como vai ficar a estrutura de System/Money? ter uma lista de indicators e assets que vão ser usados?

    # money_manager_equilizer = {"frequency": 0.5 trades per day, "avg_win", "avg_loss"}    
    
    
    """
































