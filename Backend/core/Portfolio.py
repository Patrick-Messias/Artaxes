# Holds >1 models, doesn't define Assets, Server uniquely to Manage Positions between multiple models has to dominate over all MMM and MMA

from dataclasses import dataclass, field
from typing import Optional
from PortfolioSystemManager import PortfolioSystemManager
from PortfolioMoneyManager import PortfolioMoneyManager
from Storage import Storage
from Asset import Asset
import polars as pl, uuid

@dataclass
class PortfolioParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    portfolio_data: dict=None
    portfolio_parameters: dict=None 
    portfolio_money_manager: Optional['PortfolioMoneyManager'] = None
    portfolio_system_manager: Optional['PortfolioSystemManager'] = None

    datetime_timeline: set=field(default_factory=set)
    portfolio_returns: dict=None

    date_start: Optional[str] = None
    date_end: Optional[str] = None
    data_storage_base_path: str="Backend/results"
    use_portfolio_asset_data: bool=True
    global_datetime_prefix: str="%Y-%m-%d %H:%M:%S"

class Portfolio(): 
    def __init__(self, portfolio_params: PortfolioParams):
        self.name = portfolio_params.name
        self.portfolio_data = portfolio_params.portfolio_data
        self.portfolio_parameters = portfolio_params.portfolio_parameters

        self.portfolio_money_manager = portfolio_params.portfolio_money_manager
        self.portfolio_system_manager = portfolio_params.portfolio_system_manager

        self.datetime_timeline = portfolio_params.datetime_timeline
        self.portfolio_returns = portfolio_params.portfolio_returns

        self.date_start = portfolio_params.date_start
        self.date_end = portfolio_params.date_end
        self.data_storage_base_path = portfolio_params.data_storage_base_path
        self.use_portfolio_asset_data = portfolio_params.use_portfolio_asset_data
        self.global_datetime_prefix = portfolio_params.global_datetime_prefix
    
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
            unique_dts.update(self._get_all_op_asset_datetimes())

        else: # From Portfolio Operation results data
            for *_, a_name, a_obj in self._iter_portfolio_data():        
                if "pnl_matrix" in a_obj:
                    unique_dts.update(a_obj["pnl_matrix"]["ts"])
                elif "wf" in a_obj:
                    unique_dts.update(a_obj["wf"]["datetime"])
                else: 
                    print(f"< [Error] No PnL or Walkforward data found for Asset: {a_name}")

        # From System Manager Indicators
        if self.portfolio_system_manager and self.portfolio_system_manager.sm_indicators:
            unique_dts.update(self._get_all_sm_ind_datetimes())

        # From Money Manager Indicators
        if self.portfolio_money_manager and self.portfolio_money_manager.mm_indicators:
            unique_dts.update(self._get_all_mm_ind_datetimes())

        self.datetime_timeline = sorted(list(unique_dts))

        print("> Datetime sample")
        for dt in self.datetime_timeline[:5] + self.datetime_timeline[-5:]:
            print(dt.strftime(self.global_datetime_prefix))
        return True
    
    def _get_all_op_asset_datetimes(self, data_source="local"):
        assets = Asset.load_all() # NOTE Deletar futuramente
        storage = Storage(base_path=self.data_storage_base_path)
        unique_assets_dts = set()

        for op_name, _, m_name, _, _, _, a_name, _ in self._iter_portfolio_data():
            meta = storage.load_operation_meta(op_name)
            model_meta = meta.get("models", {}).get(m_name, {})

            tf = model_meta.get("execution_timeframe", meta.get("operation_timeframe", None))
            if tf is None:
                print(f"< [Error] No timeframe found for Operation: {op_name}, Model: {m_name}. Skipping Asset: {a_name}")
                continue
            date_start = model_meta.get("date_start")
            date_end = model_meta.get("date_end")

            asset_obj = assets.get(a_name)
            asset_df = asset_obj.load(tf, data_source, date_start, date_end)

            #Adds to unique
            unique_assets_dts.update(asset_df["datetime"])

        return unique_assets_dts

    # PEGAR NOS INDICADORES PORQUE SM_ASSETS SÓ SERVE PRA INDICADORES E TEM TF DEFINIDO JÁ TMB
    def _get_all_sm_ind_datetimes(self, data_source="local"):
        assets = Asset.load_all() # NOTE Deletar futuramente
        unique_ind_dts = set()

        sm_inds = self.portfolio_system_manager.sm_indicators if (self.portfolio_system_manager and self.portfolio_system_manager.sm_indicators) else {}
        if sm_inds:
            for ind_name, ind_obj in sm_inds.items():
                tf = ind_obj.timeframe
                if tf is None:
                    print(f"< [Error] No timeframe found for System Manager Indicator: {ind_name}. Skipping.")
                    continue

                asset_obj = assets.get(ind_obj.asset)
                asset_df = asset_obj.load(tf, data_source, self.date_start, self.date_end)
                unique_ind_dts.update(asset_df["datetime"])

        return unique_ind_dts
    
    def _get_all_mm_ind_datetimes(self, data_source="local"):
        assets = Asset.load_all() # NOTE Deletar futuramente
        unique_ind_dts = set()

        mm_inds = self.portfolio_money_manager.mm_indicators if (self.portfolio_money_manager and self.portfolio_money_manager.mm_indicators) else {}
        if mm_inds:
            for ind_name, ind_obj in mm_inds.items():
                tf = ind_obj.timeframe
                if tf is None:
                    print(f"< [Error] No timeframe found for Money Manager Indicator: {ind_name}. Skipping.")
                    continue

                asset_obj = assets.get(ind_obj.asset)
                asset_df = asset_obj.load(tf, data_source, self.date_start, self.date_end)
                unique_ind_dts.update(asset_df["datetime"])

        return unique_ind_dts


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

    def _portfolio_walkforward(self):
        # Iterates over previous results and identifies each combination for OS while running
        return True

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

    portfolio_global_parameters = {
        "capital": 100000.0,
    }
    
    portfolio = Portfolio(PortfolioParams("Portfolio_Test", 
                                          portfolio_data, 
                                          portfolio_global_parameters,
                                          ))
    portfolio._run()



    """
    # XXX -> Resolver problema minutos wf
    # -> IMPORTANTE -> AO INVÉS DE PEGAR DATETIMES DOS WF/PNL, USAR DOS ASSETS DOS MODELOS, JÁ QUE PROVAVELMENTE VAI USAR NO SYSTEM/MONEY
    # -> Como vai ficar a estrutura de System/Money? ter uma lista de indicators e assets que vão ser usados?

    # money_manager_equilizer = {"frequency": 0.5 trades per day, "avg_win", "avg_loss"}    
    
    
    """
































