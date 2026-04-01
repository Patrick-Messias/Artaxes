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

    date_start: Optional[str] = None
    date_end: Optional[str] = None
    data_storage_base_path: str="Backend/results"
    use_portfolio_asset_data: bool=True
    global_datetime_prefix: str="%Y-%m-%d %H:%M:%S"

    datetime_timeline: set=field(default_factory=set)

class Portfolio(): 
    def __init__(self, portfolio_params: PortfolioParams):
        self.name = portfolio_params.name
        self.portfolio_data = portfolio_params.portfolio_data
        self.portfolio_parameters = portfolio_params.portfolio_parameters

        self.portfolio_money_manager = portfolio_params.portfolio_money_manager
        self.portfolio_system_manager = portfolio_params.portfolio_system_manager

        self.date_start = portfolio_params.date_start
        self.date_end = portfolio_params.date_end
        self.data_storage_base_path = portfolio_params.data_storage_base_path
        self.use_portfolio_asset_data = portfolio_params.use_portfolio_asset_data
        self.global_datetime_prefix = portfolio_params.global_datetime_prefix

        self.datetime_timeline = portfolio_params.datetime_timeline
        self.portfolio_returns: dict={}
        self.sim_data: dict= {}
    


    def _simulation(self):

        # 1 - Init, populating sim_data
        self._populate_sim_data()
        sim_current_equity = self.portfolio_parameters.get("capital", 100000.0)

        # {(op, mod, strat, asset): {"weight": 0.1, "id": "48_48_48"}} / "id": "parset..."
        active_positions = {} 
        hierarchy = {}
        self.portfolio_returns = {}
        

        # 2 - Run Timeline
        for step_dt in self.datetime_timeline:
            instance = self.sim_data.get(step_dt, {})
            if not instance: continue

            # Init step data
            self.portfolio_returns[step_dt] = {"assets": {}}
            step_perc_total        =     0.0
            step_pnl_nominal_total =  0.0

            #||=====================================================================================||#
            
            # A - Exits at [i] open
            for idf, pos_info in active_positions.items():
                pass

            #||=====================================================================================||#
            
            # B - Entries at [i] open - MM Tactical Level (MM can change with exit/entry)
            
                # Must recalculate position sizes if the rules call for it, else use E defined
                # First-Come First-Served - Allocates 10% until 100% is hit, following hierarchy
                # Static Hierarchy - Limits to how much each level can use margin/capital

            #||=====================================================================================||#
            
            # C - Updates PnL of open positions at [i] ends in previous step
            for idf, pos_info in active_positions.items():
                # ifs       = (op, mod, strat, asset)
                # pos_info  = {"weight": 0.1, "lot": 1.0, "type": "wf", "id": "48_48_48", "meta": {"margin": ...}}}
                tid = pos_info["id"]
                wht = pos_info["weight"] # Defined by Money Manager (capital allocated)

                asset_data = instance.get(idf, {})

                # Lógic to decide where PnL comes from (wf or pnl_matrix)
                if "wf_pnls" in asset_data and tid in asset_data["wf_pnls"]:
                    inst_ret = asset_data["wf_pnls"][tid]
                else: 
                    inst_ret = asset_data.get("pnls", {}).get(tid, 0.0)
                inst_lot = asset_data.get("lots", {}).get(tid, 1.0)

                # perc
                trade_perc = inst_ret * inst_lot    # Raw trade percentage weighted with lot_size
                pos_perc_port = trade_perc * wht    # trade percentage in relation to portfolio
                step_perc_total += pos_perc_port    # perc accumulated in this datetime

                # PnL
                pos_pnl_port = sim_current_equity * pos_perc_port # $ pnl in relation to portfolio
                step_pnl_nominal_total += pos_pnl_port # pnl accumulated in this datetime

                # Strat Returns
                self.portfolio_returns[step_dt][idf] = {
                    "trade_perc": trade_perc,
                    "pos_perc_port": pos_perc_port,
                    "pos_pnl_port": pos_pnl_port,
                    "weight": wht
                }

            # Updates global
            sim_current_equity += step_pnl_nominal_total
            self.portfolio_returns[step_dt] = {
                "portfolio_perc":    step_perc_total,
                "portfolio_pnl": step_pnl_nominal_total
            }

            #||=====================================================================================||#
            
            # D - Updates System Managers - at [i] ends
            hierarchy = self._system_managers(step_dt, hierarchy)

            #||=====================================================================================||#
            
            # E - Updates Money Managers - at [i] ends - Strategic Level
            hierarchy = self._money_managers(step_dt, hierarchy)

            #||=====================================================================================||#

        print(f"> {step_dt} - Portfolio PnL: {sim_current_equity:..2f}")

        return True

    # ── Portfolio Defs ───────────────────────────────────────────────

    def _system_managers(self, dt, hierarchy):
        for o_name, o_obj in self._iter_portfolio_data():
            operation_key = (o_name)
            operation_data = self.sim_data[:dt][operation_key]

            if self.portfolio_system_manager:
                hierarchy = self.portfolio_system_manager.rebalance(hierarchy, operation_data, self.portfolio_returns)

            for m_name, m_obj in o_obj.items():
                model_key = (o_name, m_name)
                model_data = self.sim_data[:dt][model_key]

                if m_obj.model_system_manager:
                    hierarchy = m_obj.model_system_manager.rebalance(hierarchy, model_data, self.portfolio_returns)

                for s_name, s_obj in m_obj.items():
                    strat_key = (o_name, m_name, s_name)
                    strat_data = self.sim_data[:dt][strat_key]

                    if s_obj.strat_system_manager:
                        hierarchy = s_obj.strat_system_manager.rebalance(hierarchy, strat_data, self.portfolio_returns)

        return hierarchy    

    def _money_managers(self, dt, hierarchy): # Updates self.sim_data enside each MM
        # NOTE Deve calcular o lote baseado no pnl_matrix, weight (cap alocado) e metadado do asset
        # Pode usar dados de [:i] para calcular para i+1 (prox iter) para evitar leakage
        for o_name, o_obj in self._iter_portfolio_data():
            operation_key = (o_name)
            operation_data = self.sim_data[:dt][operation_key]

            if self.portfolio_money_manager:
                self.sim_data, hierarchy = self.portfolio_money_manager.calculate_model_position_sizes(hierarchy, operation_data, self.portfolio_returns)

            for m_name, m_obj in o_obj.items():
                model_key = (o_name, m_name)
                model_data = self.sim_data[:dt][model_key]

                if m_obj.model_money_manager:
                    self.sim_data, hierarchy = m_obj.model_money_manager.calculate_asset_strat_position_sizes(hierarchy, model_data, self.portfolio_returns)

                for s_name, s_obj in m_obj.items():
                    strat_key = (o_name, m_name, s_name)
                    strat_data = self.sim_data[:dt][strat_key]

                    if s_obj.strat_money_manager:
                        self.sim_data, hierarchy = s_obj.strat_money_manager.calculate_trade_position_sizes(hierarchy, strat_data, self.portfolio_returns)

        return hierarchy  
    
    # ── Data Handling ───────────────────────────────────────────────

    def _populate_sim_data(self):
        # Populates sim_data with a dict where each key is a timestamp, with pnl/lot/wf of each dt
        # self.sim_data = {
        #     timestamp: {
        #         (op, mod, strat, asset): {
        #             "pnls": { "param_set_1": 0.001, "param_set_2": 0.0 },
        #             "lots": { "param_set_1": 1.0, "param_set_2": 0.0 },
        #             "wf": { "best_param": "...", "pnl": 0.0 } # Opcional
        #         }}}

        for o_name, _, m_name, _, s_name, _, a_name, a_obj in self._iter_portfolio_data():
            key = (o_name, m_name, s_name, a_name)
            
            pnl_df = a_obj.get("pnl_matrix")
            lot_df = a_obj.get("lot_matrix")
            wf_df  = a_obj.get("wf")

            # 1. Process PnL and Lot, ps_param_set column struct
            if pnl_df is not None: # Iterates over df and populates global dict
                for row in pnl_df.iter_rows(named=True):
                    ts = row.pop('ts')
                    if ts not in self.sim_data: self.sim_data[ts] = {}
                    if key not in self.sim_data[ts]: self.sim_data[ts][key] = {}
                    self.sim_data[ts][key]["pnls"] = row
            if lot_df is not None: # Iterates over df and populates global dict
                for row in lot_df.iter_rows(named=True):
                    ts = row.pop('ts')
                    if ts not in self.sim_data: self.sim_data[ts] = {}
                    if key not in self.sim_data[ts]: self.sim_data[ts][key] = {}
                    self.sim_data[ts][key]["lots"] = row

            # 2. Process WF, organized by wf_id
            if wf_df is not None:
                for row in wf_df.iter_rows(named=True):
                    ts = row['datetime']

                    # Makes shure timestamp is added to sim_data if already doesn't have
                    if ts not in self.sim_data: self.sim_data[ts] = {}

                    # Makes shure key of asset/strat already exists for this ts
                    if key not in self.sim_data[ts]: self.sim_data[ts][key] = {}
                    
                    # Init wf dicts if doesn't exist for this key
                    if "wf_pnls" not in self.sim_data[ts][key]:
                        self.sim_data[ts][key]["wf_pnls"] = {}
                        self.sim_data[ts][key]["wf_params"] = {}

                    # Maps PnL and Params for specific ID
                    w_id = row["wf_id"]
                    self.sim_data[ts][key]["wf_pnls"][w_id] = row["pnl"]
                    self.sim_data[ts][key]["wf_params"][w_id] = row["best_param"]

        return True

    def _load_selected_saved_returns_data(self): # Loads all data from selected map (wf, pnl, lot)
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

    # ── Datetime timeline mapping ───────────────────────────────────────────────

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
        repeated_assets = set()

        sm_inds = self.portfolio_system_manager.sm_indicators if (self.portfolio_system_manager and self.portfolio_system_manager.sm_indicators) else {}
        if sm_inds:
            for ind_name, ind_obj in sm_inds.items():
                tf = ind_obj.timeframe
                if tf is None:
                    print(f"< [Error] No timeframe found for System Manager Indicator: {ind_name}. Skipping.")
                    continue

                # Gets Asset define in ind and not in repeated_assets 
                if ind_obj.asset is None:
                    if ind_obj.asset not in repeated_assets:
                        asset_obj = assets.get(ind_obj.asset)
                        asset_df = asset_obj.load(tf, data_source, self.date_start, self.date_end)
                        unique_ind_dts.update(asset_df["datetime"])
                        repeated_assets.add(ind_obj.asset)

                # Else gets each asset defined in sm_assets and not in repeated_assets
                else:
                    sm_assets = self.portfolio_system_manager.sm_assets if self.portfolio_system_manager and self.portfolio_system_manager.sm_assets else []
                    for asset_name in sm_assets:
                        if asset_name not in repeated_assets:
                            asset_obj = assets.get(asset_name)
                            asset_df = asset_obj.load(tf, data_source, self.date_start, self.date_end)
                            unique_ind_dts.update(asset_df["datetime"])
                            repeated_assets.add(ind_obj.asset)

        return unique_ind_dts
    
    def _get_all_mm_ind_datetimes(self, data_source="local"):
        assets = Asset.load_all() # NOTE Deletar futuramente
        unique_ind_dts = set()
        repeated_assets = set()

        mm_inds = self.portfolio_money_manager.mm_indicators if (self.portfolio_money_manager and self.portfolio_money_manager.mm_indicators) else {}
        if mm_inds:
            for ind_name, ind_obj in mm_inds.items():
                tf = ind_obj.timeframe
                if tf is None:
                    print(f"< [Error] No timeframe found for Money Manager Indicator: {ind_name}. Skipping.")
                    continue

                # Gets Asset define in ind and not in repeated_assets 
                if ind_obj.asset is None:
                    if ind_obj.asset not in repeated_assets:
                        asset_obj = assets.get(ind_obj.asset)
                        asset_df = asset_obj.load(tf, data_source, self.date_start, self.date_end)
                        unique_ind_dts.update(asset_df["datetime"])
                        repeated_assets.add(ind_obj.asset)

                # Else gets each asset defined in mm_assets and not in repeated_assets
                else:
                    mm_assets = self.portfolio_money_manager.mm_assets if self.portfolio_money_manager and self.portfolio_money_manager.mm_assets else []
                    for asset_name in mm_assets:
                        if asset_name not in repeated_assets:
                            asset_obj = assets.get(asset_name)
                            asset_df = asset_obj.load(tf, data_source, self.date_start, self.date_end)
                            unique_ind_dts.update(asset_df["datetime"])
                            repeated_assets.add(ind_obj.asset)

        return unique_ind_dts

    # ── Portfolio Optimization ───────────────────────────────────────────────

    def _portfolio_optimization(self):
        # Iterates over previous results and identifies each combination for OS while running
        return True

    # ──────────────────────────────────────────────────────────────────────────── 

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
































