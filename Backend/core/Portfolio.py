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
    sm_mm_map: dict = field(default_factory=dict) # SM/MM for all levels

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
        self.sm_mm_map = portfolio_params.sm_mm_map

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
        aggr_assets_ret, aggr_strats_ret, aggr_models_ret = self._populate_sim_data()
        sim_current_equity = self.portfolio_parameters.get("capital", 100000.0)

        # {(op, mod, strat, asset): {"weight": 0.1, "id": "48_48_48"}} / "id": "parset..."
        active_positions = {} 
        hierarchy = {}
        self.portfolio_returns = {}

        # Checks if is going to simulate portfolio with strat backtest results or asset positions
        has_pnl = any("pnls" in str(key).lower() for key in self.sim_data.keys())
        has_wf = any("wf_pnls" in str(key).lower() for key in self.sim_data.keys())
        portfolio_simulation_with_backtest_results = (has_pnl or has_wf)
        update_func_to_use = self._update_pos_with_backtest_ret if portfolio_simulation_with_backtest_results else self._update_pos_with_assets_ret


        # ->>>>>> MODIFICAR _populate_sim_data
        # 1. _populate_sim_data aproveita e calcula os resultados agregados da soma dos parsets/wf para cada ts
        # 2. envia os resultados agregados para pre_compute, onde vai calcular metricas e indicadores
        # 3. Na timeline vai usar as métricas de rebalance para verificar esses dados 


        # SM and MM Pre-Compute Metrics, Indicators and Rebalance Schedule
        indicator_pool, psm_sch, msm_sch, ssm_sch, pmm_sch, mmm_sch, \
        smm_sch = self._pre_compute_and_calc_rebalance_schedule(self.sim_data, aggr_assets_ret, aggr_strats_ret, aggr_models_ret)
        # NOTE Se não for usar em mais nenhum outro lugar, zerar os aggr

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
            
            # B - Entries at [i] open - MM Tactical Level - Bottom Up (MM can change with exit/entry)
            for idf, pos_info in hierarchy.items():
                if portfolio_simulation_with_backtest_results: pass
                    
                    # B.1. Calculates All SM and MM for each item in hierarchy

                    # B.2. Checks for open positions that can be accomodated in active_positions with current margin and capital
   
                # SSM/SMM -> MSM/MMM -> PSM/PMM

                # -> NOTE Para System e Money M colocar opção de seprar long (lot_size > 0) de short
            
                # Must recalculate position sizes if the rules call for it, else use E defined
                # First-Come First-Served - Allocates 10% until 100% is hit, following hierarchy
                # Static Hierarchy - Limits to how much each level can use margin/capital

            #||=====================================================================================||#
            
            # C - Updates PnL of open positions at [i] ends in previous step
            update_func_to_use(step_dt, active_positions, instance)

            #||=====================================================================================||#
            
            # D - Updates System Managers - Top Down - at [i] ends
            hierarchy = self._system_managers(step_dt, hierarchy, psm_sch, msm_sch, ssm_sch)

            #||=====================================================================================||#
            
            # E - Updates Money Managers - Top Down - at [i] ends - Strategic Level
            hierarchy = self._money_managers(step_dt, hierarchy)

            #||=====================================================================================||#

        print(f"> {step_dt} - Portfolio PnL: {sim_current_equity:..2f}")

        return True

    """
    - If no pnl_matrix or wf then uses the structure up to model and then selects the model strats based on MSM and MMM to simulate Asset Portfolio
    - Se pnl_matrix e not wf então pode fazer walkforward em cada nível em tempo real
    
    """

    # ── Portfolio Defs ───────────────────────────────────────────────

    def _system_managers(self, dt, hierarchy, psm_sch, msm_sch, ssm_sch):
        for o_name, o_obj in self._iter_portfolio_data():
            if dt in psm_sch and self.portfolio_system_manager:
                op_key = (o_name)
                op_data = self.sim_data[:dt][op_key]
                hierarchy = self.portfolio_system_manager.main(hierarchy, op_data, self.portfolio_returns)

            for m_name, m_obj in o_obj.items():
                if dt in msm_sch and m_obj.model_system_manager:
                    mod_key = (o_name, m_name)
                    mod_data = self.sim_data[:dt][mod_key]
                    hierarchy = m_obj.model_system_manager.main(hierarchy, mod_data, self.portfolio_returns)

                for s_name, s_obj in m_obj.items():
                    if dt in ssm_sch.get((m_name, s_name)) and s_obj.strat_system_manager:
                        strat_key = (o_name, m_name, s_name)
                        strat_data = self.sim_data[:dt][strat_key]
                        hierarchy = s_obj.strat_system_manager.main(hierarchy, strat_data, self.portfolio_returns)

        return hierarchy    

    def _money_managers(self, dt, hierarchy, pmm_sch, mmm_sch, smm_sch): # Updates self.sim_data enside each MM
        # NOTE Deve calcular o lote baseado no pnl_matrix, weight (cap alocado) e metadado do asset
        # Pode usar dados de [:i] para calcular para i+1 (prox iter) para evitar leakage
        for o_name, o_obj in self._iter_portfolio_data():
            operation_key = (o_name)
            operation_data = self.sim_data[:dt][operation_key]

            if dt in pmm_sch and self.portfolio_money_manager:
                self.sim_data, hierarchy = self.portfolio_money_manager.calculate_model_position_sizes(hierarchy, operation_data, self.portfolio_returns)

            for m_name, m_obj in o_obj.items():
                model_key = (o_name, m_name)
                model_data = self.sim_data[:dt][model_key]

                if dt in mmm_sch and m_obj.model_money_manager:
                    self.sim_data, hierarchy = m_obj.model_money_manager.calculate_asset_strat_position_sizes(hierarchy, model_data, self.portfolio_returns)

                for s_name, s_obj in m_obj.items():
                    strat_key = (o_name, m_name, s_name)
                    strat_data = self.sim_data[:dt][strat_key]

                    if dt in smm_sch.get((m_name, s_name)) and s_obj.strat_money_manager:
                        self.sim_data, hierarchy = s_obj.strat_money_manager.calculate_trade_position_sizes(hierarchy, strat_data, self.portfolio_returns)

        return hierarchy  

    def _pre_compute_and_calc_rebalance_schedule(self, sim_data, aggr_assets_ret, aggr_strats_ret, aggr_models_ret):
        indicator_pool, psm_sch, msm_sch, ssm_sch, pmm_sch, mmm_sch, smm_sch = {}, {}, {}, {}, {}, {}, {}
        timeline = self.datetime_timeline

        # Portfolio
        if self.portfolio_system_manager:
            self.portfolio_system_manager.pre_compute(timeline, sim_data, aggr_models_ret, indicator_pool)
            psm_sch['root'] = self.portfolio_system_manager.get_schedule(timeline)
        if self.portfolio_money_manager:
            self.portfolio_money_manager.pre_compute(timeline, sim_data, aggr_models_ret, indicator_pool)
            psm_sch['root'] = self.portfolio_money_manager.get_schedule(timeline)

        for _, _, m_name, m_obj, *_ in self._iter_portfolio_data():
            if m_obj.model_system_manager():
                self.model_system_manager.pre_compute(timeline, sim_data, aggr_strats_ret, indicator_pool)
                msm_sch[m_name] = m_obj.model_system_manager.get_schedule(timeline)
            if m_obj.model_money_manager():
                self.model_money_manager.pre_compute(timeline, sim_data, aggr_strats_ret, indicator_pool)
                mmm_sch[m_name] = m_obj.model_money_manager.get_schedule(timeline)

            for s_name, s_obj in m_obj.strats.items():
                s_key = (m_name, s_name)
                if s_obj.strat_system_manager:
                    self.strat_system_manager.pre_compute(timeline, sim_data, aggr_assets_ret, indicator_pool)
                    ssm_sch[s_key] = s_obj.strat_system_manager.get_schedule(timeline)
                if s_obj.strat_money_manager:
                    self.strat_money_manager.pre_compute(timeline, sim_data, aggr_assets_ret, indicator_pool)
                    smm_sch[s_key] = s_obj.strat_money_manager.get_schedule(timeline)

        # # Exemplo para o Strat System Manager no ponto D
        # for s_name, s_obj in m_obj.items():
        #     s_key = (m_name, s_name)
        #     schedule = ssm_sch.get(s_key)
            
        #     # Roda se: schedule é None (sempre) OU o tempo atual está no set
        #     if schedule is None or step_dt in schedule:
        #         hierarchy = s_obj.strat_system_manager.main(hierarchy, ...)

        return indicator_pool, psm_sch, msm_sch, ssm_sch, pmm_sch, mmm_sch, smm_sch
    


    
    def _update_pos_with_backtest_ret(self, step_dt, active_positions, instance):
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

    def _update_pos_with_assets_ret(self, step_dt, active_positions, instance):
        pass

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

        aggr_assets_ret, aggr_strats_ret, aggr_models_ret = {}, {}, {}

        # Timeline template to make shure it aligns 
        timeline_df = pl.DataFrame({"ts": self.datetime_timeline})

        # Temporary Dict to group strat series by model
        strat_accumulator, model_accumulator = {}, {}

        for o_name, _, m_name, _, s_name, _, a_name, a_obj in self._iter_portfolio_data():
            key = (o_name, m_name, s_name, a_name)
            asset_key = (m_name, s_name, a_name)
            strat_key = (m_name, s_name)
            
            pnl_df = a_obj.get("pnl_matrix")
            lot_df = a_obj.get("lot_matrix")
            wf_df  = a_obj.get("wf")

            # Vectorized Asset Aggregation
            if pnl_df is not None:
                param_cols = [c for c in pnl_df.columns if c != "ts"]

                # Join garantees that assets with missing date stay aligned to datetime
                agg_pnl = timeline_df.join(pnl_df, on="ts", how="left").fill_null(0.0)
                asset_series = agg_pnl.select(pl.mean_horizontal(param_cols)).to_series()

                aggr_assets_ret[asset_key] = asset_series

                # Sums to Strategy level
                if strat_key not in strat_accumulator:
                    strat_accumulator[strat_key] = []
                strat_accumulator[strat_key].append(asset_series)

        # Populates sim_data (Quick search dict)
        if pnl_df is not None:
            for row in pnl_df.iter_rows(named=True):
                ts = row.pop('ts')
                if ts not in self.sim_data: self.sim_data[ts] = {}
                if key not in self.sim_data[ts]: self.sim_data[ts][key] = {}
                self.sim_data[ts][key]["pnls"] = row
                
        if lot_df is not None:
            for row in lot_df.iter_rows(named=True):
                ts = row.pop('ts')
                if ts not in self.sim_data: self.sim_data[ts] = {}
                if key not in self.sim_data[ts]: self.sim_data[ts][key] = {}
                self.sim_data[ts][key]["lots"] = row

        if wf_df is not None:
            for row in wf_df.iter_rows(named=True):
                ts = row['datetime']
                if ts not in self.sim_data: self.sim_data[ts] = {}
                if key not in self.sim_data[ts]: self.sim_data[ts][key] = {}
                if "wf_pnls" not in self.sim_data[ts][key]:
                    self.sim_data[ts][key]["wf_pnls"] = {}
                    self.sim_data[ts][key]["wf_params"] = {}
                
                w_id = row["wf_id"]
                self.sim_data[ts][key]["wf_pnls"][w_id] = row["pnl"]
                self.sim_data[ts][key]["wf_params"][w_id] = row["best_param"]
     
        # 3. Final Hierarchical Consolidation
        # Consolidates Strats (Avg of Assets)
        for s_k, list_of_asset_series in strat_accumulator.items():
            strat_df = pl.DataFrame(list_of_asset_series)
            final_strat_series = strat_df.select(pl.mean_horizontal(pl.all())).to_series()
            aggr_strats_ret[s_k] = final_strat_series

            # Feeds Models accumulator
            m_n = s_k[0]
            if m_n not in model_accumulator:
                model_accumulator[m_n] = []
            model_accumulator[m_n].append(final_strat_series)

        # Consolidates Models
        for m_n, list_of_strat_series in model_accumulator.items():
            model_df = pl.DataFrame(list_of_strat_series)
            aggr_models_ret[m_n] = model_df.select(pl.mean_horizontal(pl.all())).to_series()

        return aggr_assets_ret, aggr_strats_ret, aggr_models_ret

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
    from ModelMoneyManager  import ModelMoneyManager,  ModelMoneyManagerParams
    from ModelSystemManager import ModelSystemManager, ModelSystemManagerParams
    from StratMoneyManager  import StratMoneyManager,  StratMoneyManagerParams
    from StratSystemManager import StratSystemManager, StratSystemManagerParams
    from PortfolioMoneyManager  import PortfolioMoneyManager,  PortfolioMoneyManagerParams
    from PortfolioSystemManager import PortfolioSystemManager, PortfolioSystemManagerParams
    from Model import Model, ModelParams
    from Strat import Strat, StratParams
    from MA import MA # type: ignore
    from VAR import VAR # type: ignore
    from ATR_SL import ATR_SL # type: ignore

    # ── Portfolio level ───────────────────────────────────────────────────────
    assets = Asset.load_all()
    eurusd = assets.get("EURUSD")
    gbpusd = assets.get("GBPUSD")
    usdjpy = assets.get("USDJPY")
    
    global_assets = {'EURUSD': eurusd, 'GBPUSD': gbpusd, 'USDJPY': usdjpy} # Global Assets, loaded when app starts up, has all Asset and Portfolios 

    psm = PortfolioSystemManager(PortfolioSystemManagerParams(
        reb_frequency="weekly",
        reb_metric="pnl",
        reb_method="fixed",
        max_active_models=None,
        sm_params={
            "param1": range(2, 4+1, 1),
            "param2": range(20, 50+1, 30),
        },
        sm_indicators={
            'atr': VAR(asset=None, timeframe="M15", window='param2'),
            'var': VAR(asset="model", timeframe="tick", window='param2', alpha=0.01, var_type='parametric', price_col='close'),
            'var_all': VAR(asset="models", timeframe="tick", window='param2', alpha=0.01, var_type='parametric', price_col='close'),
        },
        sm_assets={'EURUSD': eurusd},
    ))

    pmm = PortfolioMoneyManager(PortfolioMoneyManagerParams(
        capital=100_000.0,
        max_capital_exposure=1.0,
        reb_method="fixed",
        alo_allocation={"MA Trend Following": 1.0},  # 100% para o único model
        mm_params={
            "param1": range(4, 12+1, 4),
            "param2": range(20, 80+1, 50),
        },
    ))

    # ── Model level ───────────────────────────────────────────────────────────
    msm = ModelSystemManager(ModelSystemManagerParams(
        reb_frequency="weekly",
    ))

    mmm = ModelMoneyManager(ModelMoneyManagerParams(
        capital=100_000.0,
        max_capital_exposure=1.0,
    ))

    # ── Strat level ───────────────────────────────────────────────────────────
    ssm = StratSystemManager(StratSystemManagerParams(
        reb_frequency="weekly",
    ))

    smm = StratMoneyManager(StratMoneyManagerParams(
        sizing_method="neutral",
        capital_method="fixed",
        capital=100_000.0,
    ))

    # ── portfolio_data com SM/MM em cada nível ────────────────────────────────
    # O portfolio_data carrega os resultados do storage.
    # SM/MM ficam num dict separado mapeado por (model, strat)
    # para não poluir a estrutura de dados de resultados.

    portfolio_data = {
        "operation_test": {
            "MA Trend Following": {
                "AT15": {
                    "EURUSD": {}
                }
            }
        }
    }

    # SM/MM mapeados por nível — referenciados durante a simulação
    sm_mm_map = {
        "portfolio": {
            "psm": psm,
            "pmm": pmm,
        },
        "models": {
            "MA Trend Following": {
                "msm": msm,
                "mmm": mmm,
                "strats": {
                    "AT15": {
                        "ssm": ssm,
                        "smm": smm,
                    }
                }
            }
        }
    }

    portfolio_global_parameters = {
        "capital": 100000.0,
    }

    portfolio = Portfolio(PortfolioParams(
        name="Portfolio_Test",
        portfolio_data=portfolio_data,
        portfolio_parameters=portfolio_global_parameters,
        portfolio_money_manager=pmm,
        portfolio_system_manager=psm,
        sm_mm_map=sm_mm_map,  
    ))

    portfolio._run()


    """
    # XXX -> Resolver problema minutos wf
    # -> IMPORTANTE -> AO INVÉS DE PEGAR DATETIMES DOS WF/PNL, USAR DOS ASSETS DOS MODELOS, JÁ QUE PROVAVELMENTE VAI USAR NO SYSTEM/MONEY
    # -> Como vai ficar a estrutura de System/Money? ter uma lista de indicators e assets que vão ser usados?

    # money_manager_equilizer = {"frequency": 0.5 trades per day, "avg_win", "avg_loss"}    
    
    
    """
































