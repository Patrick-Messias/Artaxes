# Holds >1 models, doesn't define Assets, Server uniquely to Manage Positions between multiple models has to dominate over all MMM and MMA

from dataclasses import dataclass, field
from typing import Optional
from PortfolioSystemManager import PortfolioSystemManager
from PortfolioMoneyManager import PortfolioMoneyManager
from BaseClass import BaseClass, BaseManager
from Storage import Storage
from Asset import Asset
import polars as pl, uuid, sys

sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\indicators')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')

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

class Portfolio(BaseClass, BaseManager): 
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
        self.global_assets = Asset.load_all()
    

    def _simulation(self):
        # 1 - Init, populating sim_data
        sim_current_equity = self.portfolio_parameters.get("capital", 100000.0)

        active_positions = {} 
        hierarchy = {}
        self.portfolio_returns = {}

        # Checks if is going to simulate portfolio with strat backtest results or asset positions
        has_pnl = any("pnls" in str(key).lower() for key in self.sim_data.keys())
        has_wf = any("wf_pnls" in str(key).lower() for key in self.sim_data.keys())
        portfolio_simulation_with_backtest_results = (has_pnl or has_wf)
        update_func_to_use = self._update_pos_with_backtest_ret if portfolio_simulation_with_backtest_results else self._update_pos_with_assets_ret

        # SM and MM Pre-Compute Metrics, Indicators and Rebalance Schedule
        indicator_pool, params_pool, psm_sch, msm_sch, ssm_sch, pmm_sch, mmm_sch, smm_sch \
        = self._pre_compute_and_calc_rebalance_schedule(self.global_assets, self.sm_mm_map)

        # 2 - Run Timeline
        for i, step_dt in enumerate(self.datetime_timeline):

            # Init step data
            self.portfolio_returns[step_dt] = {"assets": {}}
            step_perc_total        =     0.0
            step_pnl_nominal_total =  0.0

            #||=====================================================================================||#
            
            # Exits at [i] open
            for idf, pos_info in active_positions.items():
                pass

            #||=====================================================================================||#
            
            # Entries at [i] open - MM Tactical Level - Bottom Up (MM can change with exit/entry)
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
            
            # Updates PnL of open positions at [i] ends in previous step
            #update_func_to_use(step_dt, active_positions)

            #||=====================================================================================||#

            # Updates System and Money Managers - Top Down - at [i] ends
            hierarchy = self._system_money_managers(i, step_dt, hierarchy, psm_sch, pmm_sch, msm_sch, mmm_sch, ssm_sch, smm_sch)

            #||=====================================================================================||#

            if i < 3 or i > len(self.datetime_timeline)-3: 
                print(f"> {step_dt} - Portfolio PnL: {sim_current_equity:.2f}")

        return True

    # ── Portfolio Defs ───────────────────────────────────────────────

    def _system_money_managers(self, i, dt, hierarchy, psm_sch, pmm_sch, msm_sch, mmm_sch, ssm_sch, smm_sch):
        m_map = self.sm_mm_map

        # Portfolio Level
        psm = m_map.get("managers", {}).get("psm")
        pmm = m_map.get("managers", {}).get("pmm")
        
        # If any of the two need to run, populate data
        if (psm and dt in psm_sch.get(self.name, set())) or (pmm and dt in pmm_sch.get(self.name, set())):
            # Use helpers to take aggr PnL
            p_key = (list(self.portfolio_data.keys())[0],)
            op_df = self._populate_sim_data(p_key, i, data_type="aggr")
            
            if op_df is not None:
                if psm and dt in psm_sch.get(self.name, []):
                    #print("PSM")
                    hierarchy = psm.main(dt, hierarchy, op_df, self.portfolio_returns)
                if pmm and dt in pmm_sch.get(self.name, []):
                    #print("PMM")
                    hierarchy = pmm.main(dt, hierarchy, op_df, self.portfolio_returns)

        # Model and Strat Levels
        for o_name, o_obj, m_name, m_obj, s_name, s_obj, a_name, a_obj in self._iter_portfolio_data():
            #for m_name, m_obj in o_obj.items():
            m_key = (o_name, m_name)
            s_key = (self.name, m_name, s_name)
            
            msm = m_map.get("models", {}).get(m_name, {}).get("managers", {}).get("msm")
            mmm = m_map.get("models", {}).get(m_name, {}).get("managers", {}).get("mmm")

            # Rebalancing of Models
            if (msm and dt in msm_sch.get(m_name, set())) or (mmm and dt in mmm_sch.get(m_name, set())):
                model_df = self._populate_sim_data(m_key, i, data_type="aggr")
                
                if model_df is not None:
                    if msm and dt in msm_sch.get(m_name, set()):
                        #print("msm")
                        hierarchy = msm.main(dt, hierarchy, model_df, self.portfolio_returns)
                    if mmm and dt in mmm_sch.get(m_name, set()):
                        #print("mmm")
                        hierarchy = mmm.main(dt, hierarchy, model_df, self.portfolio_returns)

            # Strat Level
            ssm = m_map.get("models", {}).get(m_name, {}).get("strats", {}).get(s_name, {}).get("managers", {}).get("ssm")
            smm = m_map.get("models", {}).get(m_name, {}).get("strats", {}).get(s_name, {}).get("managers", {}).get("smm")
            
            if (ssm and dt in ssm_sch.get(s_key, set())) or (smm and dt in smm_sch.get(s_key, set())):
                strat_df = self._populate_sim_data((o_name, m_name, s_name), i, data_type="aggr")
                
                if strat_df is not None:
                    if ssm and dt in ssm_sch.get(s_key, set()):
                        #print("ssm")
                        hierarchy = ssm.main(dt, hierarchy, strat_df, self.portfolio_returns)
                    if smm and dt in smm_sch.get(s_key, set()):
                        #print("smm")
                        hierarchy = smm.main(dt, hierarchy, strat_df, self.portfolio_returns)

        return hierarchy

    def _update_pos_with_backtest_ret(self, step_dt, active_positions):
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

    def _update_pos_with_assets_ret(self, step_dt, active_positions):
        pass

    # ── Data Handling ───────────────────────────────────────────────

    # Used to pull real data from parquet from selected source
    def _populate_sim_data(self, key, i, data_type="pnl"):
        ref = self.sim_data.get(key)
        if not ref: return None

        # Se pediu o Aggr de Strat/Model/Portfolio
        if ref["type"] == "aggr":
            return pl.DataFrame({
                "datetime": self.datetime_timeline[:i+1], 
                "pnl": ref["pnl"].slice(0, i + 1) # slice é mais seguro que head no Polars Series
            })
            #return pl.DataFrame({"datetime": self.datetime_timeline[:i+1], "pnl": ref["pnl"].head(i + 1)})

        # Se for Nível Asset
        if ref["type"] == "disk":
            if data_type == "aggr": # ---> NOVO: Manager querendo rankear ativos
                return pl.DataFrame({"datetime": self.datetime_timeline[:i+1], "pnl": ref["aggr_pnl"].head(i + 1)})
            
            if data_type == "pnl":  # ---> Manager querendo rodar WF (lê o disco)
                return pl.read_parquet(ref["pnl_path"]).head(i + 1)
        return None

    # Loads each results data, maps path and generates aggregated results, then clears memory one by one 
    def _load_selected_saved_returns_data(self): 
        storage = Storage(base_path=self.data_storage_base_path)
        self.sim_data = {}

        strat_acc = {} # (op, mod, strat):  [series, series] 
        model_acc = {} # (op, mod):         [series, series]
        portf_acc = {} # (op)               [series, series]
        unique_dts = set()

        # 2. Loading and Alignment of Data
        for op_name, _, m_name, _, s_name, _, a_name, _ in self._iter_portfolio_data():
            a_key = (op_name, m_name, s_name, a_name)
            s_key = (op_name, m_name, s_name)

            # Loads via Storage.load()
            asset_data = storage.load(op_name, m_name, s_name, a_name)

            timeline_df = asset_data.get("timeline")
            #trade_df = asset_data.get("trades")
            wf_df = asset_data.get("wf")

            if timeline_df is None or timeline_df.is_empty():
                print(f"    < [Portfolio._load_selected_saved_returns_data] No timeline for {a_name}, skipping")
                continue

            # Colects unique datetimes from current assets results
            unique_dts.update(timeline_df['datetime'].to_list())

            # Registers lazy reference in sim_data (path to disk)
            base_path = storage._asset_path(op_name, m_name, s_name, a_name)
            self.sim_data[a_key] = {
                "type":          "disk",
                "trades_path":   str(base_path / "trades" / "trades.parquet"),
                "matrix_path":   str(base_path / "matrix" / "trades_matrix.parquet"),
                "wf_path":       str(base_path / "wfm"    / "wf.parquet"),
            }

            # WF memory map (if wf.parquet exists)
            if wf_df is not None and not wf_df.is_empty(): # timeline already has ps_id linked via trade_id, use exits to map
                #exits = timeline_df.filter(pl.col("event") == "exit")
                self.sim_data[a_key]["wf_memory_map"] = self._build_wf_memory_map(wf_df, timeline_df) # exits

            # RAM aggregated series (avg of parsets by datetime)
            #exits_only = timeline_df.filter(pl.col("event") == "exit")
            if not timeline_df.is_empty(): #exits_only
                asset_mean = (
                    timeline_df #exits_only
                    .group_by("datetime")
                    .agg(pl.col("pnl").mean())
                    .sort("datetime")["pnl"]
                    .alias(a_name)
                )
                self.sim_data[a_key]["aggr_pnl"] = asset_mean
                strat_acc.setdefault(s_key, []).append(asset_mean)

        # Constructs global timeline from collected data
        self.datetime_timeline = sorted(unique_dts)
        timeline_global = pl.DataFrame({"datetime": self.datetime_timeline})

        # Aligns all series to global timeline and consolidates hierarchy
        for s_key, asset_list in strat_acc.items():
            aligned = [
                timeline_global.join(
                    pl.DataFrame({"datetime": self.datetime_timeline, "pnl": s}),
                    on="datetime", how="left"
                ).fill_null(0.0)["pnl"]
                for s in asset_list
            ]
            strat_mean = pl.DataFrame(aligned).select(pl.mean_horizontal(pl.all())).to_series().alias(s_key[2])
            self.sim_data[s_key] = {"pnl": strat_mean, "type": "aggr"}
            model_acc.setdefault((s_key[0], s_key[1]), []).append(strat_mean)

        for m_key, strat_list in model_acc.items():
            model_mean = pl.DataFrame(strat_list).select(pl.mean_horizontal(pl.all())).to_series().alias(m_key[1])
            self.sim_data[m_key] = {"pnl": model_mean, "type": "aggr"}
            portf_acc.setdefault((m_key[0]), []).append(model_mean)

        for op_key, model_list in portf_acc.items():
            port_mean = pl.DataFrame(model_list).select(pl.mean_horizontal(pl.all())).to_series().alias(op_key[0])
            self.sim_data[op_key] = {"pnl": port_mean, "type": "aggr"}

        if self.datetime_timeline:
            print(f"     > Timeline: {self.datetime_timeline[0]} → {self.datetime_timeline[-1]} ({len(self.datetime_timeline)} pts)")

        return True
    
    def _build_wf_memory_map(self, wf_df: pl.DataFrame, trades_df: pl.DataFrame) -> dict:
        # Constructs O(1) access map for WF data during a simulation
        # # wf_df: wf.parquet - datetime | pnl | best_param | wf_id
        # trades_df: trade data with - datetime | ps_id | pnl | lot_size 
        # returns {datetime: {wf_id: (pnl, lot_size)}}

        if wf_df is None or wf_df.is_empty():
            return {}
        
        # Normalizes names to lowercase to garantee match
        wf_df = wf_df.with_columns(pl.col("best_param").str.to_lowercase())
        trades_df = trades_df.with_columns(pl.col("ps_id").str.to_lowercase())

        # Join: for each WF line, takes pnl and lot_size from exits_df in same datetime and ps_id
        joined = (
            wf_df
            .join(
                trades_df.select(["datetime", "ps_id", "pnl", "lot_size"])
                        .rename({"ps_id": "best_param", "pnl": "val_pnl", "lot_size": "val_lot"}),
                on=["datetime", "best_param"],
                how="left"
            )
            .fill_null(0.0)
        )

        memory_map = {}
        for row in joined.iter_rows(named=True):
            dt  = row["datetime"]
            wid = str(row["wf_id"])
            memory_map.setdefault(dt, {})[wid] = (row["val_pnl"], row["val_lot"])

        return memory_map

    def _pre_compute_and_calc_rebalance_schedule(self, global_assets, sm_mm_map):
        indicator_pool, params_pool = {}, {}
        psm_sch, msm_sch, ssm_sch, pmm_sch, mmm_sch, smm_sch = {}, {}, {}, {}, {}, {}
        
        timeline = self.datetime_timeline
        sd = self.sim_data

        # 1. Agregação e Separação (Long/Short)
        # Extraímos as Series já calculadas no load
        aggr_models_ret = {k: v["pnl"] for k, v in sd.items() if len(k) == 2 and v.get("type") == "aggr"}
        aggr_strats_ret = {k: v["pnl"] for k, v in sd.items() if len(k) == 3 and v.get("type") == "aggr"}
        aggr_assets_ret = {k: v["aggr_pnl"] for k, v in sd.items() if len(k) == 4 and "aggr_pnl" in v}

        # Aplica lógica de Long/Short se necessário (gera novas chaves no dict)
        aggr_models_ret = BaseClass.separate_long_short_returns(aggr_models_ret)
        aggr_strats_ret = BaseClass.separate_long_short_returns(aggr_strats_ret)
        aggr_assets_ret = BaseClass.separate_long_short_returns(aggr_assets_ret)

        # --- PORTFOLIO LEVEL ---
        p_magrs = sm_mm_map.get("managers", {})
        
        # System Manager (PSM)
        psm = p_magrs.get("psm") or PortfolioSystemManager(PortfolioSystemManagerParams())
        indicator_pool, sd, params_pool = psm.pre_compute(global_assets, timeline, sd, aggr_models_ret, indicator_pool)
        psm_sch[self.name] = psm.get_schedule(timeline)
        p_magrs["psm"] = psm
            
        # Money Manager (PMM)
        pmm = p_magrs.get("pmm") or PortfolioMoneyManager(PortfolioMoneyManagerParams())
        indicator_pool, sd, params_pool = pmm.pre_compute(global_assets, timeline, sd, aggr_models_ret, indicator_pool)
        pmm_sch[self.name] = pmm.get_schedule(timeline)
        p_magrs["pmm"] = pmm

        # --- MODELS LEVEL ---
        for m_name, m_info in sm_mm_map.get("models", {}).items():
            m_magrs = m_info.get("managers", {})
            m_strats_filter = {k: v for k, v in aggr_strats_ret.items() if k[1] == m_name}

            # Model System Manager (MSM)
            msm = m_magrs.get("msm") or ModelSystemManager(ModelSystemManagerParams())
            indicator_pool, sd, params_pool = msm.pre_compute(global_assets, timeline, sd, m_strats_filter, indicator_pool)
            msm_sch[m_name] = msm.get_schedule(timeline)
            m_magrs["msm"] = msm

            # Model Money Manager (MMM)
            mmm = m_magrs.get("mmm") or ModelMoneyManager(ModelMoneyManagerParams())
            indicator_pool, sd, params_pool = mmm.pre_compute(global_assets, timeline, sd, m_strats_filter, indicator_pool)
            mmm_sch[m_name] = mmm.get_schedule(timeline)
            m_magrs["mmm"] = mmm

            # --- STRATS LEVEL ---
            for s_name, s_info in m_info.get("strats", {}).items():
                s_magrs = s_info.get("managers", {})
                s_key = (self.name, m_name, s_name)
                s_assets_filter = {k: v for k, v in aggr_assets_ret.items() if k[:3] == s_key}

                # Strat System Manager (SSM)
                ssm = s_magrs.get("ssm") or StratSystemManager(StratSystemManagerParams())
                indicator_pool, sd, params_pool = ssm.pre_compute(global_assets, timeline, sd, s_assets_filter, indicator_pool)
                ssm_sch[s_key] = ssm.get_schedule(timeline)
                s_magrs["ssm"] = ssm

                # Strat Money Manager (SMM)
                smm = s_magrs.get("smm") or StratMoneyManager(StratMoneyManagerParams())
                indicator_pool, sd, params_pool = smm.pre_compute(global_assets, timeline, sd, s_assets_filter, indicator_pool)
                smm_sch[s_key] = smm.get_schedule(timeline)
                s_magrs["smm"] = smm

        return indicator_pool, params_pool, psm_sch, msm_sch, ssm_sch, pmm_sch, mmm_sch, smm_sch
    
    # ── Datetime timeline mapping ───────────────────────────────────────────────
    
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

        sm_inds = self.portfolio_system_manager.indicators if (self.portfolio_system_manager and self.portfolio_system_manager.indicators) else {}
        if sm_inds:
            for ind_name, ind_obj in sm_inds.items():
                tf = ind_obj.timeframe
                if tf is None:
                    print(f"< [Error] No timeframe found for System Manager Indicator: {ind_name}. Skipping.")
                    continue

                # Gets Asset define in ind and not in repeated_assets 
                if ind_obj.asset is not None:
                    if ind_obj.asset not in repeated_assets and ind_obj.asset not in ["each_aggr", "all_aggr"]:
                        asset_obj = assets.get(ind_obj.asset)
                        asset_df = asset_obj.load(tf, data_source, self.date_start, self.date_end)
                        unique_ind_dts.update(asset_df["datetime"])
                        repeated_assets.add(ind_obj.asset)

                # Else gets each asset defined in assets and not in repeated_assets
                else:
                    assets = self.portfolio_system_manager.assets if self.portfolio_system_manager and self.portfolio_system_manager.assets else []
                    for asset_name in assets:
                        if asset_name not in repeated_assets:
                            asset_obj = self.global_assets.get(asset_name)
                            asset_df = asset_obj.load(tf, data_source, self.date_start, self.date_end)
                            unique_ind_dts.update(asset_df["datetime"])
                            repeated_assets.add(ind_obj.asset)

        return unique_ind_dts
    
    def _get_all_mm_ind_datetimes(self, data_source="local"):
        assets = Asset.load_all() # NOTE Deletar futuramente
        unique_ind_dts = set()
        repeated_assets = set()

        mm_inds = self.portfolio_money_manager.indicators if (self.portfolio_money_manager and self.portfolio_money_manager.indicators) else {}
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

                # Else gets each asset defined in assets and not in repeated_assets
                else:
                    assets = self.portfolio_money_manager.assets if self.portfolio_money_manager and self.portfolio_money_manager.assets else []
                    for asset_name in assets:
                        if asset_name not in repeated_assets:
                            asset_obj = assets.get(asset_name)
                            asset_df = asset_obj.load(tf, data_source, self.date_start, self.date_end)
                            unique_ind_dts.update(asset_df["datetime"])
                            repeated_assets.add(ind_obj.asset)

        return unique_ind_dts

    # ── Global ───────────────────────────────────────────────

    def _iter_portfolio_data(self):
        for op_name, op_obj in self.portfolio_data.items():
            for m_name, m_obj in op_obj.items():
                for s_name, s_obj in m_obj.items():
                    for a_name, a_obj in s_obj.items():
                        yield op_name, op_obj, m_name, m_obj, s_name, s_obj, a_name, a_obj

    # ── Portfolio Optimization ───────────────────────────────────────────────

    def _portfolio_optimization(self):
        # Iterates over previous results and identifies each combination for OS while running
        return True

    # ──────────────────────────────────────────────────────────────────────────── 

    """"""
    # 1. -> PRIORITARIO
    # - usar timeline unificada de Storage.load para tudo
    # - padronizar tuple/key EM TODO LUGAR (USAR TUPLE)
    # - mantem mesmo path para todos


    # 2. Continuação
    # -> Saidas: 
    # Para cada trade:
    # - Se date_exit == datetime então sai 
    # - Se o mae ou mfe do datetime atual passou os limites de ganho ou perda do trade definido pelo MM então fecha 
    # Para todos: se pnl do portfolio chegar a x ou y então encerra tudo (ganho/perda mês)

    # -> Entradas: 
    # - SSM decide se First Come First Serve ou 1 trade por Strat por nível ou 1 trade por Asset
    # - Se posição aberta, verifica hierarchy, onde foi rankeado os pretendentes basedo em todos os níveis pelos SM, verifica se pode entrar durante trade aberto ou apenas na abertura
    # - Pega e executa a entrada nos trades válidos, 1 por 1, atualizando as variáveis globais (MM) a cada etapa, ao executar ele vai calcular o lote baseado nos dados unicos do ativo que o trade foi executado, analisando o lot_min, leverage, etc.

    # -> Atualização PnL:
    # - Cada posição aberta != da aberta no datetime vai atualizar o PnL, verificando o MAE e MFE para decidir se está tudo bem, atualiza lote (def que pode ser enviada, default None)
    # - Para cada trade em active_positions deve puxar os dados do trades_matrix, verificar se precisa atualizar o lot (diminuir ou aumentar, pode ser uma def enviada, default None, mantêm mesma coisa até saida) para saber o PnL * Lot atualizado
    # - Criando e enviando a imagem do datetime para self.portfolio_returns



    def _run(self):
        # Data Init - Loads data, saves unique datetimes and generates aggr results
        print("     > Populating Portfolio Data from Database")
        self._load_selected_saved_returns_data()

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
        params={
            "param1": range(2, 4+1, 1),
            "param2": range(20, 50+1, 30),
        },
        indicators={
            'atr': VAR(asset=None, timeframe="M15", window='param2'),
            'var': VAR(asset="all_aggr", timeframe="tick", window='param2', alpha=0.01, var_type='parametric', price_col='close'),
            'var_all': VAR(asset="each_aggr", timeframe="tick", window='param2', alpha=0.01, var_type='parametric', price_col='close'),
        },
        assets={'EURUSD'},
    ))

    pmm = PortfolioMoneyManager(PortfolioMoneyManagerParams(
        capital=100000.0,
        max_capital_exposure=1.0,
        reb_frequency="weekly",
        reb_metric="pnl",
        reb_method="fixed",
        reb_lookback=252,
        reb_deviation_func=None,
        params={
            "param1": range(4, 12+1, 4),
            "param2": range(20, 80+1, 50),
        },
        indicators=None,
    ))

    # ── Model level ───────────────────────────────────────────────────────────
    msm = ModelSystemManager(ModelSystemManagerParams(
        reb_frequency="weekly",
    ))

    mmm = ModelMoneyManager(ModelMoneyManagerParams(
        capital=100000.0,
        reb_frequency="weekly",
    ))

    # ── Strat level ───────────────────────────────────────────────────────────
    ssm = StratSystemManager(StratSystemManagerParams(
        reb_frequency="weekly",
    ))

    smm = StratMoneyManager(StratMoneyManagerParams(
        capital=100000.0,
        reb_frequency="weekly",
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
        "managers": {"psm": psm, "pmm": pmm},
        "models": {
            "MA Trend Following": {
                "managers": {"msm": msm, "mmm": mmm},
                "strats": {
                    "AT15": {"managers": {"ssm": ssm, "smm": smm}}
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
        sm_mm_map=sm_mm_map,  
    ))

    portfolio._run()

    """
    DEFAULT
    Portofolio
    SM: Rankear com EWPCA (aggr dos models) para definir PC1 e PC2

        LONG_SHORT_FACTOR = 0.5 # Balanced by default
    MM: Divide capital entre modelos usando o peso do SM
        (tf1 = 0.3 a 0.6 | mr1 = 0.3 a 0.6 | sn1 = 0.4)
    
    Models (Trend_Following_1, Mean_Reversion_1 e Seasonality_1)
    SM: RRG and Correlation
    MM: 

    DEFAULT

    Portfolio SM: EWPCA → RRG+Hurst sobre aggr_models_ret
        LONG_trend          = (RRG_PC1 == ('Improving or 'Leading'))    & (Hurst_PC1 > 0.5)
        LONG_reversion      = (RRG_PC2 == 'Lagging')                    & (Hurst_PC2 < 0.5)
        SHORT_trend         = (RRG_PC1 == ('Lagging' or 'Weakening'))   & (Hurst_PC1 > 0.5)
        SHORT_reversion     = (RRG_PC2 == 'Weakening')                  & (Hurst_PC2 < 0.5)
        LONG_SHORT_FACTOR: dinâmico via RRG ou fixo pelo usuário (0.5 default)

    Portfolio MM: normaliza scores → peso por model
        capital_model = capital_total * peso_model * LONG_SHORT_FACTOR(lado)
        bounds: min=0.1, max=0.6 por model

    ────────────────────────────────────────────
    Model SM: Sharpe rolling + correlação entre strats
        score_strat = sharpe_rolling(lookback) * (1 - avg_corr)
        score_asset = sharpe_rolling(lookback) * (1 * avg_corr)
        LONG_SHORT_FACTOR herdado do Portfolio, ajustado pelo long_ratio do param_set

    Model MM: normaliza scores → peso por strat
        capital_strat = capital_model * peso_strat
        bounds: min=0.05, max=0.5 por strat

    ────────────────────────────────────────────
    Strat SM: Walkforward (já implementado)
        seleciona param_set com maior lucro
        long_ratio calculado do lot_matrix histórico

    Strat MM: StratMoneyManager (já implementado)
        sizing por trade baseado em capital_strat alocado pelo MMM

    DEFAULT

    Portfolio SM: EWPCA → RRG+Hurst sobre aggr_models_ret
        scores: LONG_trend, LONG_reversion, SHORT_trend, SHORT_reversion
        LONG_SHORT_FACTOR: dinâmico via RRG ou fixo pelo usuário (0.5 default)

    Portfolio MM: normaliza scores → peso por model
        capital_model = capital_total * peso_model * LONG_SHORT_FACTOR(lado)
        bounds: min=0.1, max=0.6 por model

    ────────────────────────────────────────────
    Model SM: Sharpe rolling + correlação entre strats
        score_strat = sharpe_rolling(lookback) * (1 - avg_corr)
        LONG_SHORT_FACTOR herdado do Portfolio, ajustado pelo long_ratio do param_set

    Model MM: normaliza scores → peso por strat
        capital_strat = capital_model * peso_strat
        bounds: min=0.05, max=0.5 por strat

    ────────────────────────────────────────────
    Strat SM: Walkforward (já implementado)
        seleciona param_set ou wf_id pelo IS/OOS
        long_ratio calculado do lot_matrix histórico

    Strat MM: StratMoneyManager (já implementado)
        sizing por trade baseado em capital_strat alocado pelo MMM

    """



    """
    # XXX -> Resolver problema minutos wf
    # -> IMPORTANTE -> AO INVÉS DE PEGAR DATETIMES DOS WF/PNL, USAR DOS ASSETS DOS MODELOS, JÁ QUE PROVAVELMENTE VAI USAR NO SYSTEM/MONEY
    # -> Como vai ficar a estrutura de System/Money? ter uma lista de indicators e assets que vão ser usados?

    # money_manager_equilizer = {"frequency": 0.5 trades per day, "avg_win", "avg_loss"}    
    
    


        def _debug_sim_data_structure(self):
        print("\n" + "="*50)
        print("DEBUG: ESTRUTURA DO SIM_DATA")
        print("="*50)
        
        if not self.sim_data:
            print("ERRO: sim_data está VAZIO!")
            return

        for key, info in self.sim_data.items():
            tipo = info.get("type", "N/A")
            length = len(info["pnl"]) if "pnl" in info else "N/A"
            aggr_pnl = "Sim" if "aggr_pnl" in info else "Não"
            
            # Formata a visualização da tupla/chave
            key_desc = f"LEN({len(key)}) {key}"
            print(f"Key: {key_desc:<40} | Type: {tipo:<6} | PnL Size: {length:<8} | Has Aggr: {aggr_pnl}")
        
        print("="*50 + "\n")
    """





'''
    def _map_all_unique_datetimes(self, external_dts=None):
        unique_dts = set()
        
        if external_dts:
            unique_dts.update(external_dts)

        # Se houver indicadores globais que não estão nos arquivos de PnL
        if self.portfolio_system_manager and self.portfolio_system_manager.indicators:
            unique_dts.update(self._get_all_sm_ind_datetimes())

        self.datetime_timeline = sorted(list(unique_dts))
        
        # Print de conferência
        if self.datetime_timeline:
            print(f"> Timeline: {self.datetime_timeline[0]} até {self.datetime_timeline[-1]}")
'''




