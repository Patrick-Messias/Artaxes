# Holds >1 models, doesn't define Assets, Server uniquely to Manage Positions between multiple models has to dominate over all MMM and MMA

from dataclasses import dataclass, field
from typing import Optional, Literal
from BaseClass import BaseClass, BaseManager
from Storage import Storage
from Asset import Asset
import polars as pl, numpy as np, uuid, sys, os, json, bisect
from PortfolioSystemManager import PortfolioSystemManager, PortfolioSystemManagerParams
from ModelSystemManager import ModelSystemManager, ModelSystemManagerParams
from StratSystemManager import StratSystemManager, StratSystemManagerParams
from MoneyManager import MoneyManager, MoneyManagerParams

sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\indicators')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')

@dataclass
class PortfolioParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    portfolio_data: dict=None
    portfolio_parameters: dict=None 
    sm_mm_map: dict = field(default_factory=dict) # SM/MM for all levels
    global_assets: dict = None

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

        self.sm_mm_map = portfolio_params.sm_mm_map
        self.global_assets = portfolio_params.global_assets if portfolio_params.global_assets is not None else Asset.load_all() # NOTE Deletar futuramente 

        self.date_start = portfolio_params.date_start
        self.date_end = portfolio_params.date_end
        self.use_portfolio_asset_data = portfolio_params.use_portfolio_asset_data
        self.global_datetime_prefix = portfolio_params.global_datetime_prefix

        self.datetime_timeline = portfolio_params.datetime_timeline
        self.hierarchy: dict={}
        self.sim_data: dict= {}
        self.iter_data_cache: dict={}
        self.portfolio_returns: list=[]
        self.active_positions: dict={}
        self.indicator_pool: dict={}
        self.storage = Storage(base_path=portfolio_params.data_storage_base_path)


    # NOTE Up Next:
    # - Transform Portfolio, Model and Strat SM and MM in @dataclass
    # - Focus all steps (rank, filter, rebalance) on SM and MM 


      
    def _exits_and_updates(self, idx_datetime): # Exits positions based on previous data and open only [i] data
        if not self.active_positions:
            return True
        
        pmm = self.sm_mm_map["managers"]["pmm"]

        # Iterates over active_positions and checks if event="exit" or other exit conditions
        for pos_key in list(self.active_positions.keys()): 
            pos_obj = self.active_positions[pos_key]

            p_name, m_name, s_name, a_name, target_id = pos_key
            a_key = (p_name, m_name, s_name, a_name)
            curr_trade_data = self._get_iter_data(a_key, idx_datetime, target_id, pos_obj["is_wf"])
            
            if not curr_trade_data:
                continue

            event = curr_trade_data["event"]
            base_margin_now = curr_trade_data["margin_required"]
            scale_factor = pos_obj["scale_factor"]
            allocated_margin = pos_obj["allocated_margin"]
            pnl = curr_trade_data.get("pnl", 0.0) * scale_factor
            perc = curr_trade_data.get("perc", 0.0)

            # Checks for Exits
            if event in ["exit", "entry"]:
                # Gives back margin
                pmm.cash += allocated_margin
                pmm.allocated_margin -= allocated_margin

                self.portfolio_returns.append({
                    "c_key": pos_key,
                    "datetime": idx_datetime,
                    "allocated_margin": 0.0,
                    "lot_size": pos_obj["lot_size"],
                    "pnl": pnl,
                    "perc": perc,
                    "portfolio_weight": pos_obj["portfolio_weight"],
                    "event": "exit",
                })

                del self.active_positions[pos_key]

            # Checks for Updates
            if event == "update":
                # Recalcula margem atual necessaria
                new_actual_margin = base_margin_now * scale_factor

                # Adjusts Money Manager difference
                margin_delta = new_actual_margin - allocated_margin
                pmm.cash -= margin_delta
                pmm.allocated_margin += margin_delta

                # Partial lot_size update (WIP in CPP)
                # pos_obj["lot_size"] = curr_trade_data.get("lot_size") * scale_factor

                self.portfolio_returns.append({
                    "c_key": pos_key,
                    "datetime": idx_datetime,
                    "allocated_margin": 0.0,
                    "lot_size": pos_obj["lot_size"],
                    "pnl": pnl,
                    "perc": perc,
                    "portfolio_weight": pos_obj["portfolio_weight"],
                    "event": "update",
                })
 
        pmm.update_states(self.active_positions)
            
        return True

    def _entries(self, idx_datetime): # Enters new positions based on previous data and open only [i] data
        pmm = self.sm_mm_map["managers"]["pmm"]

        # Checks if there's still margin and space to open new positions
        if pmm and pmm.available_margin <= 0:
            print(f"- Not enought margin to open new position - DELETE THIS DEBUG PRINT AFTER") # NOTE
            return False
        
        new_entry_candidates = []

        # Filters new entry candidates
        for op_name, _, m_name, _, s_name, _, *_ in self._iter_portfolio_data():
            m_key = (op_name, m_name)
            s_key = (op_name, m_name, s_name)
            s_data = self.hierarchy[m_key]["strats"][s_key]
            m_data = self.hierarchy[m_key]

            model_weight = m_data.get("weight", 1.0)
            strat_weight = s_data.get("weight", 1.0) 

            if not s_data.get("active", True): continue
            trade_update_can_enter = s_data.get("trade_update_can_enter", False)

            for a_key, a_data in s_data["assets"].items():
                if not a_data.get("active", True): continue

                a_name = a_key[-1]
                asset_obj = self.global_assets.get(a_name)
            
                if not asset_obj: 
                    print(f"    < [Portfolio._entries] Error, asset_obj not found for asset: {a_name}")
                    continue

                # Creates lista with data for any enabled wf or ps ids
                wf_ids = a_data.get("wf_id")
                ps_ids = a_data.get("ps_id")
                a_side = a_data.get("side", "both")
                a_separate_ls = a_data.get("separate_ls", False)

                def eval_canditate(target_id, is_wf: bool):
                    curr_data = self._get_iter_data(a_key, idx_datetime, target_id, is_wf)

                    if curr_data and (
                        curr_data["event"] in ["entry", "flash_trade"] or \
                        (curr_data["event"] == "update" and trade_update_can_enter)
                    ):
                        lot_size = curr_data.get("lot_size")
                        trade_direction = "long" if lot_size > 0 else "short"

                        # Determines correct weight
                        if a_separate_ls and a_side == "both":
                            asset_weight = a_data.get(trade_direction, 1.0)
                        else:
                            asset_weight = a_data.get(a_side, 1.0)
                        trade_weight = model_weight * strat_weight * asset_weight["weight"]

                        pos_key = (*a_key, target_id)

                        # Already builds final list to sort
                        new_entry_candidates.append({
                            "key": pos_key,
                            "trade_data": curr_data,
                            "asset_obj": asset_obj,
                            "weight": trade_weight,
                            "margin_required": curr_data.get("margin_required"),
                            "event": curr_data["event"],
                            "is_wf": is_wf,
                        })
                
                for wf_key, wf_val in wf_ids.items():
                    if wf_val: eval_canditate(wf_key, True)
                for ps_key, ps_val in ps_ids.items():
                    if ps_val: eval_canditate(ps_key, False)
                        
        # No new offers
        if not new_entry_candidates:
            return True
        
        # Orders by: (1) Highest weight - (2) Lowest margin
        new_entry_candidates.sort(key=lambda x: (x["weight"], -x["margin_required"]), reverse=True)

        # First ranks and filters entries
        for can in new_entry_candidates:
            c_key = can["key"]
            event = can["event"]
            min_margin_required = can["margin_required"]
            trade_data = can["trade_data"]
            asset_obj = can["asset_obj"]

            pnl = trade_data["pnl"]
            perc = trade_data["perc"]
            min_trade_lot_size = trade_data["lot_size"] 
            a_lot_step = getattr(asset_obj, "lot_step", min_trade_lot_size)

            # Capital 
            allocated_capital = pmm.calculate_position_size(can["weight"])

            # Lot
            lot_size = pmm.calculate_lot_size(min_trade_lot_size, min_margin_required, allocated_capital, a_lot_step)
            
            # Recalculated real margin required for actual calculated lot_size
            lot_multiplier = lot_size / min_trade_lot_size
            actual_margin_required = lot_multiplier * min_margin_required

            if pmm.available_margin >= actual_margin_required:
                if event in ["entry", "update"]: # NOTE obs: se update dependendo pode ou não contar o pnl/perc agora
                    self.active_positions[c_key] = {
                        "entry_datetime": idx_datetime,
                        "lot_size": lot_size,
                        "scale_factor": lot_multiplier,
                        "allocated_margin": actual_margin_required,
                        "portfolio_weight": can["weight"],
                        "is_wf": can["is_wf"]
                    }

                    self.portfolio_returns.append({
                        "c_key": c_key,
                        "datetime": idx_datetime,
                        "allocated_margin": actual_margin_required,
                        "lot_size": lot_size,
                        "pnl": pnl * lot_multiplier,
                        "perc": perc,
                        "portfolio_weight": can["weight"],
                        "event": "entry",
                    })

                    pmm.available_margin -= actual_margin_required
                    pmm.allocated_margin += actual_margin_required
                    pmm.update_states(self.active_positions)

                # Handles case of flash trades, where they are opened and closed at same candle datetime
                elif event == "flash_trade":
                    self.portfolio_returns.append({
                        "c_key": c_key,
                        "datetime": idx_datetime,
                        "allocated_margin": actual_margin_required,
                        "lot_size": lot_size,
                        "pnl": pnl *lot_multiplier,
                        "perc": perc,
                        "portfolio_weight": can["weight"],
                        "event": "flash_trade",
                    })
                    continue
            else: 
                continue

        return True
     
    def _simulation(self):
        # 1 - Init, populating sim_data
        self.sim_current_equity = self.portfolio_parameters.get("capital", 100000.0)
        self.current_idx = 0

        # Checks if is going to simulate portfolio with strat backtest results or asset positions
        has_pnl = any("pnls" in str(key).lower() for key in self.sim_data.keys())
        has_wf = any("wf_pnls" in str(key).lower() for key in self.sim_data.keys())
        portfolio_simulation_with_backtest_results = (has_pnl or has_wf)

        # SM and MM Pre-Compute Metrics, Indicators and Rebalance Schedule
        params_pool, psm_sch, msm_sch, ssm_sch \
        = self._pre_compute_and_calc_rebalance_schedule(self.global_assets, self.sm_mm_map) # NOTE Futuramente salvar os indicadores calculados para SQL/parquet para não pesar memória

        # 2 - Run Timeline
        for i, step_dt in enumerate(self.datetime_timeline):
            self.current_idx = i

            # Init step data
            step_perc_total        =     0.0
            step_pnl_nominal_total =  0.0

            #||=====================================================================================||#

            # Exits and Updates
            self._exits_and_updates(step_dt)

            # Entries
            self._entries(step_dt)
   
            #||=====================================================================================||#
            
            # Updates System and Money Managers - Top Down - at [i] ends
            self._system_money_managers(i, step_dt, psm_sch, msm_sch, ssm_sch)
                                                    
            #||=====================================================================================||#
                           
            if i < 3 or i > len(self.datetime_timeline)-4: 
                print(f"> {step_dt} - Portfolio PnL: {self.sim_current_equity:.2f}")
            
        return True
    
    # ── Portfolio Defs ───────────────────────────────────────────────

    def _system_money_managers(self, i, dt, psm_sch, msm_sch, ssm_sch):
        m_map = self.sm_mm_map
        p_name = self.name
        p_key = (p_name,)

        # If any of the two need to run, populate data
        if (dt in psm_sch.get(p_name, set())):
            psm = m_map.get("managers", {}).get("psm")
            if psm and dt in psm_sch.get(p_name, set()):
                self.hierarchy = psm.main(i, dt, p_key)
                #print("PSM")

        # Model and Strat Levels
        seen_models = set()
        seen_strats = set()

        for op_name, _, m_name, _, s_name, _, a_name, _ in self._iter_portfolio_data():
            m_key = (op_name, m_name)
            s_key = (op_name, m_name, s_name)

            # --- NÍVEL MODELO (MSM / MMM) ---
            if m_key not in seen_models:
                seen_models.add(m_key)
                
                if (dt in msm_sch.get(m_key, set())):
                    msm = m_map.get("models", {}).get(m_name, {}).get("managers", {}).get("msm")

                    if msm and dt in msm_sch.get(m_key, set()): 
                        self.hierarchy = msm.main(i, dt, m_key)
                        #print("msM")

            # Strat level — executa apenas 1x por strat
            if s_key not in seen_strats:
                seen_strats.add(s_key)
                
                if (dt in ssm_sch.get(s_key, set())):
                    ssm = m_map.get("models", {}).get(m_name, {}).get("strats", {}).get(s_name, {}).get("managers", {}).get("ssm")

                    if ssm and dt in ssm_sch.get(s_key, set()):
                        self.hierarchy = ssm.main(i, dt, s_key)
                        #print("ssm")

        return True

    # ── Data Handling ───────────────────────────────────────────────

    # Gets index parset trade data        
    def _get_iter_data(self, key, step_datetime, psid_wfid, is_wf: bool):
        # Resolves param set
        if is_wf:
            asset_data = self.storage.load(key)
            wf_map = asset_data.get("wf")
            ps_id = self._get_psid_with_wfid(key, step_datetime, wf_map, psid_wfid)
            if not ps_id: 
                #print(f"    < [Portfolio._get_iter_data] Error unable to extract ps_id form wf_id")
                return None
        else:
            ps_id = psid_wfid
            
        # Defines main param set key
        pos_key = (*key, ps_id)

        # Constructs asset's timeline into dict O(1)
        if not hasattr(self, 'iter_data_cache'): self.iter_data_cache = {}

        if pos_key not in self.iter_data_cache:
            asset_data = self.storage.load(key)
            if not asset_data: 
                print(f"    < [Portfolio._get_iter_data] Error loading data from storage.load, None")
                return None
            
            timeline_df = asset_data.get("timeline")
            if timeline_df is None or timeline_df.is_empty():
                self.iter_data_cache[pos_key] = {}
                print(f"    < [Portfolio._get_iter_data] Error getting timeline from asset_data")
                return None
                
            # Filters correct ps_id only unce
            df_filtered = timeline_df.filter(pl.col("ps_id") == ps_id)
            
            # Converts DataFrame filtering to python native dict list
            dicts = df_filtered.to_dicts()
            
            # If list is empty, saves empty dict
            if not dicts:
                self.iter_data_cache[pos_key] = {}
            else:
                fast_dict = {row["datetime"]: row for row in dicts}
                self.iter_data_cache[pos_key] = fast_dict

        # If there's an trade during this step_datetime then returns line's dict, else None
        return self.iter_data_cache[pos_key].get(step_datetime)

    # Used to pull real data from parquet from selected source
    def _populate_sim_data(self, key, i, start_idx=0, side=None, data_type: Literal["aggr", "parset", "wf", "aggr_dynamic"]="aggr_dynamic", psid_or_wfid=None):
        """
        Recupera dados de PnL ou resultados brutos (parsets).
        
        Args:
            key (tuple): Chave da hierarquia (op,), (op, m), (op, m, s) ou (op, m, s, a).
            i (int): Índice final da timeline.
            start_idx (int): Índice inicial da busca (default 0).
            side (str|list): "both", "long", "long".
            data_type (str): "aggr" (memória) ou "parset" (disco/parquet).
            ps_id (str/int): ID específico da posição para filtragem.
        """
        
        # --- CASO 1: DADOS AGREGADOS (Rápido, em Memória) ---
        if data_type == "aggr":
            node = self.sim_data.get(key)
            if not node: return None

            def slice_data(block):
                data_slice = block["data"][start_idx : i + 1]
                cols = block["cols"]
                return {col: data_slice[:, idx].tolist() for idx, col in enumerate(cols)}
            
            if isinstance(side, str):
                data_block = node.get(side.lower())
                return slice_data(data_block) if data_block else None

            target_sides = [s.lower() for s in side] if isinstance(side, list) else ["both", "long", "short"]
            payload = {}
            for s in target_sides:
                data_block = node.get(s)
                if data_block:
                    payload[s] = slice_data(data_block)
            
            return payload if payload else None

        # --- CASO 2: DADOS DE PARSET (Leitura de Disco/Storage) ---
        elif data_type == "parset":
            try:
                asset_data = self.storage.load(key)
                raw_df = asset_data.get("timeline")
                if raw_df is None or raw_df.is_empty(): return None

                if psid_or_wfid is not None:
                    raw_df = raw_df.filter(pl.col("ps_id") == psid_or_wfid)

                end_dt = self.datetime_timeline[i]
                if start_idx is not None:
                    start_dt = self.datetime_timeline[start_idx]
                    raw_df = raw_df.filter((pl.col("datetime") >= start_dt) & (pl.col("datetime") <= end_dt))
                else:
                    raw_df = raw_df.filter(pl.col("datetime") == end_dt)

                return raw_df.to_dicts()
            except Exception as e:
                print(f"Erro ao carregar parset para {key}: {e}")
                return None
            
        elif data_type == "wf":
            try:
                start_dt_val = None if start_idx is None else (start_idx if isinstance(start_idx, str) else self.datetime_timeline[start_idx])
                end_dt_val = None if i is None else (i if isinstance(i, str) else self.datetime_timeline[i])

                wfm_df = self.storage.load_walkforward_matrix_v2(
                    key=key, res_price="perc", side_val=side, wf_ids=psid_or_wfid, 
                    timeline_df=None, wf_map=None, start_dt=start_dt_val, end_dt=end_dt_val
                )
                if wfm_df is None or wfm_df.is_empty(): return None
                return wfm_df.to_dicts()
            except Exception as e:
                print(f" < [Portfolio._populate_sim_data] error constructing Walkforward Matrix for {key}: {e}")
                return None
            
        elif data_type == "aggr_dynamic":
            node = self.sim_data.get(key)
            if not node: return None

            target_sides = [side.lower()] if isinstance(side, str) else (side if isinstance(side, list) else ["both", "long", "short"])
            
            # Captura de forma resiliente o estado atual da hierarquia viva do Manager
            h_viva = getattr(self, "hierarchy", {}) or getattr(getattr(self, "portfolio", None), "hierarchy", {})

            # Função auxiliar interna para varrer a árvore viva e validar herança de filtros/pesos
            def _check_hierarchy_status(col_name, current_side):
                parts = col_name.split(":")
                if len(parts) < 4: return True, 1.0
                op, m, s, a = parts[0], parts[1], parts[2], parts[3]

                active = True
                weight = 1.0

                # Varredura defensiva multinível (suporta dicionários aninhados ou chaves diretas)
                if h_viva and isinstance(h_viva, dict):
                    # Level 1: Operação
                    op_node = h_viva.get(op)
                    if isinstance(op_node, dict):
                        if not op_node.get("active", True): active = False
                        weight *= op_node.get("weight", 1.0)

                        # Level 2: Modelo
                        m_node = op_node.get("models", {}).get(m) or op_node.get(m)
                        if isinstance(m_node, dict):
                            if not m_node.get("active", True): active = False
                            weight *= m_node.get("weight", 1.0)

                            # Level 3: Estratégia
                            s_node = m_node.get("strats", {}).get(s) or m_node.get(s)
                            if isinstance(s_node, dict):
                                if not s_node.get("active", True): active = False
                                weight *= s_node.get("weight", 1.0)

                                # Level 4: Ativo
                                a_node = s_node.get("assets", {}).get(a) or s_node.get(a)
                                if isinstance(a_node, dict):
                                    if not a_node.get("active", True): active = False
                                    weight *= a_node.get("weight", 1.0)
                                    
                                    # Valida se o lado específico (long/short) possui travas locais
                                    if current_side in a_node and isinstance(a_node[current_side], dict):
                                        if not a_node[current_side].get("active", True): active = False
                                        weight *= a_node[current_side].get("weight", 1.0)

                return active, weight

            def slice_dynamic_block(block, current_side):
                cols = block["cols"]
                data_matrix = block["data"][start_idx : i + 1] # Shape: (tamanho_slice, numero_colunas)
                
                active_series = []
                weights = []

                # Filtra e aplica pesos apenas nas colunas que passarem na validação viva
                for idx, col in enumerate(cols):
                    is_active, w = _check_hierarchy_status(col, current_side)
                    if is_active:
                        active_series.append(data_matrix[:, idx])
                        weights.append(w)

                # Fallback caso tudo no ramo tenha sido desligado pelo System Manager
                if not active_series:
                    return {"@total": [0.0] * (i + 1 - start_idx)}

                # Combinação inteligente das séries em vetor NumPy acelerado
                if all(w == 1.0 for w in weights):
                    # Se todos os pesos forem 1.0, mantém a média horizontal padrão (Equal Weight)
                    combined = np.mean(active_series, axis=0)
                else:
                    # Se houver pesos customizados (Ex: Risk Parity ou Performance Sizing do MM), faz a soma ponderada
                    combined = np.zeros(data_matrix.shape[0])
                    for series, w in zip(active_series, weights):
                        combined += series * w

                return {"@total": combined.tolist()}

            # Processamento do Payload de retorno
            if isinstance(side, str):
                data_block = node.get(side.lower())
                return slice_dynamic_block(data_block, side.lower()) if data_block else None

            payload = {}
            for s in target_sides:
                data_block = node.get(s)
                if data_block:
                    payload[s] = slice_dynamic_block(data_block, s)
            
            return payload if payload else None

        print(f"    < [Portfolio._populate_sim_data] data_type unknown")
        return None

    # Loads each results data, maps path and generates aggregated results, then clears memory one by one 
    def _load_selected_saved_returns_data(self): 
        storage = self.storage #Storage(base_path=self.data_storage_base_path)
        self.sim_data = {}

        # Acumuladores hierárquicos: { key: { direction: { child_name: series } } }
        flat_granular_data = {"both": {}, "long": {}, "short": {}}
        raw_collected = [] # Tuple list (a_key, dir_label, col_name, df_pnl)
        unique_dts = set()

        REF_CAPITAL = self.portfolio_parameters.get("capital", 100000.0)

        # 1. Data collection
        for op_n, _, m_n, _, s_n, _, a_n, _ in self._iter_portfolio_data():
            config = self.portfolio_data[op_n][m_n][s_n][a_n]
            a_key = (op_n, m_n, s_n, a_n)

            # Extracts configs
            side_pref = config.get("side", "both").lower() if isinstance(config, dict) else "both"
            separate_ls = config.get("analise_long_short_separate", False) if isinstance(config, dict) else False
            calculate_on_data = config.get("calculate_on_data", "all") if isinstance(config, dict) else "all"

            # Loads brute data (Parset)
            asset_data = storage.load(a_key)
            timeline_df = asset_data.get("timeline")
            
            # Preparates direction
            vias = {"both": side_pref}
            if separate_ls:
                vias.update({"long": "long", "short": "short"})

            for dir_label, side_val in vias.items():
                # Source A - Param Sets
                if calculate_on_data in ["all", "parset"] and timeline_df is not None and not timeline_df.is_empty():
                    ps_ids = timeline_df["ps_id"].unique().to_list() if "ps_id" in timeline_df.columns else ["default"]

                    for pid in ps_ids:
                        df_pid = timeline_df.filter(pl.col("ps_id") == pid) if pid != "default" else timeline_df
                        p_aggr = self.get_aggr_pnl_by_side(df_pid, side_val, a_n, metric_col="perc")
                        
                        if p_aggr is not None and not p_aggr.is_empty():
                            val_col = [c for c in p_aggr.columns if c != "datetime"][0]
                            normalized_parset = p_aggr.with_columns(
                                (pl.col(val_col) * REF_CAPITAL).alias("pnl")
                            ).select(["datetime", "pnl"])
                            
                            # Complete nomeclature alignment preserving setup granularity
                            col_name = f"{op_n}:{m_n}:{s_n}:{a_n}:parset:{pid}"
                            raw_collected.append((a_key, dir_label, col_name, normalized_parset))
                            unique_dts.update(normalized_parset['datetime'].to_list())
                
                # Source B - Walkforward
                if calculate_on_data in ["all", "wf"]:
                    wfm_wide = storage.load_walkforward_matrix_v2(a_key, side_val=side_val, res_price="perc")
                    if wfm_wide is not None and not wfm_wide.is_empty():
                        wf_cols = [c for c in wfm_wide.columns if c != "datetime"]
                        
                        for wfid in wf_cols:
                            wf_series_df = wfm_wide.select([
                                pl.col("datetime"),
                                (pl.col(wfid) * REF_CAPITAL).alias("pnl")
                            ])
                            col_name = f"{op_n}:{m_n}:{s_n}:{a_n}:wf:{wfid}"
                            raw_collected.append((a_key, dir_label, col_name, wf_series_df))
                            unique_dts.update(wf_series_df['datetime'].to_list())

        # If no data is collected then ends safely
        if not unique_dts:
            print(" < [Portfolio._load_selected_saved_returns_data] Error: No data available to load.")
            return False
        
        # 2. Main timeline build
        self.datetime_timeline = sorted(unique_dts)
        timeline_global = pl.DataFrame({"datetime": self.datetime_timeline})
        zeros_series = pl.Series("pnl", values=[0.0] * len(self.datetime_timeline))
        all_directions = ["both", "long", "short"]

        # Metadata init
        for a_key, _, _, _ in raw_collected:
            if a_key not in self.sim_data:
                base_path = storage._asset_path(*a_key)
                self.sim_data[a_key] = {
                    "type": "disk",
                    "trades_path": str(base_path / "trades" / "trades.parquet"),
                }

        # Horizontal alignment for all collected time series
        for a_key, dir_label, col_name, df_pnl in raw_collected:
            aligned = timeline_global.join(df_pnl, on="datetime", how="left").fill_null(0.0)
            flat_granular_data[dir_label][col_name] = aligned.get_column("pnl")

        # 3. Recursive key mapping for hierarchical levels
        # Collects all structure keys possible present in portfolio config
        all_hierarchical_keys = set()
        all_hierarchical_keys.add((self.name,)) # Global portfolio level

        for op_n, op_obj in self.portfolio_data.items():
            all_hierarchical_keys.add((op_n,)) # Op level
            for m_n, m_obj in op_obj.items():
                all_hierarchical_keys.add((op_n, m_n)) # Model level
                for s_n, s_obj in m_obj.items():
                    all_hierarchical_keys.add((op_n, m_n, s_n)) # Strat level
                    for a_n in s_obj.keys():
                        all_hierarchical_keys.add((op_n, m_n, s_n, a_n)) # Asset level

        # Populates sim_data associating every key with own granular column
        for key in all_hierarchical_keys:
            if key not in self.sim_data:
                self.sim_data[key] = {"type": "aggr"}

            for d_name in all_directions:
                dir_cols = flat_granular_data[d_name]
                matched_series = {}

                # Filters series based on size and fit of prefix tuple-key
                if len(key) == 1:
                    if key[0] == self.name:
                        matched_series = dir_cols # Global portfolio inherits all
                    else:
                        prefix = f"{key[0]}:"
                        matched_series = {k: v for k, v in dir_cols.items() if k.startswith(prefix)}
                elif len(key) == 2:
                    prefix = f"{key[0]}:{key[1]}:"
                    matched_series = {k: v for k, v in dir_cols.items() if k.startswith(prefix)}
                elif len(key) == 3:
                    prefix = f"{key[0]}:{key[1]}:{key[2]}:"
                    matched_series = {k: v for k, v in dir_cols.items() if k.startswith(prefix)}
                elif len(key) == 4:
                    prefix = f"{key[0]}:{key[1]}:{key[2]}:{key[3]}:"
                    matched_series = {k: v for k, v in dir_cols.items() if k.startswith(prefix)}

                # Transformas filtered series group into a bidimensional numpy clean matrix
                if matched_series:
                    wide_df = pl.DataFrame(matched_series)
                    self.sim_data[key][d_name] = {
                        "data": wide_df.to_numpy(),
                        "cols": wide_df.columns
                    }
                else: # Safe fallback to an specific side in case hasen't valid data
                    self.sim_data[key][d_name] = {
                        "data": zeros_series.to_numpy().reshape(-1, 1),
                        "cols": ["zeros"]
                    }

        return True

    # Gets ps_id with wf_id best_param [>= initial IS optmization param and <= current datetime]
    def _get_psid_with_wfid(self, key, step_datetime, wf_map, wfid):
        
        if wf_map is None or wf_map.is_empty() or wfid is None:
            return None
        
        pos_key = (*key, wfid)

        # Creates {datetime: ps_id} map only once in memory
        if not hasattr(self, '_wf_map_cache'): self._wf_map_cache = {}
        
        if pos_key not in self._wf_map_cache:
            # Garante o cast correto da data
            if wf_map.schema["datetime"] not in [pl.Datetime, pl.Date]:
                wf_map = wf_map.with_columns([
                    pl.coalesce([
                        pl.col("datetime").cast(pl.Utf8).str.to_datetime("%Y%m%d%H%M%S", strict=False),
                        pl.col("datetime").cast(pl.Int64).cast(pl.Datetime("us"))
                    ])
                ])

            wf_map_filtered = (
                wf_map.filter(pl.col("wf_id") == wfid)
                .drop_nulls("datetime")
                .sort("datetime")
            )

            dates = wf_map_filtered["datetime"].to_list()
            params = wf_map_filtered["best_param"].to_list()

            self._wf_map_cache[pos_key] = (dates, params)

        # Gets cache lists
        dates, params = self._wf_map_cache[pos_key]
        if not dates: 
            return None
        
        # Binary search to find date <= step_datetime
        idx = bisect.bisect_right(dates, step_datetime) - 1
        
        # BLINDAGEM DO PERÍODO IN-SAMPLE:
        if idx < 0:
            # O step_datetime atual é ANTES da primeira atualização (In-Sample / Warm-up).
            # Retorna None para impedir qualquer operação neste ativo com este WF.
            return None
        
        return params[idx]

    def _init_hierarchy(self):
        models_found = {(op, m) for op, _, m, *_ in self._iter_portfolio_data()}
        n_models = len(models_found)
        hcy = {}

        for op_name, _, m_name, _, s_name, _, a_name, _ in self._iter_portfolio_data():
            m_key = (op_name, m_name)
            s_key = (op_name, m_name, s_name)
            a_key = (op_name, m_name, s_name, a_name)

            # Model Level
            if m_key not in hcy:
                model_config = self.sm_mm_map.get("models", {}).get(m_name, {})
                
                hcy[m_key] = {
                    "active":   True,
                    "side": model_config.get("managers", {}).get("side", "both"),
                    "separate_ls": model_config.get("managers", {}).get("separate_ls", False),
                    "both":  {"score": 0.0, "exposure": 0.0, "capital": 0.0, "weight": 1.0 / n_models},
                    "long":  {"score": 0.0, "exposure": 0.0, "capital": 0.0, "weight": (1.0 / n_models) /2},
                    "short": {"score": 0.0, "exposure": 0.0, "capital": 0.0, "weight": (1.0 / n_models) /2},
                    "strats":   {}
                }

            # Strat level
            if s_key not in hcy[m_key]["strats"]:
                strat_config = self.sm_mm_map.get("strats", {}).get(s_name, {})
                
                hcy[m_key]["strats"][s_key] = {
                    "active":      True,
                    "side":        strat_config.get("managers", {}).get("side", "both"),
                    "separate_ls": strat_config.get("managers", {}).get("separate_ls", False),
                    "trade_update_can_enter": strat_config.get("managers", {}).get("trade_update_can_enter", False),
                    "both":        {"score": 0.0, "exposure": 0.0, "capital": 0.0, "weight": 1.0},
                    "long":        {"score": 0.0, "exposure": 0.0, "capital": 0.0, "weight": 0.0},
                    "short":       {"score": 0.0, "exposure": 0.0, "capital": 0.0, "weight": 0.0},
                    "assets":      {}
                }

            # Asset level
            if a_key not in hcy[m_key]["strats"][s_key]["assets"]:
                config = self.portfolio_data[op_name][m_name][s_name][a_name]

                hcy[m_key]["strats"][s_key]["assets"][a_key] = {
                    "active":      True,
                    "side":        config.get("side", "both") if isinstance(config, dict) else "both",
                    "separate_ls": config.get("analise_long_short_separate", False) if isinstance(config, dict) else False,
                    "both":        {"score": 0.0, "exposure": 0.0, "capital": 0.0, "weight": 1.0},
                    "long":        {"score": 0.0, "exposure": 0.0, "capital": 0.0, "weight": 0.0},
                    "short":       {"score": 0.0, "exposure": 0.0, "capital": 0.0, "weight": 0.0},
                    "wf_id":       {"12_12_12": True}, # NOTE REMOVER, DEVE SER INPUT USUARIO 12_12_12=True
                    "ps_id":       {}  # param1-50_param2-2.0=False
                }

        self.hierarchy = hcy
        return True

    def _pre_compute_and_calc_rebalance_schedule(self, global_assets, sm_mm_map):
        psm_sch, msm_sch, ssm_sch = {}, {}, {}
        params_pool = {}

        DEFAULT_MGR_CONFIG = {
            "psm": (PortfolioSystemManager, PortfolioSystemManagerParams),
            "msm": (ModelSystemManager, ModelSystemManagerParams),
            "ssm": (StratSystemManager, StratSystemManagerParams),
        }
        
        timeline = self.datetime_timeline
        last_idx = len(timeline) - 1

        # 1. Portfolio Level
        p_name = self.name
        p_key = (p_name,)
        p_magrs = sm_mm_map.get("managers", {})

        # Searches data via _populate_sim_data
        p_data = self._populate_sim_data(key=p_key, i=last_idx, start_idx=0, data_type="aggr")
        
        if p_data:
            p_node = {p_key: p_data}
           
            # NOTE MUDAR PSM and PMM
            for mgr_key, mgr_class, sch_dict in [
                ("psm", DEFAULT_MGR_CONFIG["psm"], psm_sch)
            ]: 
                mgr = p_magrs.get(mgr_key) or mgr_class() 
                mgr.set_portfolio(self)
                self.indicator_pool, _ = mgr.pre_compute(
                    global_assets, timeline, p_node, self.indicator_pool, p_key)
                sch_dict[p_name] = mgr.get_schedule(timeline)
                p_magrs[mgr_key] = mgr

        # Models
        for op_name, op_models in self.portfolio_data.items():
            for m_name, m_strats in op_models.items():
                m_key = (op_name, m_name)
                m_info = sm_mm_map.get("models", {}).get(m_name, {})
                m_magrs = m_info.get("managers", {})

                # Seaches model data
                m_data = self._populate_sim_data(key=m_key, i=last_idx, start_idx=0, data_type="aggr")

                if m_data:
                    m_node = {m_key: m_data}
                    for mgr_key, (mgr_class, params_class), sch_dict in [
                        ("msm", DEFAULT_MGR_CONFIG["msm"], msm_sch)
                    ]:
                        mgr = m_magrs.get(mgr_key) or mgr_class()
                        mgr.set_portfolio(self)
                        self.indicator_pool, _ = mgr.pre_compute(
                            global_assets, timeline, m_node, self.indicator_pool, m_key)
                        sch_dict[m_key] = mgr.get_schedule(timeline)
                        m_magrs[mgr_key] = mgr

                # Strats
                for s_name in m_strats.keys():
                    s_key = (op_name, m_name, s_name)
                    s_info = m_info.get("strats", {}).get(s_name, {})
                    s_magrs = s_info.get("managers", {})

                    # Searches Strat data
                    s_data = self._populate_sim_data(key=s_key, i=last_idx, start_idx=0, data_type="aggr")
                    
                    if s_data:
                        s_node = {s_key: s_data}
                        for mgr_key, mgr_class, sch_dict in [
                            ("ssm", DEFAULT_MGR_CONFIG["ssm"], ssm_sch)
                        ]: 
                            mgr = s_magrs.get(mgr_key) or mgr_class()
                            mgr.set_portfolio(self)
                            self.indicator_pool, _ = mgr.pre_compute(
                                global_assets, timeline, s_node, self.indicator_pool, s_key)
                            sch_dict[s_key] = mgr.get_schedule(timeline)
                            s_magrs[mgr_key] = mgr

        return params_pool, psm_sch, msm_sch, ssm_sch

    # ── Datetime timeline mapping ───────────────────────────────────────────────

    # PEGAR NOS INDICADORES PORQUE SM_ASSETS SÓ SERVE PRA INDICADORES E TEM TF DEFINIDO JÁ TMB
    def _get_all_sm_ind_datetimes(self, data_source="local"):
        assets = self.global_assets #Asset.load_all() # NOTE Deletar futuramente
        unique_ind_dts = set()
        repeated_assets = set()

        psm = self.sm_mm_map.get("managers", {}).get("psm")
        sm_inds = psm.indicators if (psm and psm.indicators) else {}
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
                    assets = psm.assets if psm and psm.assets else []
                    for asset_name in assets:
                        if asset_name not in repeated_assets:
                            asset_obj = self.global_assets.get(asset_name)
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

    def iter_hierarchy(self, target_level="assets", active_only=False):
        # target_level (str): To which level should it iterate to
        # active_only (bool): If True, ignores where "active" == False
        models_dict = self.hierarchy.get("models", self.hierarchy)

        # Models
        for m_key, m_info in models_dict.items():
            if active_only and not m_info.get("active", True):
                continue
            if target_level == "models":
                yield m_key, m_info
                continue

            # Strats
            strats_dict = m_info.get("strats", {})
            for s_key, s_info in strats_dict.items():
                if active_only and not s_info.get("active", True):
                    continue
                if target_level == "strats":
                    yield m_key, m_info, s_key, s_info
                    continue

                # Assets
                assets_dict = s_info.get("assets", {})
                for a_key, a_info in assets_dict.items():
                    if active_only and not a_info.get("active", True):
                        continue
                    if target_level == "assets":
                        yield m_key, m_info, s_key, s_info, a_key, a_info

    def get_metadata_by_key(self, key: tuple, meta_path: str = "Backend/results/operation_test/operation_meta.json"):
        """
        Navega no JSON de metadados usando a hierarquia da key.
        Exemplos de key:
        - ("operation_test",) -> Retorna dados da operação
        - ("operation_test", "MA Trend Following") -> Retorna dados do modelo
        - ("operation_test", "MA Trend Following", "AT15") -> Retorna dados da estratégia
        """
        if not os.path.exists(meta_path):
            print(f"Erro: Metadado não encontrado em {meta_path}")
            return None

        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        try:
            # Nível 0: Operação (op_name)
            # O JSON fornecido é a própria operação
            current_node = data
            
            # Nível 1: Modelo
            if len(key) > 1:
                model_name = key[1]
                current_node = current_node.get("models", {}).get(model_name)
            
            # Nível 2: Estratégia
            if len(key) > 2 and current_node:
                strat_name = key[2]
                current_node = current_node.get("strats", {}).get(strat_name)
                
            return current_node

        except Exception as e:
            print(f"Erro ao ler metadados para a chave {key}: {e}")
            return None

    def get_assets_from_model_meta(self, model_meta):
        """Extrai todos os ativos únicos de todas as estratégias de um modelo"""
        assets = set()
        if not model_meta or "strats" not in model_meta:
            return []
        
        for strat in model_meta["strats"].values():
            assets.update(strat.get("assets", []))
        return list(assets)

    # ── Portfolio Optimization ───────────────────────────────────────────────

    def _portfolio_optimization(self):
        # Iterates over previous results and identifies each combination for OS while running
        return True

    # ──────────────────────────────────────────────────────────────────────────── 

    def debug_plot_portfolio_performance(self, portfolio_returns, initial_capital=100000.0):
        import polars as pl
        import matplotlib.pyplot as plt
        """
        Gera um report visual do backtest dividindo a tela em:
        1. Curva de Capital Global do Portfólio (Equity Curve)
        2. Curvas de PnL de todas as Estratégias juntas para comparação
        """
        if not portfolio_returns:
            print("[-] Erro no Debug: portfolio_returns está vazio.")
            return

        # 1. Converte o log flat para um DataFrame do Polars
        df = pl.DataFrame(portfolio_returns)

        # Extrai uma chave única para a estratégia combinando Model Name e Strat Name
        # c_key formato: (op_name, m_name, s_name, a_name, ps_id)
        df = df.with_columns([
            pl.col("c_key").map_elements(lambda x: f"{x[1]} | {x[2]}", return_dtype=pl.String).alias("strat")
        ])

        # Cria uma timeline mestre contendo todos os datetimes e estratégias do sistema
        all_dts = df["datetime"].unique().sort()
        all_strats = df["strat"].unique()
        grid = pl.DataFrame({"datetime": all_dts}).join(pl.DataFrame({"strat": all_strats}), how="cross")

        # --- PROCESSA LUCRO REALIZADO (EXITS) ---
        df_exits = (
            df.filter(pl.col("event") == "exit")
            .group_by(["datetime", "strat"])
            .agg(pl.col("pnl").sum().alias("realized_pnl_instant"))
        )

        # --- PROCESSA LUCRO FLUTUANTE (UPDATES) ---
        df_updates = (
            df.filter(pl.col("event") == "update")
            .group_by(["datetime", "strat"])
            .agg(pl.col("pnl").sum().alias("floating_pnl"))
        )

        # --- RECONSTRUÇÃO DA MATRIZ DO BACKTEST ---
        df_curves = (
            grid
            .join(df_exits, on=["datetime", "strat"], how="left")
            .join(df_updates, on=["datetime", "strat"], how="left")
            .with_columns([
                pl.col("realized_pnl_instant").fill_null(0.0),
                pl.col("floating_pnl").fill_null(0.0)
            ])
            .sort(["strat", "datetime"]) # Garante a ordem cronológica por grupo
        )

        # Calcula o PnL acumulado fechado de cada estratégia ao longo do tempo
        df_curves = df_curves.with_columns(
            pl.col("realized_pnl_instant").cum_sum().over("strat").alias("cum_realized_pnl")
        )

        # PnL Total de uma estratégia = Realizado acumulado até a barra + Flutuante aberto na barra
        df_curves = df_curves.with_columns(
            (pl.col("cum_realized_pnl") + pl.col("floating_pnl")).alias("total_strat_pnl")
        )

        # --- AGREGAÇÃO DO PORTFÓLIO GLOBAL ---
        # Soma o PnL de todas as estratégias vigentes em cada ponto do tempo
        df_portfolio = (
            df_curves.group_by("datetime")
            .agg(pl.col("total_strat_pnl").sum().alias("portfolio_pnl"))
            .sort("datetime")
            .with_columns(
                (pl.col("portfolio_pnl") + initial_capital).alias("portfolio_equity")
            )
        )

        # --- CONSTRUÇÃO DOS GRÁFICOS (MATPLOTLIB) ---
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Renderiza os eixos x com os valores reais (seja string de datetime ou inteiros)
        x_axis_values = df_portfolio["datetime"].to_list()

        # GRÁFICO 1: Curva de Capital Total do Portfólio
        axes[0].plot(
            x_axis_values, 
            df_portfolio["portfolio_equity"].to_list(), 
            color="#2ca02c", 
            linewidth=2.5, 
            label="Patrimônio Líquido (Equity)"
        )
        axes[0].axhline(initial_capital, color="red", linestyle=":", alpha=0.7, label="Capital Inicial")
        axes[0].set_title("Curva de Capital Global do Portfólio (Total Equity)", fontsize=14, fontweight="bold", pad=10)
        axes[0].set_ylabel("Capital Disponível ($)", fontsize=12)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="upper left")

        # GRÁFICO 2: Todas as Estratégias Separadas na Mesma Aba
        for strat_name in all_strats.to_list():
            strat_data = df_curves.filter(pl.col("strat") == strat_name).sort("datetime")
            axes[1].plot(
                strat_data["datetime"].to_list(),
                strat_data["total_strat_pnl"].to_list(),
                label=strat_name,
                alpha=0.85,
                linewidth=1.5
            )

        axes[1].axhline(0, color="black", linestyle="-", alpha=0.3)
        axes[1].set_title("Evolução de Performance por Estratégia (PnL Individual)", fontsize=14, fontweight="bold", pad=10)
        axes[1].set_xlabel("Linha do Tempo (Datetime / Index)", fontsize=12)
        axes[1].set_ylabel("PnL Acumulado ($)", fontsize=12)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
        # Posiciona a legenda para fora do gráfico para não cobrir as linhas caso existam muitas sub-estratégias
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

        plt.tight_layout()
        plt.show()

    # TEMP - DELETAR
    def debug_model_aggregations(self):
        """
        Imprime e plota o PnL Acumulado dos Modelos separados por BOTH, LONG e SHORT.
        Deve ser chamado após self._load_selected_saved_returns_data()
        """
        import matplotlib.pyplot as plt
        import polars as pl

        print("\n" + "═"*60)
        print(" 🔍 DEBUG: AGREGAÇÃO HIERÁRQUICA DOS MODELOS (100k BASE)")
        print("═"*60)

        # 1. Filtra apenas as chaves referentes a Modelos no sim_data
        # Modelos são guardados com chaves de tupla tamanho 2: (op_name, model_name)
        model_keys = [k for k in self.sim_data.keys() if isinstance(k, tuple) and len(k) == 2]

        if not model_keys:
            print("      > Nenhum modelo agregado encontrado no sim_data.")
            return

        # 2. Configura o visual do plot (Mesmo estilo escuro original)
        plt.style.use('dark_background')
        fig, axes = plt.subplots(len(model_keys), 1, figsize=(14, 5 * len(model_keys)), squeeze=False)
        fig.patch.set_facecolor('#0a0a0a')

        colors = {"both": "#4169E1", "long": "#00FA9A", "short": "#FF6347"} # Azul, Verde, Vermelho

        for i, m_key in enumerate(model_keys):
            op_name, model_name = m_key
            print(f"\n 📈 Modelo Identificado: [{op_name}] -> {model_name}")
            
            ax = axes[i, 0]
            ax.set_facecolor('#0a0a0a')
            ax.set_title(f"Model: {model_name} | Cumulative Normal PnL", color='white', loc='left', pad=15)
            
            # Vamos iterar exatamente na ordem que você pediu: both, long, short
            for d_name in ["both", "long", "short"]:
                if d_name in self.sim_data[m_key]:
                    # Resgata a matriz bruta guardada no sim_data
                    data_dict = self.sim_data[m_key][d_name]
                    cols = data_dict["cols"]
                    
                    # Reconstrói temporariamente para extrair a coluna do modelo
                    df = pl.DataFrame(data_dict["data"], schema=cols, orient="row")
                    
                    # O nome da coluna agregada do modelo final é o próprio m_key[1]
                    if "@total" in df.columns:
                        # Extrai a série e calcula a curva de capital acumulada
                        pnl_series = df.get_column("@total")
                        cum_pnl = pnl_series.cum_sum().to_list()
                        
                        total_pnl = cum_pnl[-1] if cum_pnl else 0.0
                        total_bars = len(cum_pnl)
                        
                        # Print no console
                        print(f"      > {d_name.upper():<5} | Barras: {total_bars:<5} | PnL Acumulado: $ {total_pnl:,.2f}")
                        
                        # Adiciona ao Plot
                        c = colors.get(d_name, "white")
                        alpha = 1.0 if d_name == "both" else 0.7 # Deixa o Both mais sólido
                        ax.plot(cum_pnl, label=f"{d_name.upper()} (Total: ${total_pnl:,.0f})", color=c, lw=2, alpha=alpha)
                    else:
                        print(f"      > {d_name.upper():<5} | [Erro] Coluna @total não encontrada.")
                else:
                    print(f"      > {d_name.upper():<5} | Sem dados disponíveis para esta via.")
            
            # Estilização do eixo e legenda
            ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.legend(frameon=False, fontsize=10, loc='upper left', ncol=3)
            ax.grid(True, linestyle=':', alpha=0.15)
            ax.tick_params(colors='#888888')

        plt.tight_layout()
        plt.show()

    def debug_print_portfolio_summary(self, portfolio_returns):
        import polars as pl
        """
        Imprime um sumário resumido e limpo no terminal agrupado por Modelo -> Estratégia,
        mostrando estatísticas de trades fechados e posições abertas.
        """
        if not portfolio_returns:
            print("\n[-] Erro no Debug: portfolio_returns está vazio.")
            return

        # 1. Converte para DataFrame do Polars
        df = pl.DataFrame(portfolio_returns)

        # 2. Extrai identificadores amigáveis a partir da tupla c_key
        # c_key: (op_name, m_name, s_name, a_name, ps_id)
        df = df.with_columns([
            pl.col("c_key").map_elements(lambda x: f"{x[1]} -> {x[2]}", return_dtype=pl.String).alias("model_strat"),
            pl.col("c_key").map_elements(lambda x: x[3], return_dtype=pl.String).alias("asset"),
            pl.col("c_key").map_elements(lambda x: x[4], return_dtype=pl.String).alias("trade_id")
        ])

        print("\n" + "=" * 70)
        print("         RELATÓRIO CONSOLIDADO DE PERFORMANCE (MÉTRICAS OPERACIONAIS)       ")
        print("=" * 70)

        # Filtra apenas eventos de encerramento para calcular métricas reais de trades finalizados
        df_exits = df.filter(pl.col("event") == "exit")

        if df_exits.height == 0:
            print("[!] Nenhum trade foi fechado ('exit') durante o período do backtest ainda.")
        else:
            # Agrupa por estratégia e calcula estatísticas clássicas
            summary = df_exits.group_by("model_strat").agg([
                pl.len().alias("total_trades"),
                pl.col("pnl").sum().alias("pnl_total"),
                pl.col("pnl").mean().alias("pnl_medio"),
                
                # CORREÇÃO: Aplica o filtro na coluna 'pnl' e conta as ocorrências
                pl.col("pnl").filter(pl.col("pnl") > 0).count().alias("ganhos"),
                pl.col("pnl").filter(pl.col("pnl") < 0).count().alias("perdas"),
                
                pl.col("pnl").max().alias("maior_lucro"),
                pl.col("pnl").min().alias("maior_prejuizo")
            ]).with_columns(
                # Calcula a Taxa de Acerto (Win Rate)
                (pl.col("ganhos") / (pl.col("ganhos") + pl.col("perdas")) * 100).round(2).alias("win_rate")
            ).sort("pnl_total", descending=True)

            # Imprime o sumário de cada estratégia
            for row in summary.iter_rows(named=True):
                print(f"\n[ESTRATÉGIA]: {row['model_strat']}")
                print(f"  ├─ Total de Trades Fechados: {row['total_trades']}")
                print(f"  ├─ Taxa de Acerto (Win Rate): {row['win_rate']}%  ({row['ganhos']} Vitórias / {row['perdas']} Derrotas)")
                print(f"  ├─ Lucro/Prejuízo Total:     ${row['pnl_total']:.2f}")
                print(f"  ├─ Retorno Médio por Trade:  ${row['pnl_medio']:.2f}")
                print(f"  └─ Extremos (Max Win/Loss):  +${row['maior_lucro']:.2f} / -${abs(row['maior_prejuizo']):.2f}")

        print("\n" + "=" * 70)
        print("                    POSIÇÕES QUE TERMINARAM ABERTAS                   ")
        print("=" * 70)

        # Descobre quais posições terminaram abertas pegando o ÚLTIMO estado cronológico de cada c_key única
        df_latest_state = df.sort("datetime").group_by("c_key").last()
        df_open_positions = df_latest_state.filter(pl.col("event") == "update")

        if df_open_positions.height == 0:
            print("  [•] Nenhuma posição aberta pendente no final do backtest.")
        else:
            # Recria as colunas amigáveis para o print das abertas
            df_open_positions = df_open_positions.with_columns([
                pl.col("c_key").map_elements(lambda x: f"{x[1]} -> {x[2]}", return_dtype=pl.String).alias("model_strat"),
                pl.col("c_key").map_elements(lambda x: x[3], return_dtype=pl.String).alias("asset")
            ]).sort("model_strat")

            for row in df_open_positions.iter_rows(named=True):
                print(f"  • [{row['model_strat']}] Ativo: {row['asset']} "
                    f"| Lote Operado: {row['lot_size']} "
                    f"| Margem Retida: ${row['allocated_margin']:.2f} "
                    f"| PnL Flutuante Atual: ${row['pnl']:.2f}")

        print("=" * 70 + "\n")

    def _run(self):
        # Data Init - Loads data, saves unique datetimes and generates aggr results
        print("     > Populating Portfolio Data from Database")
        self._load_selected_saved_returns_data()
        #self.debug_model_aggregations()
        self._init_hierarchy()

        # Runs Portfolio Simulation
        print("     > Executing Portfolio Simulation")
        self._simulation()

        self.debug_print_portfolio_summary(self.portfolio_returns)
        self.debug_plot_portfolio_performance(self.portfolio_returns)
            
        return True



# XXX Fallback para SM/MM antigo
# XXX Eliminar todos MM, apenas um geral
# XXX SM focam em gerar peso apenas
# XXX SystemManager permanece como está servindo de repositório de func pai para os níveis
# XXX Modificar get_data para poder ao invés de usar o AGGR criar um com lookback atual
# Mover funções de gestão para MM onde vai selecionar entradas concorrentes
# Dynamic WF in SSM and WF that will need to look for ps_ids even if not selected to Portfolio


if __name__ == "__main__":
    from MA import MA # type: ignore
    from VAR import VAR # type: ignore
    from ATR_SL import ATR_SL # type: ignore
    from Volatility import Volatility # type: ignore

    # ── Portfolio level ───────────────────────────────────────────────────────
    assets = Asset.load_all()
    eurusd = assets.get("EURUSD")
    gbpusd = assets.get("GBPUSD")
    usdjpy = assets.get("USDJPY")
    winfut = assets.get("WIN$")
    
    global_assets = {'EURUSD': eurusd, 'GBPUSD': gbpusd, 'USDJPY': usdjpy, 'WIN$': winfut} # Global Assets, loaded when app starts up, has all Asset and Portfolios 
    pmm = MoneyManager(MoneyManagerParams())
    psm = PortfolioSystemManager(PortfolioSystemManagerParams(
        parset_order = "highest",
        parset_metric = "pnl",
        parset_allocation = "1/n",
        parset_number_cutoff = 1,
        parset_sides_overwrite = None,
        params={
            "param1": range(21, 21+1, 1),
        },
        indicators={
            "vol": Volatility(asset="@total_both", timeframe="tick", 
                              window="param1", aggr_days=True, 
                              price_col="pnl", min_periods="param1")
        },
        #assets={'EURUSD'},
    ))

    # ── Model level ───────────────────────────────────────────────────────────

    msm = ModelSystemManager(ModelSystemManagerParams(
        reb_frequency="weekly",
        params={
            "param1": range(21, 21+1, 1),
        },
        indicators={
            "vol": Volatility(asset="@each_both", timeframe="tick", 
                              window="param1", aggr_days=True, 
                              price_col="pnl", min_periods="param1"),
            # "ma": MA(asset="@each_both", timeframe="tick", 
            #          window="param1", aggr_days=True, 
            #          price_col="pnl", min_periods="param1"),
        },
    ))

    # ── Strat level ───────────────────────────────────────────────────────────

    ssm = StratSystemManager(StratSystemManagerParams(
        params={
            "param1": range(21, 21+1, 1),
        },
        indicators={
            "vol": Volatility(asset="@total_both", timeframe="tick", 
                              window="param1", aggr_days=True, 
                              price_col="pnl", min_periods="param1"),
            "ema": MA(asset="EURUSD", timeframe="M15", 
                      window="param1", ma_type="ema", price_col="close"),
        },
    ))
    ssm2 = StratSystemManager(StratSystemManagerParams(
        params={
            "param1": range(50, 200+1, 50),
        },
        indicators={
            "vol": Volatility(asset="@total_both", timeframe="tick", 
                              window="param1", aggr_days=True, 
                              price_col="pnl", min_periods="param1"),
        },
    ))

    # ── portfolio_data com SM/MM em cada nível ────────────────────────────────
    # O portfolio_data carrega os resultados do storage.
    # SM/MM ficam num dict separado mapeado por (model, strat)
    # para não poluir a estrutura de dados de resultados.

    portfolio_data = {
        "operation_test": {
            "FX MA Trend Following": {
                "AT15": {
                    "EURUSD": {
                        "side": "both",
                        "analise_long_short_separate": True, # if side=both, and True, creates separate Aggr
                        "calculate_on_data": "wf",
                    },
                    "GBPUSD": {
                        "side": "both",
                        "analise_long_short_separate": True, 
                        "calculate_on_data": "wf",
                    },
                    "USDJPY": {
                        "side": "both",
                        "analise_long_short_separate": True, 
                        "calculate_on_data": "wf",
                    },
                }
            },
            "FX Mean Reversion": {
                "AT30": {
                    "EURUSD": {
                        "side": "both",
                        "analise_long_short_separate": True,
                        "calculate_on_data": "wf",
                    },
                    "GBPUSD": {
                        "side": "both",
                        "analise_long_short_separate": True, 
                        "calculate_on_data": "wf",
                    },
                    "USDJPY": {
                        "side": "both",
                        "analise_long_short_separate": True, 
                        "calculate_on_data": "wf",
                    },
                }
            },
            "WIN Day Sazonality": {
                "AT20": {
                    "WIN$": {
                        "side": "both",
                        "analise_long_short_separate": True,
                        "calculate_on_data": "wf",
                    }
                },
            },
        }
    }

    # SM/MM mapeados por nível — referenciados durante a simulação
    sm_mm_map = {
        "managers": {"psm": psm, "pmm": pmm, "separate_ls": True, "side": 'both'},
        "models": {
            "FX MA Trend Following": {
                "managers": {"msm": msm, "separate_ls": True, "side": 'both'},
                "strats": {
                    "AT15": {"managers": {"ssm": ssm, "separate_ls": True, "side": 'both', "trade_update_can_enter": False}}
                }
            },
            "FX Mean Reversion": {
                "managers": {"msm": msm, "separate_ls": True, "side": 'both'},
                "strats": {
                    "AT30": {"managers": {"ssm": ssm, "separate_ls": True, "side": 'both', "trade_update_can_enter": False}}
                }
            },
            "WIN Day Sazonality": {
                "managers": {"msm": msm, "separate_ls": True, "side": 'both'},
                "strats": {
                    "AT20": {"managers": {"ssm": ssm2, "separate_ls": True, "side": 'both', "trade_update_can_enter": False}},
                }
            },
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
        global_assets=global_assets, 
    ))

    portfolio._run()



    """
    PCA-Regime-Adjusted Momentum (PCA-RAM)
    EW-PCA (Entropy Weighting-PCA)
    HRP

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

    """ NOTE
    Pontos para melhorar para V2
    - 3 SM para rankear, filtrar e limitar com pesos cada Nível e Asset
    - 1 MM para gerenciar as Strat(s) e Asset(s) dos param_set selecionados
    - Carregar o objeto do Model também, além dos resultados, para ter acesso aos ativos


    """









