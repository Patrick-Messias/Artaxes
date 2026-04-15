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
        self.use_portfolio_asset_data = portfolio_params.use_portfolio_asset_data
        self.global_datetime_prefix = portfolio_params.global_datetime_prefix

        self.datetime_timeline = portfolio_params.datetime_timeline

        self.portfolio_returns: dict={}
        self.sim_data: dict= {}
        self.global_assets = Asset.load_all()
        self.storage = Storage(base_path=portfolio_params.data_storage_base_path)
    

    def _simulation(self):
        # 1 - Init, populating sim_data
        sim_current_equity = self.portfolio_parameters.get("capital", 100000.0)

        active_positions = {} 
        hierarchy = {}
        self.portfolio_returns = {}
        self.indicator_pool = {}

        # Checks if is going to simulate portfolio with strat backtest results or asset positions
        has_pnl = any("pnls" in str(key).lower() for key in self.sim_data.keys())
        has_wf = any("wf_pnls" in str(key).lower() for key in self.sim_data.keys())
        portfolio_simulation_with_backtest_results = (has_pnl or has_wf)
        update_func_to_use = self._update_pos_with_backtest_ret if portfolio_simulation_with_backtest_results else self._update_pos_with_assets_ret

        # SM and MM Pre-Compute Metrics, Indicators and Rebalance Schedule
        params_pool, psm_sch, msm_sch, ssm_sch, pmm_sch, mmm_sch, smm_sch \
        = self._pre_compute_and_calc_rebalance_schedule(self.global_assets, self.sm_mm_map) # NOTE Futuramente salvar os indicadores calculados para SQL/parquet para não pesar memória

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

            # Dentro do seu loop principal, após o i atingir 150
            if i == 30000:
                print(f"\n{'='*20} TESTANDO WALKFORWARD OOS (i={i}) {'='*20}")
                
                # 1. Definição das chaves (Ajuste para um ativo que possua wf.parquet)
                op_n = list(self.portfolio_data.keys())[0]
                m_n = list(self.portfolio_data[op_n].keys())[0]
                s_n = list(self.portfolio_data[op_n][m_n].keys())[0]
                a_n = list(self.portfolio_data[op_n][m_n][s_n].keys())[0]
                a_key = (op_n, m_n, s_n, a_n)

                # ID do Walkforward que você deseja testar (ex: configurado no seu WFM)
                target_wf_id = "12_1_1" 

                print(f"\n[TESTE WF] Reconstruindo Curva OOS para ID: {target_wf_id}")
                
                # 2. Chamada do novo data_type="wf"
                # start_idx=0 para pegar todo o histórico até o candle atual 'i'
                wf_oos_data = self._populate_sim_data(
                    a_key, i, 
                    start_idx=0, 
                    data_type="wf", 
                    psid_or_wfid=target_wf_id
                )

                if wf_oos_data and len(wf_oos_data) > 0:
                    import matplotlib.pyplot as plt
                    import pandas as pd

                    # 3. Processamento dos dados para Plot
                    df_plot = pd.DataFrame(wf_oos_data)
                    df_plot['datetime'] = pd.to_datetime(df_plot['datetime'])
                    
                    # O PnL aqui vem da timeline real, baseada no parâmetro que o WF escolheu
                    df_plot['cum_pnl'] = df_plot['pnl'].cumsum()

                    print(f" -> Sucesso! {len(df_plot)} pontos de dados OOS encontrados.")
                    print(f" -> Parâmetros únicos utilizados no período: {df_plot['best_param'].nunique()}")
                    print(f" -> PnL Final Acumulado: {df_plot['cum_pnl'].iloc[-1]:.2f}")

                    # 4. Plotagem da Curva
                    plt.figure(figsize=(12, 6))
                    plt.plot(df_plot['datetime'], df_plot['cum_pnl'], label=f"OOS Curve - WF {target_wf_id}", color='#2ecc71')
                    plt.title(f"Walkforward OOS Performance: {a_n} ({target_wf_id})")
                    plt.xlabel("Timeline")
                    plt.ylabel("Cumulative PnL")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Salva ou mostra (dependendo do seu ambiente)
                    plt.show()
                else:
                    print(f" -> [Erro] Nenhum dado OOS retornado para {a_key}. Verifique se o ID '{target_wf_id}' existe no wf.parquet.")

                print(f"\n{'='*60}\n")

            
            if i < 3 or i > len(self.datetime_timeline)-3: 
                print(f"> {step_dt} - Portfolio PnL: {sim_current_equity:.2f}")
            
        return True
    





    # XXX 1. Testar Walkforward, def deve puxar por ps_id de um wf_id com dado de storage.load e recriar a curva
    # 2. Desenvolver SM/MM  








    # ── Portfolio Defs ───────────────────────────────────────────────

    def _system_money_managers(self, i, dt, hierarchy, psm_sch, pmm_sch, msm_sch, mmm_sch, ssm_sch, smm_sch):
        m_map = self.sm_mm_map
        p_name = self.name
        p_key = (p_name,)

        # If any of the two need to run, populate data
        if (dt in psm_sch.get(p_name, set())) or (dt in pmm_sch.get(p_name, set())):
            op_data = self._populate_sim_data(p_key, i)

            if op_data is not None:
                op_df = op_data.get("both", op_data) 
                
                psm = m_map.get("managers", {}).get("psm")
                if psm and dt in psm_sch.get(p_name, set()):
                    hierarchy = psm.main(dt, hierarchy, self.indicator_pool, op_df, self.portfolio_returns)
                    #print("PSM")
                pmm = m_map.get("managers", {}).get("pmm")
                if pmm and dt in pmm_sch.get(p_name, set()):
                    hierarchy = pmm.main(dt, hierarchy, self.indicator_pool, op_df, self.portfolio_returns)
                    #print("PMM")

        # Model and Strat Levels
        seen_models = set()
        seen_strats = set()

        for op_name, _, m_name, _, s_name, _, a_name, _ in self._iter_portfolio_data():
            m_key = (op_name, m_name)
            s_key = (op_name, m_name, s_name)

            # --- NÍVEL MODELO (MSM / MMM) ---
            if m_key not in seen_models:
                seen_models.add(m_key)
                
                if (dt in msm_sch.get(m_key, set())) or (dt in mmm_sch.get(m_key, set())):
                    model_data = self._populate_sim_data(m_key, i)
                    
                    if model_data is not None:
                        model_df = model_data.get("both", model_data)
                        msm = m_map.get("models", {}).get(m_name, {}).get("managers", {}).get("msm")
                        mmm = m_map.get("models", {}).get(m_name, {}).get("managers", {}).get("mmm")

                        if msm and dt in msm_sch.get(m_key, set()): 
                            hierarchy = msm.main(dt, hierarchy, self.indicator_pool, model_df, self.portfolio_returns)
                            #print("msM")
                        if mmm and dt in mmm_sch.get(m_key, set()):
                            hierarchy = mmm.main(dt, hierarchy, self.indicator_pool, model_df, self.portfolio_returns)
                            #print("mmM")

            # Strat level — executa apenas 1x por strat
            if s_key not in seen_strats:
                seen_strats.add(s_key)
                
                if (dt in ssm_sch.get(s_key, set())) or (dt in smm_sch.get(s_key, set())):
                    strat_data = self._populate_sim_data(s_key, i)
                    
                    if strat_data is not None:
                        strat_df = strat_data.get("both", strat_data)
                        ssm = m_map.get("models", {}).get(m_name, {}).get("strats", {}).get(s_name, {}).get("managers", {}).get("ssm")
                        smm = m_map.get("models", {}).get(m_name, {}).get("strats", {}).get(s_name, {}).get("managers", {}).get("smm")

                        if ssm and dt in ssm_sch.get(s_key, set()):
                            hierarchy = ssm.main(dt, hierarchy, self.indicator_pool, strat_df, self.portfolio_returns)
                            #print("ssm")
                        if smm and dt in smm_sch.get(s_key, set()):
                            hierarchy = smm.main(dt, hierarchy, self.indicator_pool, strat_df, self.portfolio_returns)
                            #print("smm")
        return hierarchy

    def _update_pos_with_backtest_ret(self, step_dt, active_positions):
        for idf, pos_info in active_positions.items():
            # ifs       = (op, mod, strat, asset)
            # pos_info  = {"weight": 0.1, "lot": 1.0, "type": "wf", "id": "48_48_48", "meta": {"margin": ...}}}
            tid = pos_info["id"]
            wht = pos_info["weight"] # Defined by Money Manager (capital allocated)

            asset_data = None# instance.get(idf, {})

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
    def _populate_sim_data(self, key, i, start_idx=0, side=None, data_type="aggr", psid_or_wfid=None):
        """
        Recupera dados de PnL ou resultados brutos (parsets).
        
        Args:
            key (tuple): Chave da hierarquia (op,), (op, m), (op, m, s) ou (op, m, s, a).
            i (int): Índice final da timeline.
            start_idx (int): Índice inicial da busca (default 0).
            side (str): "BOTH", "LONG", "SHORT".
            data_type (str): "aggr" (memória) ou "parset" (disco/parquet).
            ps_id (str/int): ID específico da posição para filtragem.
        """
        
        # --- CASO 1: DADOS AGREGADOS (Rápido, em Memória) ---
        if data_type == "aggr":
            node = self.sim_data.get(key)
            if not node: return None

            directions = ["BOTH", "LONG", "SHORT"]
            
            def slice_data(block):
                # Retorna o slice do start_idx até o i atual (inclusive)
                # zipando com as colunas para manter o formato de dicionário
                data_slice = block["data"][start_idx : i + 1]
                cols = block["cols"]
                return {col: data_slice[:, idx].tolist() for idx, col in enumerate(cols)}

            if side is not None:
                side_upper = side.upper()
                data_block = node.get(side_upper)
                return slice_data(data_block) if data_block else None

            # Retorno multi-direcional
            full_payload = {}
            for d in directions:
                data_block = node.get(d)
                if data_block:
                    full_payload[d.lower()] = slice_data(data_block)
            return full_payload if full_payload else None

        # --- CASO 2: DADOS DE PARSET (Leitura de Disco/Storage) ---
        elif data_type == "parset":
            try:
                # Chama o seu método load conforme definido na sua classe Storage
                asset_data = self.storage.load(key)
                
                # O seu load retorna um dict. O que queremos para simulação é a 'timeline'
                raw_df = asset_data.get("timeline")
                
                if raw_df is None or raw_df.is_empty():
                    return None

                # Filtragem por PS_ID (Seu ID longo)
                if psid_or_wfid is not None:
                    raw_df = raw_df.filter(pl.col("ps_id") == psid_or_wfid)

                # Filtragem Temporal baseada na sua timeline do backtest
                end_dt = self.datetime_timeline[i]
                
                if start_idx is not None:
                    start_dt = self.datetime_timeline[start_idx]
                    # Note que usamos a coluna 'datetime' que o seu _build_timeline cria
                    raw_df = raw_df.filter(
                        (pl.col("datetime") >= start_dt) & 
                        (pl.col("datetime") <= end_dt)
                    )
                else:
                    raw_df = raw_df.filter(pl.col("datetime") == end_dt)

                return raw_df.to_dicts()
            
            except Exception as e:
                print(f"Erro ao carregar parset para {key}: {e}")
                return None
            
        elif data_type=="wf":
            try: 
                if psid_or_wfid is None:
                    print(f"    < [Portfolio._populate_sim_data] no ps_id")

                end_dt = self.datetime_timeline[i]
                start_dt = self.datetime_timeline[start_idx] if start_idx is not None else 0

                return self.storage.load_walkforward(
                    key=key,
                    wf_id=str(psid_or_wfid),
                    start_dt=start_dt,
                    end_dt=end_dt
                )

            except Exception as e:
                print(f"    < [Portfolio._populate_sim_data] error constructing Walkforward {psid_or_wfid} for {key}: {e}")

        print(f"    < [Portfolio._populate_sim_data] data_type unknown")
        return None

    # Loads each results data, maps path and generates aggregated results, then clears memory one by one 
    def _load_selected_saved_returns_data(self): 
        storage = self.storage #Storage(base_path=self.data_storage_base_path)
        self.sim_data = {}

        # # Specific Aggr
        # asset_aggr = [dt, ps_id1, ps_id2, ps_id3] # (op_name, m_name, s_name, a_name)
        # strat_aggr = [dt, asset1, asset2, asset3] # (op_name, m_name, s_name)
        # model_aggr = [dt, strat1, strat2, strat3] # (op_name, m_name)
        # opera_aggr = [dt, model1, model2, model3] # (op_name)
        # # Global Aggr
        # portf_aggr = [dt, model1, model2, model3, model4, ...] # (self.name) # Joins all opera_aggr into one

        # Acumuladores hierárquicos: { key: { direction: { child_name: series } } }
        temp_asset_cache, strat_acc, model_acc, opera_acc, portf_acc = {}, {}, {}, {}, {} # { (op, m, s, a): { "BOTH": df, "LONG": df... } }
        unique_dts = set()

        # --- 1. COLETA DE DADOS E TIMELINE ---
        for op_n, _, m_n, _, s_n, _, a_n, _ in self._iter_portfolio_data():
            config = self.portfolio_data[op_n][m_n][s_n][a_n]
            side_pref = config.get("side", "BOTH").lower() if isinstance(config, dict) else str(config).lower()
            separate_ls = config.get("analise_long_short_separate", False) if isinstance(config, dict) else False

            asset_data = storage.load(op_n, m_n, s_n, a_n)
            timeline_df = asset_data.get("timeline")
            if timeline_df is None or timeline_df.is_empty(): continue

            unique_dts.update(timeline_df['datetime'].to_list())
            
            # Prepara as vias deste ativo
            vias = {"BOTH": side_pref}
            if separate_ls:
                vias.update({"LONG": "long", "SHORT": "short"})

            asset_entry = {}
            for dir_name, side_val in vias.items():
                asset_entry[dir_name] = self.get_aggr_pnl_by_side(timeline_df, side_val, a_n)
            
            temp_asset_cache[(op_n, m_n, s_n, a_n)] = asset_entry
            
            # Registro de Metadados de Disco (apenas uma vez por ativo, independente da via)
            base_path = storage._asset_path(op_n, m_n, s_n, a_n)
            self.sim_data[(op_n, m_n, s_n, a_n)] = {
                "type": "disk",
                "trades_path": str(base_path / "trades" / "trades.parquet"),
            }

        # --- 2. ALINHAMENTO E AGREGAÇÃO (WIDE) ---
        self.datetime_timeline = sorted(unique_dts)
        timeline_global = pl.DataFrame({"datetime": self.datetime_timeline})

        # A. Ativos -> Estratégias
        for a_key, vias_dict in temp_asset_cache.items():
            op_n, m_n, s_n, a_n = a_key
            s_key = (op_n, m_n, s_n)
            
            for d_name, pnl_df in vias_dict.items():
                aligned = timeline_global.join(pnl_df, on="datetime", how="left").fill_null(0.0)
                pnl_series = aligned.get_column(a_n)
                
                # Salva no sim_data do Ativo
                self.sim_data[a_key].setdefault(d_name, {})
                self.sim_data[a_key][d_name] = {"data": pnl_series.to_numpy().reshape(-1, 1), "cols": [a_n]}
                
                # Alimenta acumulador da Estratégia
                strat_acc.setdefault(s_key, {}).setdefault(d_name, {})[a_n] = pnl_series

        del temp_asset_cache

        # B. Estratégias -> Modelos
        for s_key, directions in strat_acc.items():
            self.sim_data[s_key] = {"type": "aggr"}
            for d_name, assets in directions.items():
                wide_df = pl.DataFrame(assets)
                self.sim_data[s_key][d_name] = {"data": wide_df.to_numpy(), "cols": wide_df.columns}
                
                # Média para o nível Modelo
                m_series = wide_df.select(pl.mean_horizontal(pl.all())).to_series().alias(s_key[2])
                model_acc.setdefault((s_key[0], s_key[1]), {}).setdefault(d_name, {})[s_key[2]] = m_series

        # C. Modelos -> Portfólio
        for m_key, directions in model_acc.items():
            self.sim_data[m_key] = {"type": "aggr"}
            op_n = m_key[0]
            m_n = m_key[1]

            for d_name, strats in directions.items():
                wide_df = pl.DataFrame(strats)
                self.sim_data[m_key][d_name] = {"data": wide_df.to_numpy(), "cols": wide_df.columns}

                # PnL serie of this model
                m_series = wide_df.select(pl.mean_horizontal(pl.all())).to_series().alias(m_n)
                
                opera_acc.setdefault((op_n,), {}).setdefault(d_name, {})[m_n] = m_series

                global_col = f"{op_n}_{m_n}"
                portf_acc.setdefault((self.name,), {}).setdefault(d_name, {})[global_col] = m_series

        # D. Operation
        for o_key, directions in opera_acc.items():
            self.sim_data[o_key] = {"type": "aggr"}
            for d_name, models in directions.items():
                wide_df = pl.DataFrame(models)
                self.sim_data[o_key][d_name] = {"data": wide_df.to_numpy(), "cols": wide_df.columns}

        # E. Portfólio
        for p_key, directions in portf_acc.items():
            self.sim_data[p_key] = {"type": "aggr"}
            for d_name, all_models in directions.items():
                wide_df = pl.DataFrame(all_models)
                self.sim_data[p_key][d_name] = {"data": wide_df.to_numpy(), "cols": wide_df.columns}

        return True
    
    def _pre_compute_and_calc_rebalance_schedule(self, global_assets, sm_mm_map):
        psm_sch, msm_sch, ssm_sch, pmm_sch, mmm_sch, smm_sch = {}, {}, {}, {}, {}, {}
        params_pool = {}
        
        timeline = self.datetime_timeline
        sd = self.sim_data

        # 1. Portfolio Level (PSM / PMM)
        p_n = self.name
        p_key = (p_n,)
        p_magrs = sm_mm_map.get("managers", {})
        p_node = {p_key: sd.get(p_key)}

        # Aqui usamos o nome da operação raiz conforme definido no sm_mm_map
        if p_node[p_key]:
            # PSM
            psm = p_magrs.get("psm") or PortfolioSystemManager(PortfolioSystemManagerParams())
            self.indicator_pool, sd, params_pool = psm.pre_compute(global_assets, timeline, sd, p_node, self.indicator_pool)
            psm_sch[p_n] = psm.get_schedule(timeline) # Key: str (op_name)
            p_magrs["psm"] = psm
            
            # PMM
            pmm = p_magrs.get("pmm") or PortfolioMoneyManager(PortfolioMoneyManagerParams())
            self.indicator_pool, sd, params_pool = pmm.pre_compute(global_assets, timeline, sd, p_node, self.indicator_pool)
            pmm_sch[p_n] = pmm.get_schedule(timeline) # Key: str (op_name)
            p_magrs["pmm"] = pmm

            # 2. Nível Modelos
            for op_name, op_models in self.portfolio_data.items():
                for m_name, m_strats in op_models.items():
                    m_key = (op_name, m_name)
                    # Busca o manager no mapa usando o nome do modelo
                    m_info = sm_mm_map.get("models", {}).get(m_name, {})
                    m_magrs = m_info.get("managers", {})
                    m_node = {m_key: sd.get(m_key)}

                    if m_node[m_key]:
                        # MSM
                        msm = m_magrs.get("msm") or ModelSystemManager(ModelSystemManagerParams())
                        self.indicator_pool, sd, params_pool = msm.pre_compute(global_assets, timeline, sd, m_node, self.indicator_pool)
                        msm_sch[m_key] = msm.get_schedule(timeline)
                        m_magrs["msm"] = msm

                        # MMM
                        mmm = m_magrs.get("mmm") or ModelMoneyManager(ModelMoneyManagerParams())
                        self.indicator_pool, sd, params_pool = mmm.pre_compute(global_assets, timeline, sd, m_node, self.indicator_pool)
                        mmm_sch[m_key] = mmm.get_schedule(timeline)
                        m_magrs["mmm"] = mmm

                    # --- 3. NÍVEL ESTRATÉGIAS ---
                    for s_name in m_strats.keys():
                        s_key = (op_name, m_name, s_name)
                        s_info = m_info.get("strats", {}).get(s_name, {})
                        s_magrs = s_info.get("managers", {})
                        s_node = {s_key: sd.get(s_key)}

                        if s_node[s_key]:
                            # SSM
                            ssm = s_magrs.get("ssm") or StratSystemManager(StratSystemManagerParams())
                            self.indicator_pool, sd, params_pool = ssm.pre_compute(global_assets, timeline, sd, s_node, self.indicator_pool)
                            ssm_sch[s_key] = ssm.get_schedule(timeline)
                            s_magrs["ssm"] = ssm

                            # SMM
                            smm = s_magrs.get("smm") or StratMoneyManager(StratMoneyManagerParams())
                            self.indicator_pool, sd, params_pool = smm.pre_compute(global_assets, timeline, sd, s_node, self.indicator_pool)
                            smm_sch[s_key] = smm.get_schedule(timeline)
                            s_magrs["smm"] = smm

        return params_pool, psm_sch, msm_sch, ssm_sch, pmm_sch, mmm_sch, smm_sch
    
    # ── Datetime timeline mapping ───────────────────────────────────────────────

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
                    "EURUSD": {
                        "side": "BOTH",
                        "analise_long_short_separate": True
                    }
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
            self.datetime_timeline.update(asset_df["datetime"])

        return unique_assets_dts

        
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




