# Holds >1 models, doesn't define Assets, Server uniquely to Manage Positions between multiple models has to dominate over all MMM and MMA

from dataclasses import dataclass, field
from typing import Optional
from PortfolioSystemManager import PortfolioSystemManager
from PortfolioMoneyManager import PortfolioMoneyManager
from BaseClass import BaseClass, BaseManager
from Storage import Storage
from Asset import Asset
import polars as pl, uuid, sys, os, json

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



    # XXX - ADICIONAR opção para pegar apenas 1 valor [i] do ind
    # XXX - Gerar nova DEBUG para testar todas as configurações do BaseClass para os indicadores
    # XXX - Eliminar todos envios de indicador_pool e portfolio_returns, usar self.portfolio.indicator_pool e self.portfolio.portfolio_returns
    # XXX - Consolidar todo tipo de SM/MM para sm_mm_map, eliminar do self do portfolio
    # XXX - Transformar hierarchy em self.hierarchy no nível Portfolio.py, assim com self.indicator_pool
    
    # XXX - Gerar mais estratégias mockup para testar
    

#                                                                                               I M P O R T A N T E   L E R
    
#     - No backtest de Operation para cada Asset-Strat calcular o capital para lot_min com leverage mínimo aproximando de 100k, add para ledger: pnl (agora é o pnl $ final de 1 lot), perc (% final de 1 lot), raw_perc (% final de 1 lot)
# add para matrix: lot (agora é asset.lot_min ao invés de 1), required_capital_unit (capital mínimo necessário para lot_min e leverage min), pnl (resultado $ para dt atual), perc (resultado % para dt atual), raw_perc (var % do asset mesmo)
# Assim posso ter o menor lot dentro de 100k e depois, na hora de fazer o aggr usar o fator de multiplicação para igualar todos mais próximo de 100k possível
# apenas diferenciar margin, capital, etc. para isso vai precisar reimplementar o sistema de gestão financeira básica, usando tudo no mínimo. NOTE Permitido fator de multiplicação < 1 para casos onde a margem > capital minimo

# Em resumo:
# - Backtest calcular os resultados com lot minimo dentro de 100k
# - Na hora de calcular o aggr, fazer 100k/required_capital_unit para calcular para ter o fator de multiplicação para * os resultados e "igualar" ao máximo os resultados diferentes
# - Em Portfolio posso apenas usar esses dados para encontrar a posição ideal para o capital alocado ao Model-Strat-Asset
# - ELIMINAR lot_value de asset

#    - Ler metadados dos resultados para ter acesso aos assets dos models



#     - Terminar todos os SystemManagers com a ultima versão no Gemini, resolvendo qual aggr o MSM vai receber
#     - Gerar o sistema de entrada e junto desenvolver os Money Managers
#     - Desenvolver sistema de saida 
    

    # - analise_long_short_separate está errado na hora do calculo do aggr
    # - Está crashando muito, procurar otimizar o código e talvez instanciar
    # - Para otimizar, talvez gerar parquet de aggr no Operation e carregar direto
    # - Salvar ind em parquet?





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
        self.portfolio_returns: dict={}
        self.sim_data: dict= {}
        self.storage = Storage(base_path=portfolio_params.data_storage_base_path)
    
        
    def _simulation(self):
        # 1 - Init, populating sim_data
        self.sim_current_equity = self.portfolio_parameters.get("capital", 100000.0)
        self.active_positions = {} 
        self.portfolio_returns = {}
        self.indicator_pool = {}
        self.current_idx = 0

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
            self.current_idx = i

            # Init step data
            self.portfolio_returns[step_dt] = {"assets": {}}
            step_perc_total        =     0.0
            step_pnl_nominal_total =  0.0

            #||=====================================================================================||#
            
            # Exits at [i] open
            for idf, pos_info in self.active_positions.items():
                pass

            #||=====================================================================================||#
            
            # Entries at [i] open - MM Tactical Level - Bottom Up (MM can change with exit/entry)
            for idf, pos_info in self.hierarchy.items():
                if portfolio_simulation_with_backtest_results: pass
                    
                # -> NOTE Para System e Money M colocar opção de seprar long (lot_size > 0) de short
                # 1. Add long_active e short_active ao invés de só active, padrão ambos true
                # 2. Iterares hierarchy, checks all parsets that are active (NOTE WF must activate only the current parset in SMM)
                # 3. Creates temporary portfolio arrangements with current assets + new positions, finds best portfolio arrangement and calculates positions sizes
                # First-Come First-Served - Allocates 10% until 100% is hit, following hierarchy


                
                # https://www.youtube.com/watch?v=-99_Cn1qzak
                # Risk Parity & Volatility Targeting in Trend Following Portfolios

                # Size i = Target Risk / α i * Price i

                # Ajustar para usar o fator de multiplicação do asset
                def apply_correlation_discount(weights, returns_df, lookback=252):
                    # Compute rolling correlation matrix
                    corr_matrix = returns_df.rolling(window=lookback).corr().iloc[-1]

                    # Calculate Portfolio Variance scalar (w^T * Sigma * w)
                    port_variance = np.dot(weights.T, np.dot(corr_matrix, weights))

                    # Derive Diversification Multiplier
                    div_multiplier = 1.0 / np.sqrt(port_variance)

                    # Scale position sizes down dynamically
                    adjust_weights = weights * div_multiplier
                    return adjust_weights

                from scipy.optimize import minimize
                def risk_parity_objective(weights, cov_matrix):
                    # Marginal risk contribution of each asset
                    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                    marginal_contrib = np.dot(cov_matrix, weights)
                    risk_contrib = weights * marginal_contrib / port_variance

                    # Target equal risk distribution (1/N)
                    target_risk = 1.0 / len(weights)

                    # Return sum of squared penalties
                    return np.sum((risk_contrib - target_risk)**2)

                result = minimize(risk_parity_objective, initial_weights, args=(cov_matrix,), method='SLSQP')

                def apply_regime_filter(optimal_weights, prices_df, sma_window=252):
                    # Calculate long term macro trend
                    sma = prices_df.rolling(window=sma_window).mean()

                    # Generate binary regime mask (1.0 if bullish, 0.0 if bearish)
                    # Element wise comparison across all assets
                    regime_mask = np.where(prices_df > sma, 1.0, 0.0)

                    # Final Portfolio Weights overlay
                    final_weights = optimal_weights * regime_mask

                    return final_weights

            #||=====================================================================================||#
            
            # Updates PnL of open positions at [i] ends in previous step
            #update_func_to_use(step_dt, self.active_positions)

            #||=====================================================================================||#
            
            # Updates System and Money Managers - Top Down - at [i] ends
            self._system_money_managers(i, step_dt, psm_sch, pmm_sch, msm_sch, mmm_sch, ssm_sch, smm_sch)
                                                    
            #||=====================================================================================||#
                
            import matplotlib.pyplot as plt
            import numpy as np
            import polars as pl
            import matplotlib as mpl # Import para colormaps novos

            if i == int(len(self.datetime_timeline)-4):
                print("\n" + "="*80)
                print("      DEBUG SYSTEM: INDICATOR HIERARCHY & LOOKUP TEST")
                print("="*80)
                for i, k in enumerate(list(self.indicator_pool.keys())):
                    print(f" Exemplo de Chave {i}: {k}")
                print("="*80)
           
            if i < 3 or i > len(self.datetime_timeline)-4: 
                print(f"> {step_dt} - Portfolio PnL: {self.sim_current_equity:.2f}")
            
        return True
    
    # ── Portfolio Defs ───────────────────────────────────────────────

    def _system_money_managers(self, i, dt, psm_sch, pmm_sch, msm_sch, mmm_sch, ssm_sch, smm_sch):
        m_map = self.sm_mm_map
        p_name = self.name
        p_key = (p_name,)

        # If any of the two need to run, populate data
        if (dt in psm_sch.get(p_name, set())) or (dt in pmm_sch.get(p_name, set())):
            psm = m_map.get("managers", {}).get("psm")
            if psm and dt in psm_sch.get(p_name, set()):
                self.hierarchy = psm.main(i, dt, p_key)
                #print("PSM")
            pmm = m_map.get("managers", {}).get("pmm")
            if pmm and dt in pmm_sch.get(p_name, set()):
                self.hierarchy = pmm.main(i, dt, p_key)
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
                    msm = m_map.get("models", {}).get(m_name, {}).get("managers", {}).get("msm")
                    mmm = m_map.get("models", {}).get(m_name, {}).get("managers", {}).get("mmm")

                    if msm and dt in msm_sch.get(m_key, set()): 
                        self.hierarchy = msm.main(i, dt, m_key)
                        #print("msM")
                    if mmm and dt in mmm_sch.get(m_key, set()):
                        self.hierarchy = mmm.main(i, dt, m_key)
                        #print("mmM")

            # Strat level — executa apenas 1x por strat
            if s_key not in seen_strats:
                seen_strats.add(s_key)
                
                if (dt in ssm_sch.get(s_key, set())) or (dt in smm_sch.get(s_key, set())):
                    ssm = m_map.get("models", {}).get(m_name, {}).get("strats", {}).get(s_name, {}).get("managers", {}).get("ssm")
                    smm = m_map.get("models", {}).get(m_name, {}).get("strats", {}).get(s_name, {}).get("managers", {}).get("smm")

                    if ssm and dt in ssm_sch.get(s_key, set()):
                        self.hierarchy = ssm.main(i, dt, s_key)
                        #print("ssm")
                    if smm and dt in smm_sch.get(s_key, set()):
                        self.hierarchy = smm.main(i, dt, s_key)
                        #print("smm")
        return True

    def _update_pos_with_backtest_ret(self, step_dt):
        for idf, pos_info in self.active_positions.items():
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
            pos_pnl_port = self.sim_current_equity * pos_perc_port # $ pnl in relation to portfolio
            step_pnl_nominal_total += pos_pnl_port # pnl accumulated in this datetime

            # Strat Returns
            self.portfolio_returns[step_dt][idf] = {
                "trade_perc": trade_perc,
                "pos_perc_port": pos_perc_port,
                "pos_pnl_port": pos_pnl_port,
                "weight": wht
            }

        # Updates global
        self.sim_current_equity += step_pnl_nominal_total
        self.portfolio_returns[step_dt].update({
            "portfolio_perc": step_perc_total,
            "portfolio_pnl": step_pnl_nominal_total,
            "equity": self.sim_current_equity
        })

    def _update_pos_with_assets_ret(self, step_dt):
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
            side (str|list): "both", "long", "long".
            data_type (str): "aggr" (memória) ou "parset" (disco/parquet).
            ps_id (str/int): ID específico da posição para filtragem.
        """
        
        # --- CASO 1: DADOS AGREGADOS (Rápido, em Memória) ---
        if data_type == "aggr":
            node = self.sim_data.get(key)
            if not node: return None

            def slice_data(block):
                # Retorna o slice do start_idx até o i atual (inclusive)
                # zipando com as colunas para manter o formato de dicionário
                data_slice = block["data"][start_idx : i + 1]
                cols = block["cols"]
                return {col: data_slice[:, idx].tolist() for idx, col in enumerate(cols)}
            
            # Case of only 1 str
            if isinstance(side, str):
                data_block = node.get(side.lower())
                return slice_data(data_block) if data_block else None

            # Case of list or None, returns mapped dict {"side": {}}
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
            
        elif data_type == "wf": # wf_ids can be str, list[str] or None (all ps_id)
            try:
                if start_idx is None:
                    start_dt_val = None
                elif isinstance(start_idx, str):
                    start_dt_val = start_idx
                else:
                    start_dt_val = self.datetime_timeline[start_idx]
                    
                if i is None:
                    end_dt_val = None
                elif isinstance(i, str):
                    end_dt_val = i
                else:
                    end_dt_val = self.datetime_timeline[i]

                wfm_df = self.storage.load_walkforward_matrix(
                    key=key, side_val=side, wf_ids=psid_or_wfid, 
                    start_dt=start_dt_val, end_dt=end_dt_val
                )
                
                if wfm_df is None or wfm_df.is_empty():
                    print(f"    < [Portfolio._populate_sim_data] wfm_df empty for Walkforward Matrix {psid_or_wfid} for {key}: {e}")
                    return None
                
                # Format: to_dicts() returns [{col1: val, col2: val}, ...] best to itearate line by line
                return wfm_df.to_dicts()
            
            except Exception as e:
                print(f"    < [Portfolio._populate_sim_data] error constructing Walkforward Matrix {psid_or_wfid} for {key}: {e}")

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
        temp_asset_cache, strat_acc, model_acc, opera_acc, portf_acc = {}, {}, {}, {}, {} # { (op, m, s, a): { "both": df, "long": df... } }
        unique_dts = set()

        # --- 1. COLETA DE DADOS E TIMELINE ---
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
            if timeline_df is None or timeline_df.is_empty(): continue
            
            #unique_dts.update(timeline_df['datetime'].to_list())

            # Preparates direction
            vias = {"both": side_pref}
            if separate_ls:
                vias.update({"long": "long", "short": "short"})
            asset_entry = {}
            

            for dir_label, side_val in vias.items():
                sources = []

                # Source A: Parsets
                if calculate_on_data in ["all", "parset"] and timeline_df is not None and not timeline_df.is_empty():
                    p_aggr = self.get_aggr_pnl_by_side(timeline_df, side_val, a_n)
                    if p_aggr is not None and not p_aggr.is_empty():
                        val_col = [c for c in p_aggr.columns if c != "datetime"][0]
                        sources.append(p_aggr.rename({val_col: "parset_pnl"}))

                # Source B: Walkforward
                if calculate_on_data in ["all", "wf"]:
                    wfm_wide = storage.load_walkforward_matrix(a_key, side_val=side_val)
                    if wfm_wide is not None and not wfm_wide.is_empty():
                        wf_cols = [c for c in wfm_wide.columns if c != "datetime"]
                        
                        if wf_cols:
                            wf_aggr = wfm_wide.select([
                                pl.col("datetime"),
                                pl.mean_horizontal(wf_cols).alias("wf_pnl")
                            ])
                            sources.append(wf_aggr)

                # Combines all sources to generate aggr for the asset
                if not sources: continue

                if len(sources) == 1: 
                    val_col = [c for c in sources[0].columns if c != "datetime"][0]
                    combined = sources[0].rename({val_col: a_n})
                else: # If "all", takes avg between parsets and wf
                    combined = (
                        sources[0]
                        .join(sources[1], on="datetime", how="full", coalesce=True)
                        .fill_null(0.0)
                        .select([
                            pl.col("datetime"),
                            pl.mean_horizontal(["parset_pnl", "wf_pnl"]).alias(a_n)
                        ])
                    )
                
                asset_entry[dir_label] = combined

                # Updates unique datetime
                unique_dts.update(combined['datetime'].to_list())
            
            # Registro de Metadados de Disco (apenas uma vez por ativo, independente da via)
            if asset_entry:
                temp_asset_cache[a_key] = asset_entry
                base_path = storage._asset_path(*a_key)
                self.sim_data[a_key] = {
                    "type": "disk",
                    "trades_path": str(base_path / "trades" / "trades.parquet"),
                }

        # --- 2. ALINHAMENTO TEMPORAL E AGREGAÇÃO SUBORDINADA ---
        if not unique_dts:
            print(" < [Portfolio._load_selected_saved_returns_data] Error: No data available to load.")
            return False

        self.datetime_timeline = sorted(unique_dts)
        timeline_global = pl.DataFrame({"datetime": self.datetime_timeline})

        # A. Ativos -> Estratégias
        for a_key, directions in temp_asset_cache.items():
            op_n, m_n, s_n, a_n = a_key
            s_key = (op_n, m_n, s_n)
            
            for d_name, pnl_df in directions.items():
                # Alinha o PnL do ativo com a timeline global do portfólio
                aligned = timeline_global.join(pnl_df, on="datetime", how="left").fill_null(0.0)
                pnl_series = aligned.get_column(a_n)
                
                self.sim_data[a_key].setdefault(d_name, {})
                self.sim_data[a_key][d_name] = {
                    "data": pnl_series.to_numpy().reshape(-1, 1), 
                    "cols": [a_n]
                }
                strat_acc.setdefault(s_key, {}).setdefault(d_name, {})[a_n] = pnl_series

        # B. Estratégias -> Modelos
        for s_key, directions in strat_acc.items():
            self.sim_data[s_key] = {"type": "aggr"}
            for d_name, assets in directions.items():
                wide_df = pl.DataFrame(assets)

                s_avg = wide_df.select(pl.mean_horizontal(pl.all())).to_series().alias("@total")
                wide_df = wide_df.with_columns(s_avg)

                self.sim_data[s_key][d_name] = {"data": wide_df.to_numpy(), "cols": wide_df.columns}
                
                m_series = wide_df.get_column("@total").alias(s_key[2])
                model_acc.setdefault((s_key[0], s_key[1]), {}).setdefault(d_name, {})[s_key[2]] = m_series

        # C. Modelos -> Portfólio
        for m_key, directions in model_acc.items():
            self.sim_data[m_key] = {"type": "aggr"}
            for d_name, strats in directions.items():
                wide_df = pl.DataFrame(strats)
                m_avg = wide_df.select(pl.mean_horizontal(pl.all())).to_series().alias("@total")
                wide_df = wide_df.with_columns(m_avg)

                self.sim_data[m_key][d_name] = {"data": wide_df.to_numpy(), "cols": wide_df.columns}

                o_series = wide_df.get_column("@total").alias(m_key[1])
                opera_acc.setdefault((m_key[0],), {}).setdefault(d_name, {})[m_key[1]] = o_series
                
                port_col = f"{m_key[0]}_{m_key[1]}"
                portf_acc.setdefault((self.name,), {}).setdefault(d_name, {})[port_col] = o_series

        # D. Operation
        for o_key, directions in opera_acc.items():
            self.sim_data[o_key] = {"type": "aggr"}
            for d_name, models in directions.items():
                wide_df = pl.DataFrame(models)
                self.sim_data[o_key][d_name] = {"data": wide_df.to_numpy(), "cols": wide_df.columns}
                #print(f"Operation {o_key} - {d_name}:\n{wide_df.head()}")

        # E. Portfólio
        for p_key, directions in portf_acc.items():
            self.sim_data[p_key] = {"type": "aggr"}
            for d_name, components in directions.items():
                wide_df = pl.DataFrame(components)
                self.sim_data[p_key][d_name] = {"data": wide_df.to_numpy(), "cols": wide_df.columns}
                #print(f"Portfolio {p_key} - {d_name}:\n{wide_df.head()}")

        return True
    

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
                    "wf_ids":      {},
                }

        self.hierarchy = hcy
        return True

    def _pre_compute_and_calc_rebalance_schedule(self, global_assets, sm_mm_map):
        psm_sch, msm_sch, ssm_sch, pmm_sch, mmm_sch, smm_sch = {}, {}, {}, {}, {}, {}
        params_pool = {}

        DEFAULT_MGR_CONFIG = {
            "psm": (PortfolioSystemManager, PortfolioSystemManagerParams),
            "pmm": (PortfolioMoneyManager, PortfolioMoneyManagerParams),
            "msm": (ModelSystemManager, ModelSystemManagerParams),
            "mmm": (ModelMoneyManager, ModelMoneyManagerParams),
            "ssm": (StratSystemManager, StratSystemManagerParams),
            "smm": (StratMoneyManager, StratMoneyManagerParams),
        }
        
        timeline = self.datetime_timeline
        last_idx = len(timeline) - 1

        # 1. Portfolio Level (PSM / PMM)
        p_name = self.name
        p_key = (p_name,)
        p_magrs = sm_mm_map.get("managers", {})

        # Searches data via _populate_sim_data
        p_data = self._populate_sim_data(key=p_key, i=last_idx, start_idx=0, data_type="aggr")
        
        if p_data:
            p_node = {p_key: p_data}
           
            # PSM and PMM
            for mgr_key, mgr_class, sch_dict in [
                ("psm", DEFAULT_MGR_CONFIG["psm"], psm_sch),
                ("pmm", DEFAULT_MGR_CONFIG["pmm"], pmm_sch)
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
                        ("msm", DEFAULT_MGR_CONFIG["msm"], msm_sch),
                        ("mmm", DEFAULT_MGR_CONFIG["mmm"], mmm_sch)
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
                            ("ssm", DEFAULT_MGR_CONFIG["ssm"], ssm_sch),
                            ("smm", DEFAULT_MGR_CONFIG["smm"], smm_sch)
                        ]: 
                            mgr = s_magrs.get(mgr_key) or mgr_class()
                            mgr.set_portfolio(self)
                            self.indicator_pool, _ = mgr.pre_compute(
                                global_assets, timeline, s_node, self.indicator_pool, s_key)
                            sch_dict[s_key] = mgr.get_schedule(timeline)
                            s_magrs[mgr_key] = mgr

        return params_pool, psm_sch, msm_sch, ssm_sch, pmm_sch, mmm_sch, smm_sch

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
    
    def _get_all_mm_ind_datetimes(self, data_source="local"):
        assets = self.global_assets #Asset.load_all() # NOTE Deletar futuramente
        unique_ind_dts = set()
        repeated_assets = set()

        pmm = self.sm_mm_map.get("managers", {}).get("pmm")
        mm_inds = pmm.indicators if (pmm and pmm.indicators) else {}
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
                    assets = pmm.assets if pmm and pmm.assets else []
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

    """"""
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
    # - Para cada trade em self.active_positions deve puxar os dados do trades_matrix, verificar se precisa atualizar o lot (diminuir ou aumentar, pode ser uma def enviada, default None, mantêm mesma coisa até saida) para saber o PnL * Lot atualizado
    # - Criando e enviando a imagem do datetime para self.portfolio_returns



    def _run(self):
        # Data Init - Loads data, saves unique datetimes and generates aggr results
        print("     > Populating Portfolio Data from Database")
        self._load_selected_saved_returns_data()
        self._init_hierarchy()

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
    from Volatility import Volatility # type: ignore

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
            "param1": range(21, 21+1, 1),
        },
        indicators={
            "vol": Volatility(asset="@total_both", timeframe="tick", 
                              window="param1", aggr_days=True, 
                              price_col="pnl", min_periods="param1")
        },
        #assets={'EURUSD'},
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

    mmm = ModelMoneyManager(ModelMoneyManagerParams(
        capital=100000.0,
        reb_frequency="weekly",
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
                }
            },
            "FX Mean Reversion": {
                "AT30": {
                    "USDJPY": {
                        "side": "both",
                        "analise_long_short_separate": True,
                        "calculate_on_data": "wf",
                    }
                }
            },
            "Futures Mean Reversion": {
                "AT20": {
                    "WIN$": {
                        "side": "both",
                        "analise_long_short_separate": True,
                        "calculate_on_data": "wf",
                    }
                },
                "AT30": {
                    "WIN$": {
                        "side": "both",
                        "analise_long_short_separate": True,
                        "calculate_on_data": "wf",
                    }
                }
            },
        }
    }

    # SM/MM mapeados por nível — referenciados durante a simulação
    sm_mm_map = {
        "managers": {"psm": psm, "pmm": pmm, "separate_ls": True, "side": 'both'},
        "models": {
            "FX MA Trend Following": {
                "managers": {"msm": msm, "mmm": mmm, "separate_ls": True, "side": 'both'},
                "strats": {
                    "AT15": {"managers": {"ssm": ssm, "smm": smm, "separate_ls": True, "side": 'both'}}
                }
            },
            "FX Mean Reversion": {
                "managers": {"msm": msm, "mmm": mmm, "separate_ls": True, "side": 'both'},
                "strats": {
                    "AT30": {"managers": {"ssm": ssm, "smm": smm, "separate_ls": True, "side": 'both'}}
                }
            },
            "Futures Mean Reversion": {
                "managers": {"msm": msm, "mmm": mmm, "separate_ls": True, "side": 'both'},
                "strats": {
                    "AT20": {"managers": {"ssm": ssm, "smm": smm, "separate_ls": True, "side": 'both'}},
                    "AT30": {"managers": {"ssm": ssm, "smm": smm, "separate_ls": True, "side": 'both'}}
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



    """
    Pontos para melhorar para V2
    - 3 SM para rankear, filtrar e limitar com pesos cada Nível e Asset
    - 1 MM para gerenciar as Strat(s) e Asset(s) dos param_set selecionados
    - Carregar o objeto do Model também, além dos resultados, para ter acesso aos ativos


    """









