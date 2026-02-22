import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- FUNÇÃO DE CARREGAMENTO (SUA VERSÃO INTEGRADA) ---

def load_data_robust(data_path):
    """Carrega dados tratando a união de date e time para datetime."""
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

        # 1. Leitura inicial
        if data_path.endswith(('.xlsx', '.xls')):
            df = pl.read_excel(data_path)
        elif data_path.endswith('.csv'):
            df = pl.read_csv(data_path)
        else:
            return None

        # Normalize para lowercase
        df = df.select([pl.col(c).alias(c.lower()) for c in df.columns])

        # 2. TRATAMENTO DE DATETIME (Baseado na sua def)
        if "datetime" not in df.columns:
            if "date" in df.columns and "time" in df.columns:
                df = df.with_columns([
                    pl.col("date").cast(pl.Utf8),
                    pl.col("time").cast(pl.Utf8)
                ])
                df = df.with_columns(
                    (pl.col("date") + pl.lit(" ") + pl.col("time")).alias("datetime")
                )
            else:
                raise ValueError(f"Colunas temporais não encontradas.")

        # 3. CONVERSÃO E FORMATAÇÃO (Tratando '.' ou '-' do MT5)
        df = df.with_columns(
            pl.col("datetime").str.replace_all(r"\.", "-").str.to_datetime(strict=False)
        )

        if df["datetime"].null_count() == len(df):
            raise ValueError("Falha na conversão de datetime: todos os valores resultaram em nulo.")

        return df.sort("datetime")

    except Exception as e:
        print(f"Erro ao processar: {str(e)}")
        return None

# --- INDICADORES (LÓGICA PRIORCOTE E DAYOPEN) ---

def apply_indicators(df):
    # 1. Máxima/Mínima da Semana Anterior (PriorCote Logic)
    df_work = df.with_columns(
        pl.col("datetime").dt.truncate("1w").alias("_week_start")
    )
    
    week_stats = df_work.group_by("_week_start").agg([
        pl.col("high").max().alias("prev_week_high"),
        pl.col("low").min().alias("prev_week_low")
    ]).sort("_week_start")
    
    # Shift(1) para não ter look-ahead bias
    week_stats = week_stats.with_columns([
        pl.col("prev_week_high").shift(1),
        pl.col("prev_week_low").shift(1)
    ])
    
    df_work = df_work.join(week_stats, on="_week_start", how="left")

    # 2. Primeiro Open do Dia (DayOpen Logic)
    df_work = df_work.with_columns(
        pl.col("open").first().over(pl.col("datetime").dt.date()).alias("day_open")
    )
    
    return df_work.drop_nulls()

# --- BACKTEST ---

PATH = r'C:\Users\Patrick\Desktop\Artaxes Portfolio\MAIN\MT5_Dados\WIN$_M10.xlsx'
df = load_data_robust(PATH)

if df is not None:
    df = apply_indicators(df)
    
    trades = []
    pnl_history = []
    total_pnl = 0.0
    
    # Parâmetros
    TP = 2000
    SL = 400
    
    current_trade = None
    day_long_count = 0
    day_short_count = 0
    current_day = None

    data_list = df.to_dicts()

    for i, bar in enumerate(data_list):
        dt = bar['datetime']
        d_date = dt.date()
        
        # Reset Diário
        if d_date != current_day:
            current_day = d_date
            day_long_count = 0
            day_short_count = 0
            
        # Identificar última barra do dia (Saída DayTrade)
        is_last_bar = False
        if i + 1 < len(data_list):
            if data_list[i+1]['datetime'].date() != d_date:
                is_last_bar = True
        else:
            is_last_bar = True

        # 1. GESTÃO DE TRADE ABERTA
        if current_trade:
            entry = current_trade['entry_price']
            side = current_trade['side']
            
            if side == 'long':
                if bar['low'] <= entry - SL:
                    total_pnl -= SL
                    trades.append({'exit_dt': dt, 'pnl': -SL, 'type': 'SL'})
                    current_trade = None
                elif bar['high'] >= entry + TP:
                    total_pnl += TP
                    trades.append({'exit_dt': dt, 'pnl': TP, 'type': 'TP'})
                    current_trade = None
                elif is_last_bar:
                    pnl = bar['open'] - entry
                    total_pnl += pnl
                    trades.append({'exit_dt': dt, 'pnl': pnl, 'type': 'EOD'})
                    current_trade = None
                    
            elif side == 'short':
                if bar['high'] >= entry + SL:
                    total_pnl -= SL
                    trades.append({'exit_dt': dt, 'pnl': -SL, 'type': 'SL'})
                    current_trade = None
                elif bar['low'] <= entry - TP:
                    total_pnl += TP
                    trades.append({'exit_dt': dt, 'pnl': TP, 'type': 'TP'})
                    current_trade = None
                elif is_last_bar:
                    pnl = entry - bar['open']
                    total_pnl += pnl
                    trades.append({'exit_dt': dt, 'pnl': pnl, 'type': 'EOD'})
                    current_trade = None

        # 2. LÓGICA DE ENTRADA
        if not current_trade and not is_last_bar:
            day_open = bar['day_open']
            max_w = bar['prev_week_high']
            min_w = bar['prev_week_low']
            
            # Filtro: Open do dia entre range da semana passada
            if min_w < day_open < max_w:
                # Long: Rompimento máxima semanal
                if day_long_count == 0 and bar['high'] > max_w:
                    entry_p = max(bar['open'], max_w)
                    current_trade = {'side': 'long', 'entry_price': entry_p}
                    day_long_count += 1
                # Short: Rompimento mínima semanal
                elif day_short_count == 0 and bar['low'] < min_w:
                    entry_p = min(bar['open'], min_w)
                    current_trade = {'side': 'short', 'entry_price': entry_p}
                    day_short_count += 1

        pnl_history.append(total_pnl)

    # --- RESULTADOS ---
    print(f"\n--- RELATÓRIO FINAL ---")
    print(f"PNL Total: {total_pnl:.0f} pontos")
    print(f"Trades realizadas: {len(trades)}")
    
    if trades:
        win_rate = (len([t for t in trades if t['pnl'] > 0]) / len(trades)) * 100
        print(f"Win Rate: {win_rate:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(pnl_history, color='blue', lw=1.5)
    plt.axhline(0, color='black', ls='--')
    plt.title(f'Equity Curve - WIN 10min (Weekly Breakout)')
    plt.ylabel('Pontos Acumulados')
    plt.grid(alpha=0.3)
    plt.show()

else:
    print("Falha ao carregar o DataFrame. Verifique o caminho e as colunas.")