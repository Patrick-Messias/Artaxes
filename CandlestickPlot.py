import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

class CandlestickChart:
    def __init__(self, 
                df, 
                timeframe,
                datetime_col='datetime',
                ohlc_cols=None, 
                signals=None,
                indicators=None,
                areas_config=None,
                config=None,
                plot_window=None,
                figsize=(12, 6)):
        
        # Inicializa o gráfico de candlestick
        
        # Parâmetros:
        # df: DataFrame do pandas com colunas OHLC + datetime
        # datetime_col: Nome da coluna de data/hora
        # ohlc_cols: Dicionário com nomes das colunas OHLC (padrão: {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'})
        # signals: Dicionário com sinais de compra/venda
        # indicators: Dicionário com indicadores
        # config: Dicionário com configurações do gráfico
        # figsize: Tamanho da figura
        
        # Configurações padrão
        default_ohlc = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}
        self.ohlc_cols = {**default_ohlc, **(ohlc_cols or {})}
        
        # Validação das colunas
        required_cols = list(self.ohlc_cols.values()) + [datetime_col]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame deve conter as colunas: {required_cols}")
        
        self.df = df.copy()
        self.timeframe = timeframe
        self._setup_timeframe_config()

        self.df[datetime_col] = pd.to_datetime(self.df[datetime_col])
        self.df = self.df.sort_values(datetime_col).reset_index(drop=True)
        self.datetime_col = datetime_col

        if plot_window is not None:
            if isinstance(plot_window, slice):
                self.df = self.df.iloc[plot_window]
            elif isinstance(plot_window, (list, tuple)):
                self.df = self.df.iloc[plot_window[0]:plot_window[1]]
            else:
                raise ValueError("plot_window deve ser slice (ex: [-300:]) ou lista/tupla com início e fim")
        
        # Configurações
        self.signals = signals or {
            'long': {
                'data': None,
                'signal_type': 'compra',
                'color': 'darkgreen',
                'marker': '^',
                'size': 50
            },
            'short': {
                'data': None,
                'signal_type': 'venda', 
                'color': 'darkred',
                'marker': 'v',
                'size': 50
            }
        }
        self.indicators = indicators or {}
        
        default_areas_config = {
            'position': {
                'data': None,
                'colors': {1: 'darkgreen', -1: 'darkred', 0: None},
                'alpha': 0.25,
                'label': 'Posição'
            }
        }
        
        if areas_config:
            # Mescla a configuração passada com a padrão
            for area_name, area_config in areas_config.items():
                if area_name in default_areas_config:
                    # Se area_config é uma Series (apenas dados), converte para dicionário
                    if isinstance(area_config, (pd.Series, list, np.ndarray)):
                        area_config = {'data': area_config}
                    
                    # Mescla mantendo os valores padrão para keys não fornecidas
                    default_areas_config[area_name] = {
                        **default_areas_config[area_name],  # Valores padrão
                        **area_config  # Valores passados (sobrescrevem os padrão)
                    }
                else:
                    # Nova área
                    default_areas_config[area_name] = area_config
        self.areas_config = default_areas_config
        
        self.config = config or {
            'grid': {'alpha': 0.25, 'color': 'grey', 'linestyle': '-', 'linewidth': 0.75},
            'background': 'black',
            'candle_pos': 'green', 
            'candle_neg': 'darkred'
        }
        
        # Aplicar estilo dark se solicitado
        if self.config.get('background') == 'dark_background':
            plt.style.use('dark_background')
            self.config['background'] = 'black'  # Define cor de fundo como preto
        
        # Inicializar gráfico
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.setup_chart()
        
        # Plotar automaticamente
        self.plot_all()

    def _setup_timeframe_config(self):
        """Configura largura dos candles baseado no timeframe"""
        timeframe_configs = {
            'M1': {'candle_width': 0.002, 'body_width': 0.0015, 'risk_length_hours': 0.1},
            'M5': {'candle_width': 0.005, 'body_width': 0.004, 'risk_length_hours': 0.2},
            'M15': {'candle_width': 0.01, 'body_width': 0.008, 'risk_length_hours': 0.5},
            'M30': {'candle_width': 0.02, 'body_width': 0.015, 'risk_length_hours': 1},
            'H1': {'candle_width': 0.03, 'body_width': 0.025, 'risk_length_hours': 2},
            'H4': {'candle_width': 0.06, 'body_width': 0.05, 'risk_length_hours': 6},
            'D1': {'candle_width': 0.8, 'body_width': 0.6, 'risk_length_hours': 24},
            'W1': {'candle_width': 2.0, 'body_width': 1.5, 'risk_length_hours': 48},
            'MN1': {'candle_width': 5.0, 'body_width': 4.0, 'risk_length_hours': 96}
        }
        
        # Usar configuração padrão D1 se timeframe não for reconhecido
        config = timeframe_configs.get(self.timeframe.upper(), timeframe_configs['D1'])
        
        self.candle_width = config['candle_width']
        self.body_width = config['body_width']
        self.risk_length_hours = config['risk_length_hours']
    
    def setup_chart(self):
        """Configura o gráfico base"""
        # Cores para candles de alta e baixa
        self.colors = {'up': self.config.get('candle_pos', 'darkgreen'), 
                    'down': self.config.get('candle_neg', 'darkred')}

        # Configurar background
        background_color = self.config.get('background', 'white')
        self.fig.patch.set_facecolor(background_color)
        self.ax.set_facecolor(background_color)
        
        # Se o background for escuro, ajusta as cores dos textos
        if background_color in ['black', 'darkgray', 'darkblue', 'darkgreen']:
            self.ax.tick_params(colors='white')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.title.set_color('white')
            # Ajusta a cor do grid para melhor contraste
            grid_config = self.config.get('grid', {})
            if 'color' not in grid_config:
                grid_config['color'] = 'lightgray'
        
        # ← AJUSTE DINÂMICO DO LOCATOR baseado no timeframe ↓
        self._setup_dynamic_locator()
        
        # Formatação do eixo x
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        #self.fig.autofmt_xdate()
        
        # Aplicar configurações de grid
        grid_config = self.config.get('grid', {'alpha': 0.2, 'color': 'grey'})
        self.ax.grid(True, **grid_config)
        
        # Tamanho pequeno para as legendas dos eixos
        self.ax.tick_params(axis='x', labelsize=6)
        self.ax.tick_params(axis='y', labelsize=6)

    def _setup_dynamic_locator(self):
        """Configura o locator do eixo x baseado no timeframe e número de candles"""
        num_candles = len(self.df)
        
        # Define intervalos baseados no timeframe e quantidade de candles
        if self.timeframe in ['M1', 'M5', 'M15']:
            # Timeframes muito curtos - mostrar menos labels
            if num_candles <= 100:
                self.ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            elif num_candles <= 500:
                self.ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            else:
                self.ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                
        elif self.timeframe in ['M30', 'H1']:
            # Timeframes curtos
            if num_candles <= 200:
                self.ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            elif num_candles <= 1000:
                self.ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
            else:
                self.ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                
        elif self.timeframe in ['H4', 'D1']:
            # Timeframes médios
            if num_candles <= 100:
                self.ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            elif num_candles <= 300:
                self.ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            else:
                self.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                
        else:  # W1, MN1
            # Timeframes longos
            if num_candles <= 100:
                self.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            else:
                self.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    def plot_all(self):
        """Plota todos os elementos do gráfico"""
        # Plotar candles
        self.plot_candles()
        
        # Plotar indicadores
        self.plot_indicators()
        
        # Plotar sinais
        self.plot_signals()
        
        # Plotar áreas (se configurado)
        if hasattr(self, 'areas_config'):
            self.plot_areas(self.areas_config)
    
    def plot_candles(self):
        """Plota os candles no gráfico"""
        for idx, row in self.df.iterrows():
            self._plot_single_candle(idx, row)
    
    def _plot_single_candle(self, idx, row):
        """Plota um único candle"""
        open_val = row[self.ohlc_cols['open']]
        high_val = row[self.ohlc_cols['high']]
        low_val = row[self.ohlc_cols['low']]
        close_val = row[self.ohlc_cols['close']]
        date = row[self.datetime_col]
        
        # Determina a cor do candle
        color = self.colors['up'] if close_val >= open_val else self.colors['down']
        
        # Define cores baseadas no background
        is_dark_background = self.config.get('background') in ['black', 'dark_background', 'dark']
        
        if is_dark_background:
            line_color = 'gray'
            line_alpha = 0.8
            edge_color = 'gray'
        else:
            line_color = 'black'
            line_alpha = 1.0
            edge_color = 'black'
        
        # Plota a linha superior (high) e inferior (low) PRIMEIRO com zorder baixo
        self.ax.plot([date, date], [low_val, high_val], color=line_color, linewidth=0.5, alpha=line_alpha, zorder=1)
        
        # Plota o corpo do candle DEPOIS com zorder mais alto
        body_bottom = min(open_val, close_val)
        body_top = max(open_val, close_val)
        body_height = body_top - body_bottom
        
        # Ajusta a altura mínima do corpo para garantir que fique visível acima das linhas
        min_body_height = 0.0001  # Altura mínima para candles muito pequenos
        if body_height < min_body_height:
            body_height = min_body_height
            # Centraliza o corpo pequeno
            body_center = (open_val + close_val) / 2
            body_bottom = body_center - min_body_height / 2
            body_top = body_center + min_body_height / 2
        
        # ← USA LARGURA DINÂMICA baseada no timeframe
        rect = Rectangle(
            (mdates.date2num(date) - self.body_width/2, body_bottom),
            self.body_width,
            body_height,
            facecolor=color,
            edgecolor=edge_color,
            alpha=1,
            linewidth=0.5,  # Linha mais fina
            zorder=2
        )
        self.ax.add_patch(rect)
            
    def _adjust_to_plot_window(self, data):
        """Ajusta os dados (Series, listas, arrays) para corresponder ao DataFrame filtrado"""
        if data is None:
            return None
        
        if isinstance(data, pd.Series):
            # Mantém apenas os índices que existem no DataFrame filtrado
            return data.loc[data.index.intersection(self.df.index)]
        elif isinstance(data, (list, np.ndarray)):
            # Converte para array numpy e filtra pelos índices
            if len(data) == len(self.df):
                return data
            else:
                # Se não tem o mesmo tamanho, assume que são booleanos/máscaras
                return np.array(data)[self.df.index]
        else:
            return data


  

    def plot_signals(self):
        """Plota todos os sinais definidos no dicionário de sinais"""
        for signal_type, signal_config in self.signals.items():
            if signal_type == 'long':
                self._plot_signal(signal_config, default_type='compra', default_color='lime', signal_position='high')
            elif signal_type == 'short':
                self._plot_signal(signal_config, default_type='venda', default_color='r', signal_position='low')
            else:
                self._plot_signal(signal_config)

    def _plot_signal(self, signal_config, default_type='sinal', default_color='blue', alpha=0.8, signal_position='close'):
        signal_data = signal_config.get('data')
        signal_type = signal_config.get('signal_type', default_type)
        color = signal_config.get('color', default_color)
        alpha = signal_config.get('alpha', alpha)
        
        # ← AJUSTE PARA plot_window ↓
        signal_data = self._adjust_to_plot_window(signal_data)
        
        if signal_data is None:
            return
        
        if isinstance(signal_data, pd.Series):
            signal_dates = self.df.loc[signal_data, self.datetime_col]
            # Seleciona o preço baseado na posição do sinal
            if signal_position == 'high':
                signal_prices = self.df.loc[signal_data, self.ohlc_cols['high']]
            elif signal_position == 'low':
                signal_prices = self.df.loc[signal_data, self.ohlc_cols['low']]
            else:  # default
                signal_prices = self.df.loc[signal_data, self.ohlc_cols['close']]
        elif isinstance(signal_data, (list, np.ndarray)):
            signal_dates = self.df.loc[signal_data, self.datetime_col]
            # Seleciona o preço baseado na posição do sinal
            if signal_position == 'high':
                signal_prices = self.df.loc[signal_data, self.ohlc_cols['high']]
            elif signal_position == 'low':
                signal_prices = self.df.loc[signal_data, self.ohlc_cols['low']]
            else:  # default
                signal_prices = self.df.loc[signal_data, self.ohlc_cols['close']]
        else:
            raise ValueError("Sinal deve ser Series, lista ou array numpy")
        
        # Calcula o offset em pixels baseado no range de preços do gráfico
        y_min, y_max = self.ax.get_ylim()
        pixel_offset = (y_max - y_min) * 0.01  # 1% do range total como offset
        
        # Aplica o offset baseado no tipo de sinal
        if signal_position == 'high':
            # Compra: high + pixels
            signal_prices = signal_prices + pixel_offset
        elif signal_position == 'low':
            # Venda: low - pixels  
            signal_prices = signal_prices - pixel_offset
        
        # Plota pequenos riscos horizontais '-' em vez de marcadores
        for i, (date, price) in enumerate(zip(signal_dates, signal_prices)):
            # ← USA COMPRIMENTO DINÂMICO baseado no timeframe
            risk_length_hours = self.risk_length_hours  # ← Variável dinâmica
            
            self.ax.plot(
                [date - pd.Timedelta(hours=risk_length_hours/2), date + pd.Timedelta(hours=risk_length_hours/2)],
                [price, price],
                color=color,
                linewidth=2.5,  # Espessura do risco
                alpha=alpha,
                zorder=5,
                label=f'Sinal de {signal_type}' if i == 0 else ""  # Label apenas no primeiro
            )


    def plot_areas(self, areas_config=None):
        if areas_config is None:
            areas_config = {}
        
        for area_name, area_config in areas_config.items():
            self._plot_single_area(area_config, area_name)
            
    def _plot_single_area(self, area_config, area_name=''):
        """Plota uma área individual"""
        data = area_config.get('data')
        colors = area_config.get('colors', {})
        alpha = area_config.get('alpha', 0.1)
        label = area_config.get('label', area_name)
        
        # ← AJUSTE PARA plot_window ↓
        data = self._adjust_to_plot_window(data)
        
        if data is None or len(data) != len(self.df):
            return
            
        # Encontra as regiões onde cada condição é verdadeira
        for value, color in colors.items():
            if color is None:  # Pula valores que não devem ser plotados
                continue
                
            # Cria uma máscara para o valor específico
            mask = (data == value) if isinstance(value, (int, float)) else data
            
            if not mask.any():  # Se não há valores True, pula
                continue
            
            # Encontra os blocos contínuos da condição
            mask_filled = mask.astype(int).diff().ne(0).cumsum()
            
            for block_id in mask_filled.unique():
                block_mask = (mask_filled == block_id) & mask
                
                if not block_mask.any():
                    continue
                    
                # ← CORREÇÃO SIMPLIFICADA: Use os índices diretamente
                block_dates = self.df.loc[block_mask, self.datetime_col]
                if len(block_dates) == 0:
                    continue
                    
                start_date = block_dates.iloc[0]
                end_date = block_dates.iloc[-1]
                
                # Plota a área para este bloco
                y_min, y_max = self.ax.get_ylim()
                
                self.ax.fill_between(
                    [start_date, end_date],
                    y_min, y_max,
                    color=color,
                    alpha=alpha,
                    label=label if block_id == mask_filled.unique()[0] else "",  # Label apenas no primeiro bloco
                    zorder=1  # Atrás dos candles
                )
        
    def plot_indicators(self):
        """Plota todos os indicadores definidos no dicionário de indicadores"""
        for indicator_name, indicator_config in self.indicators.items():
            self._plot_indicator(indicator_config, label=indicator_name)

    def _plot_indicator(self, indicator_config, label=None):
        """Plota um indicador individual"""
        data = indicator_config.get('data')
        color = indicator_config.get('color', 'orange')
        linewidth = indicator_config.get('linewidth', 1.5)
        label = label or indicator_config.get('label', 'Indicador')
        
        # ← AJUSTE PARA plot_window ↓
        data = self._adjust_to_plot_window(data)
        
        if data is None:
            return
        
        if len(data) != len(self.df):
            raise ValueError("O indicador deve ter o mesmo número de pontos que o DataFrame")
        
        self.ax.plot(
            self.df[self.datetime_col],
            data,
            color=color,
            label=label,
            linewidth=linewidth,
            alpha=0.8
        )
        
    def show(self, title='Gráfico de Candlestick', legend=False, ylabel=None, xlabel=None):
        """Exibe o gráfico"""
        # Configurar cores do título baseado no background
        title_color = 'white' if self.config.get('background') in ['black', 'dark_background'] else 'black'
        
        self.ax.set_title(title, fontsize=14, fontweight='bold', color=title_color)
        
        if ylabel: 
            self.ax.set_ylabel(ylabel['name'], fontsize=ylabel.get('fontsize', 6),  # ← Mude para 6
                            color=title_color if self.config.get('background') in ['black', 'dark_background'] else 'black')
        if xlabel: 
            self.ax.set_xlabel(xlabel['name'], fontsize=xlabel.get('fontsize', 6),  # ← Mude para 6
                            color=title_color if self.config.get('background') in ['black', 'dark_background'] else 'black')
        
        if legend:
            self.ax.legend(loc='best')
        
        plt.tight_layout()
        plt.show()

# Exemplo de uso:
if __name__ == "__main__":
    import sys
    sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')
    df = pd.read_csv(r'C:\Users\Patrick\Desktop\Artaxes Portfolio\MAIN\MT5_Dados\Forex\EURUSD_D1.csv')
    if 'datetime' not in df.columns: df['datetime'] = df['date'] 
    
    # Criar sinais e indicadores
    buy_signals = df['low'] < df['close'].shift(1) * 0.995
    sell_signals = df['high'] > df['close'].shift(1) * 1.005
    
    # Calcular média móvel
    ma_period = 10
    df['MA'] = df['close'].rolling(ma_period).mean()
    
    # SIMULAR POSIÇÕES DE TRADE (exemplo)
    np.random.seed(42)
    positions = np.zeros(len(df))

    # Simula alguns trades long
    positions[100:300] = 1
    positions[500:800] = 1
    # Simula alguns trades short
    positions[1000:1050] = -1
    positions[1200:1275] = -1
    df['position'] = positions

    indicators = {
        'Média Móvel 10': {
            'data': df['MA'],
            'color': 'khaki',  # Cor mais visível em fundo escuro
            'linewidth': 0.8
        }
    }
    
    # Criar e mostrar gráfico
    chart = CandlestickChart(
        df=df,
        timeframe='D1',
        datetime_col='datetime',
        signals={'long': {'data': buy_signals}, 'short': {'data': sell_signals}},
        areas_config={'position': df['position']},
        indicators=indicators,
        plot_window=slice(-500, -200),
        figsize=(12, 6)
    )
    
    chart.show(title=None, legend=False, xlabel=None, ylabel=None)



