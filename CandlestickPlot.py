import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

class CandlestickPlot:
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
        self.signals = signals or {}
        self.indicators = indicators or {}
        
        default_areas_config = {
            'position': {
                'data': None,
                'colors': {1: 'darkgreen', -1: 'darkred', 0: None},
                'alpha': 0.3,
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
        # Plotar candles PRIMEIRO - SEMPRE
        self.plot_candles()
        
        # Plotar indicadores (que podem criar subplots)
        self.plot_indicators()
        
        # ← CORREÇÃO: Se subplots foram criados, replotar candles no gráfico principal
        if hasattr(self, 'indicator_axes_list') and self.indicator_axes_list:
            # Limpar e replotar tudo no gráfico principal
            self.ax.clear()
            self.setup_chart()  # Reconfigurar após clear
            self.plot_candles()  # Replotar candles
        
        # Plotar sinais (se existirem)
        if self.signals:
            self.plot_signals()
        
        # ← CORREÇÃO: Plotar áreas por último, com limites travados
        current_ylim = self.ax.get_ylim()
        self.ax.set_autoscale_on(False)
        
        if hasattr(self, 'areas_config'):
            self.plot_areas(self.areas_config)
        
        # Manter os limites
        self.ax.set_ylim(current_ylim)
    
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
        if not self.signals:
            return  # Nenhum sinal a plotar
        
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
        
        data = self._adjust_to_plot_window(data)
        
        if data is None or len(data) != len(self.df):
            return
            
        # ← TRAVAR OS LIMITES antes de plotar áreas ↓
        # Primeiro plote tudo (candles, indicadores, sinais)
        # Depois obtenha os limites finais e trave
        current_ylim = self.ax.get_ylim()
        
        # Agora plote as áreas com os limites travados
        for value, color in colors.items():
            if color is None:
                continue
                
            mask = (data == value) if isinstance(value, (int, float)) else data
            
            if not mask.any():
                continue
            
            mask_filled = mask.astype(int).diff().ne(0).cumsum()
            
            for block_id in mask_filled.unique():
                block_mask = (mask_filled == block_id) & mask
                
                if not block_mask.any():
                    continue
                    
                block_dates = self.df.loc[block_mask, self.datetime_col]
                if len(block_dates) == 0:
                    continue
                    
                start_date = block_dates.iloc[0]
                end_date = block_dates.iloc[-1]
                
                # ← USA OS LIMITES NORMAIS DO PLOT ↓
                y_min, y_max = current_ylim
                
                self.ax.fill_between(
                    [start_date, end_date],
                    y_min, y_max,  # Limites normais do gráfico
                    color=color,
                    alpha=alpha,
                    label=label if block_id == mask_filled.unique()[0] else "",
                    zorder=1
                )
        
        # ← MANTÉM OS LIMITES TRAVADOS ↓
        self.ax.set_ylim(current_ylim)
        
    def plot_indicators(self):
        """Plota todos os indicadores definidos no dicionário de indicadores"""
        # Criar subplots para indicadores que não plotam no gráfico principal
        self.indicator_axes = {}
        
        for indicator_name, indicator_config in self.indicators.items():
            plot_on_graph = indicator_config.get('plot_on_graph', True)
            
            if plot_on_graph:
                # Plota no gráfico principal
                self._plot_indicator(indicator_config, label=indicator_name)
            else:
                # Cria subplot separado
                self._create_indicator_subplot(indicator_config, indicator_name)

    def _plot_indicator(self, indicator_config, label=None):
        """Plota um indicador individual NO GRÁFICO PRINCIPAL"""
        data = indicator_config.get('data')
        color = indicator_config.get('color', 'orange')
        linewidth = indicator_config.get('linewidth', 1.5)
        label = label or indicator_config.get('label', 'Indicador')
        
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

    def _create_indicator_subplot(self, indicator_config, indicator_name):
        """Cria um subplot SEPARADO ABAIXO para indicador com escala diferente"""
        data = indicator_config.get('data')
        color = indicator_config.get('color', 'orange')
        linewidth = indicator_config.get('linewidth', 1.5)
        label = indicator_config.get('label', indicator_name)
        
        data = self._adjust_to_plot_window(data)
        
        if data is None or len(data) != len(self.df):
            return
        
        # ← MODIFICAÇÃO: Criar subplots com proporção 2/3 - 1/3 ↓
        if not hasattr(self, 'indicator_axes_list'):
            self.indicator_axes_list = []
            self.indicator_count = 0
            
            # Fechar a figura atual e criar uma nova com subplots
            plt.close(self.fig)
            total_indicators = len([ind for ind in self.indicators.values() if not ind.get('plot_on_graph', True)])
            
            # Calcular alturas: 2/3 para gráfico principal, 1/3 dividido entre indicadores
            main_height_ratio = 2/3  # 66.6% para gráfico principal
            indicator_height_ratio = 1/3  # 33.3% para todos os indicadores juntos
            
            # Se houver múltiplos indicadores, dividir igualmente o espaço de 1/3
            if total_indicators > 0:
                each_indicator_ratio = indicator_height_ratio / total_indicators
                height_ratios = [main_height_ratio] + [each_indicator_ratio] * total_indicators
            else:
                height_ratios = [1]
            
            # Criar figura com subplots verticais e proporções customizadas
            self.fig, axes = plt.subplots(
                total_indicators + 1, 1,  # +1 para o gráfico principal
                figsize=(12, 8),  # Altura fixa, as proporções controlam a distribuição
                sharex=True,
                gridspec_kw={'height_ratios': height_ratios}
            )
            
            # O primeiro axes é o gráfico principal
            if total_indicators > 0:
                self.ax = axes[0]
                self.indicator_axes_list = list(axes[1:])
            else:
                self.ax = axes
                self.indicator_axes_list = []
            
            # Reconfigurar o gráfico principal
            self.setup_chart()
        
        # Usar o próximo subplot disponível
        if self.indicator_count < len(self.indicator_axes_list):
            indicator_ax = self.indicator_axes_list[self.indicator_count]
            self.indicator_count += 1
            
            # Plotar no subplot
            indicator_ax.plot(
                self.df[self.datetime_col],
                data,
                color=color,
                label=label,
                linewidth=linewidth,
                alpha=0.8
            )
            
            # Configurar subplot
            indicator_ax.tick_params(axis='y', labelsize=6, colors=color)
            indicator_ax.tick_params(axis='x', labelsize=6)
            indicator_ax.grid(True, alpha=0.3)

            indicator_ax.text(
                0.02,  # 2% da esquerda
                0.98,  # 98% do topo
                indicator_name,
                transform=indicator_ax.transAxes,  # Coordenadas relativas ao axes
                fontsize=6,  # Letra bem pequena
                color=color,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor=color, linewidth=0.5)
            )
            
            # Se o background for escuro, ajustar cores
            if self.config.get('background') in ['black', 'dark_background', 'dark']:
                indicator_ax.set_facecolor('black')
                indicator_ax.tick_params(colors='white')
                indicator_ax.yaxis.label.set_color('white')
                indicator_ax.spines['bottom'].set_color('white')
                indicator_ax.spines['top'].set_color('white')
                indicator_ax.spines['right'].set_color('white')
                indicator_ax.spines['left'].set_color('white')
                
    def show(self, title='', legend=False, ylabel=None, xlabel=None):
        """Exibe o gráfico"""
        # Configurar cores do título
        title_color = 'white' if self.config.get('background') in ['black', 'dark_background'] else 'black'
        
        self.ax.set_title(title, fontsize=14, fontweight='bold', color=title_color)
        
        if ylabel: 
            self.ax.set_ylabel(ylabel['name'], fontsize=ylabel.get('fontsize', 6), 
                            color=title_color if self.config.get('background') in ['black', 'dark_background'] else 'black')
        
        # Ajustar labels de x apenas no último subplot
        if hasattr(self, 'indicator_axes_list') and self.indicator_axes_list:
            # Se há subplots, mostrar xlabel apenas no último
            last_ax = self.indicator_axes_list[-1]
            if xlabel: 
                last_ax.set_xlabel(xlabel['name'], fontsize=xlabel.get('fontsize', 6),
                                color=title_color if self.config.get('background') in ['black', 'dark_background'] else 'black')
            # Formatar datas no último subplot
            last_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        else:
            # Sem subplots, mostrar xlabel no gráfico principal
            if xlabel: 
                self.ax.set_xlabel(xlabel['name'], fontsize=xlabel.get('fontsize', 6),
                                color=title_color if self.config.get('background') in ['black', 'dark_background'] else 'black')
        
        # Travar escala final após tudo plotado
        self.ax.set_autoscale_on(False)
        
        # Calcular limites baseados nos dados reais (sem áreas)
        price_margin = 0.02  # 2% de margem
        y_min = self.df[self.ohlc_cols['low']].min()
        y_max = self.df[self.ohlc_cols['high']].max()
        y_range = y_max - y_min
        final_ymin = y_min - (y_range * price_margin)
        final_ymax = y_max + (y_range * price_margin)
        
        self.ax.set_ylim(final_ymin, final_ymax)
        
        # Adicionar legenda
        if legend:
            self.ax.legend(loc='best')
        
        # ← MODIFICAÇÃO: Ajustar layout para proporção 2/3 - 1/3 ↓
        if hasattr(self, 'indicator_axes_list') and self.indicator_axes_list:
            # Remover TODOS os espaços horizontais
            self.fig.subplots_adjust(
                left=0.03,    # 2% da esquerda
                right=0.98,   # 98% da direita  
                top=0.98,     # 5% do topo
                bottom=0.05,  # 5% da base
                hspace=0.05   # Pouco espaço entre subplots
            )
        else:
            # Sem subplots, também preencher horizontalmente
            self.fig.subplots_adjust(
                left=0.03,
                right=0.98,
                top=0.98,
                bottom=0.05
            )
        
        plt.show()

# Exemplo de uso:
if __name__ == "__main__":
    import sys
    sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')
    df = pd.read_csv(r'C:\Users\Patrick\Desktop\Artaxes Portfolio\MAIN\MT5_Dados\Forex\USDJPY_D1.csv')
    if 'datetime' not in df.columns: df['datetime'] = df['date'] 
    
    # Criar sinais e indicadores
    df['vol'] = np.log(df['high'] / df['low']).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    df['vol_avg'] = df['vol'].rolling(window=21).mean().fillna(0)

    buy_signals = ((df['close'] < df['low'].shift(1) * 0.99) & (df['vol'] > df['vol_avg'].shift(1)))
    sell_signals = ((df['close'] > df['high'].shift(1) * 1.01) & (df['vol'] > df['vol_avg'].shift(1)))
    
    # Calcular média móvel
    ma_period = 10
    df['MA'] = df['close'].rolling(ma_period).mean()

    # Versão vetorizada (mais rápida para DataFrames grandes)
    hold_period=3
    strongest_buy = buy_signals #& (df['vol'] > df['vol_avg'] * 2) & (df['close'] > df['high'].shift(1) * 1.01)
    strongest_sell = sell_signals #& (df['vol'] > df['vol_avg'] * 2) & (df['close'] < df['low'].shift(1) * 0.99)

    # Apenas 1 posição ativa por vez
    df['position'] = 0
    in_position = False
    position_end = -1

    for i in range(len(df)):
        if i <= position_end:
            continue  # Manter posição atual
        
        in_position = False
        
        if strongest_buy.iloc[i]:
            df.loc[i+1:min(i+hold_period, len(df)-1), 'position'] = 1
            position_end = min(i+hold_period, len(df)-1)
            in_position = True
        elif strongest_sell.iloc[i]:
            df.loc[i+1:min(i+hold_period, len(df)-1), 'position'] = -1
            position_end = min(i+hold_period, len(df)-1)
            in_position = True


    # Criar e mostrar gráfico
    chart = CandlestickPlot(
        df=df,
        timeframe='D1',
        datetime_col='datetime',
        signals={'long': {'data': buy_signals}, 'short': {'data': sell_signals}},
        areas_config={'position': df['position']},
        indicators={'Média Móvel 10': {'data': df['MA'], 'color': 'khaki', 'linewidth': 0.8, 'plot_on_graph': True}},
        plot_window=slice(-500, None),
        figsize=(12, 6)
    )
    
    chart.show(title=None, legend=False, xlabel=None, ylabel=None)




