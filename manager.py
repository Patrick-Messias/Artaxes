import os
import sys
import sqlite3
import json
import importlib
from typing import Dict, List, Union, Any, Type, Optional
from dataclasses import asdict

from classes import Asset, Asset_Portfolio, Strat, Strat_Parameters
from classes import TimeRules, TradeManagementRules, RiskManagementRules, ExecutionRules, DataSettings

class DatabaseManager:
    def __init__(self, db_name='artaxes_data.db'):
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Use the database in the same directory as the script
        self.db_path = os.path.join(script_dir, db_name)
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()

        # Mapeamento de classes para informações da tabela e colunas
        self._table_map = {
            Asset: {
                'table_name': 'assets',
                'pk_column': 'name',
                'columns': ['name', 'type', 'market', 'data_path', 'timeframes', 'params'],
                'serializer': self._serialize_asset,
                'deserializer': self._deserialize_asset
            },
            Asset_Portfolio: {
                'table_name': 'asset_portfolios',
                'pk_column': 'name',
                'columns': ['name', 'asset_names'],
                'serializer': self._serialize_asset_portfolio,
                'deserializer': self._deserialize_asset_portfolio
            },
            Strat: {
                'table_name': 'strats',
                'pk_column': 'name',
                'columns': ['name', 'asset_portfolio_name', 'params', 'template_type'],
                'serializer': self._serialize_strat,
                'deserializer': self._deserialize_strat
            }
        }
        
        # Adiciona mapeamento para Strategy_Template e outras estratégias
        try:
            strategies_path = os.path.join(script_dir, "Strategies")
            if strategies_path not in sys.path:
                sys.path.append(strategies_path)
                
            # Importa todas as estratégias disponíveis
            strategy_files = [f for f in os.listdir(strategies_path) if f.endswith('.py') and f != '__init__.py']
            for strategy_file in strategy_files:
                strategy_name = os.path.splitext(strategy_file)[0]
                try:
                    module = __import__(strategy_name)
                    importlib.reload(module)  # Recarrega o módulo para garantir versão mais recente
                    
                    # Procura por classes de estratégia no módulo
                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        if isinstance(item, type) and issubclass(item, Strat) and item != Strat:
                            self._table_map[item] = self._table_map[Strat].copy()
                            
                    # Se existe uma instância de estratégia, mapeia sua classe também
                    if hasattr(module, 'strategy'):
                        strategy = module.strategy
                        strategy_class = type(strategy)
                        if strategy_class not in self._table_map:
                            self._table_map[strategy_class] = self._table_map[Strat].copy()
                            
                except ImportError as e:
                    print(f"Aviso: Não foi possível importar {strategy_name}: {e}")
                    
        except Exception as e:
            print(f"Aviso: Erro ao carregar estratégias: {e}")

    def _connect(self):
        """Estabelece uma conexão com o banco de dados"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()

    def _ensure_connection(self):
        """Garante que a conexão está ativa"""
        try:
            # Tenta executar uma query simples para testar a conexão
            self.cursor.execute("SELECT 1")
        except (sqlite3.OperationalError, AttributeError):
            # Se falhar, reconecta
            self._connect()

    def _create_tables(self):
        """Cria as tabelas se não existirem"""
        # Tabela para Assets
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS assets (
                name TEXT PRIMARY KEY,
                type TEXT,
                market TEXT,
                data_path TEXT,
                timeframes TEXT,
                params TEXT
            )
        ''')
        # Tabela para Asset_Portfolios
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS asset_portfolios (
                name TEXT PRIMARY KEY,
                asset_names TEXT
            )
        ''')
        # Tabela para Strats
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS strats (
                name TEXT PRIMARY KEY,
                asset_portfolio_name TEXT,
                params TEXT,
                template_type TEXT DEFAULT 'traditional'
            )
        ''')
        
        # Verifica se a coluna template_type existe na tabela strats
        self.cursor.execute("PRAGMA table_info(strats)")
        columns = [col[1] for col in self.cursor.fetchall()]
        if 'template_type' not in columns:
            self.cursor.execute('ALTER TABLE strats ADD COLUMN template_type TEXT DEFAULT "traditional"')
        
        self.conn.commit()

    # ==================== CRUD Operations ====================

    def create(self, obj: Any) -> bool:
        """Cria um novo objeto no banco de dados"""
        try:
            self._ensure_connection()
            table_info = self._get_table_info(obj)
            
            # Verifica se já existe
            self.cursor.execute(f"SELECT 1 FROM {table_info['table_name']} WHERE {table_info['pk_column']} = ?", 
                              [getattr(obj, table_info['pk_column'])])
            if self.cursor.fetchone():
                print(f"Objeto com nome '{getattr(obj, table_info['pk_column'])}' já existe.")
                return False
            
            # Serializa e insere
            values = table_info['serializer'](obj)
            placeholders = ','.join(['?' for _ in table_info['columns']])
            self.cursor.execute(f"INSERT INTO {table_info['table_name']} ({','.join(table_info['columns'])}) VALUES ({placeholders})",
                              values)
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Erro ao criar objeto: {e}")
            return False

    def read(self, obj_type: Type[Any], name: str) -> Optional[Any]:
        """Lê um objeto do banco de dados"""
        try:
            self._ensure_connection()
            table_info = self._get_table_info(obj_type)
            
            self.cursor.execute(f"SELECT {','.join(table_info['columns'])} FROM {table_info['table_name']} WHERE {table_info['pk_column']} = ?", 
                              [name])
            row = self.cursor.fetchone()
            
            if row:
                obj = table_info['deserializer'](row)
                if obj is None:  # Se a deserialização falhou
                    print(f"Erro ao deserializar objeto do tipo {obj_type.__name__} com nome '{name}'")
                    return None
                return obj
            return None
        except Exception as e:
            print(f"Erro ao ler objeto: {e}")
            return None

    def update(self, obj: Any) -> bool:
        """Atualiza um objeto existente no banco de dados"""
        try:
            self.save(obj)
            return True
        except Exception as e:
            print(f"Erro ao atualizar objeto: {e}")
            return False

    def delete(self, obj_type: Type[Any], name: str) -> bool:
        """Remove um objeto do banco de dados"""
        info = self._get_table_info(obj_type)
        try:
            self.cursor.execute(f"DELETE FROM {info['table_name']} WHERE {info['pk_column']} = ?", (name,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Erro ao deletar objeto: {e}")
            return False

    def list_all(self, obj_type: Type[Any]) -> List[Any]:
        """Lista todos os objetos de um tipo específico"""
        try:
            info = self._get_table_info(obj_type)
            self.cursor.execute(f"SELECT {info['pk_column']} FROM {info['table_name']}")
            results = [row[0] for row in self.cursor.fetchall()]
            # Exclui Strategy_Template da listagem usando apenas o nome
            if obj_type == Strat:
                results = [r for r in results if r != 'Strategy_Template']
            return results
        except Exception as e:
            print(f"Erro ao listar objetos: {e}")
            return []

    # ==================== Detailed View Operations ====================

    def get_details(self, obj_type: Type[Any], name: str) -> Dict[str, Any]:
        """Obtém detalhes completos de um objeto"""
        try:
            # Ignora Strategy_Template
            if name == 'Strategy_Template':
                return {}
                
            obj = self.read(obj_type, name)
            if obj is None:
                return {}
            
            if isinstance(obj, Asset):
                return {
                    'name': obj.name,
                    'type': obj.type,
                    'market': obj.market,
                    'data_path': obj.data_path,
                    'timeframes': obj.timeframes_list(),
                    'params': {k: v for k, v in obj.list_values().items() 
                            if k not in ['name', 'type', 'market', 'data_path', 'data']}
                }
            elif isinstance(obj, Asset_Portfolio):
                return {
                    'name': obj.name,
                    'assets': list(obj.assets.keys()),  # Get list of asset names from dict keys
                    'stats': obj.stats()
                }
            elif isinstance(obj, Strat):  # Removido Strategy_Template da verificação
                details = {
                    'name': obj.name,
                    'type': 'template' if hasattr(obj, 'config') else 'traditional'
                }

                # Adiciona timeframe se disponível
                if hasattr(obj, 'time_rules') and hasattr(obj.time_rules, 'execution_timeframe'):
                    details['timeframe'] = obj.time_rules.execution_timeframe

                # Adiciona risco se disponível
                if hasattr(obj, 'risk_rules') and hasattr(obj.risk_rules, 'trade_risk_default'):
                    details['risk'] = obj.risk_rules.trade_risk_default

                # Adiciona portfólio se disponível
                if hasattr(obj, 'asset_portfolio') and obj.asset_portfolio:
                    details['portfolio'] = obj.asset_portfolio.name
                else:
                    details['portfolio'] = None

                # Adiciona caminho do arquivo
                details['file_path'] = os.path.join("MT5_Strategies/Strategies", f"{obj.name}.py")

                return details
            return {}
        except Exception as e:
            print(f"Erro ao obter detalhes: {e}")
            return {}

    def modify_attribute(self, obj_type: Type[Any], name: str, attribute: str, new_value: Any) -> bool:
        """Modifica um atributo específico de um objeto"""
        obj = self.read(obj_type, name)
        if obj is None:
            return False
        
        try:
            # Para objetos simples
            if hasattr(obj, attribute):
                setattr(obj, attribute, new_value)
            # Para dataclasses aninhadas (como as regras da Strat)
            elif '.' in attribute:
                parts = attribute.split('.')
                parent = obj
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_value)
            
            return self.update(obj)
        except Exception as e:
            print(f"Erro ao modificar atributo: {e}")
            return False

    # ==================== Utility Methods ====================

    def _get_table_info(self, obj_or_type: Any) -> Dict[str, Any]:
        """Obtém informações da tabela para um objeto ou tipo"""
        try:
            # Se for uma instância, pega o tipo
            obj_type = obj_or_type if isinstance(obj_or_type, type) else type(obj_or_type)
            
            # Procura a classe exata primeiro
            if obj_type in self._table_map:
                return self._table_map[obj_type]
            
            # Se não encontrar, procura por herança
            for base_type, info in self._table_map.items():
                if issubclass(obj_type, base_type):
                    # Adiciona o mapeamento para futuras consultas
                    self._table_map[obj_type] = info.copy()
                    return info
                    
            # Se ainda não encontrou, tenta importar o módulo da estratégia
            try:
                module_name = obj_type.__name__
                if module_name not in sys.modules:
                    module = __import__(module_name)
                    importlib.reload(module)
                    
                    # Se a classe herda de Strat, mapeia para o mesmo formato
                    if issubclass(obj_type, Strat):
                        self._table_map[obj_type] = self._table_map[Strat].copy()
                        return self._table_map[obj_type]
            except ImportError:
                pass
                    
            raise ValueError(f"Tipo de objeto/classe '{obj_type.__name__}' não mapeado no DatabaseManager.")
        except Exception as e:
            raise ValueError(f"Erro ao obter informações da tabela: {e}")

    # ==================== Serialization Methods ====================

    def _serialize_asset(self, asset: Asset) -> List[Any]:
        timeframes = json.dumps(asset.timeframes_list())
        params_dict = {k: v for k, v in asset.list_values().items() 
                      if k not in ['name', 'type', 'market', 'data_path', 'data']}
        params_json = json.dumps(params_dict)
        return [asset.name, asset.type, asset.market, asset.data_path, timeframes, params_json]

    def _serialize_asset_portfolio(self, portfolio: Asset_Portfolio) -> List[Any]:
        """Serializa um portfólio para armazenamento no banco de dados"""
        # Garante que estamos usando o nome correto do portfólio
        if not portfolio.name:
            raise ValueError("Nome do portfólio não pode estar vazio")
            
        # Serializa os nomes dos assets
        asset_names = json.dumps(list(portfolio.assets.keys()))
        return [portfolio.name, asset_names]

    def _serialize_strat(self, strat: Union[Strat, 'Strategy_Template']) -> List[Any]:
        """Serializa uma estratégia para armazenamento no banco de dados"""
        try:
            # Garante que a classe da estratégia está mapeada
            strat_class = type(strat)
            if strat_class not in self._table_map:
                self._table_map[strat_class] = self._table_map[Strat].copy()
            
            # Guarda apenas informações básicas
            basic_info = {
                "name": strat.name,
                "type": "template" if hasattr(strat, 'config') else "traditional",
                "timeframe": getattr(strat.time_rules, 'execution_timeframe', 'M15'),
                "risk": getattr(strat.risk_rules, 'trade_risk_default', 0.01)
            }
            
            # Adiciona informação do portfólio se existir
            portfolio_name = None
            if hasattr(strat, 'asset_portfolio') and strat.asset_portfolio:
                portfolio_name = strat.asset_portfolio.name
            basic_info["portfolio"] = portfolio_name
            
            return [strat.name, portfolio_name, json.dumps(basic_info), basic_info["type"]]
        except Exception as e:
            print(f"Erro na serialização: {e}")
            # Retorna valores padrão em caso de erro
            return [strat.name, None, json.dumps({"name": strat.name}), "traditional"]

    # ==================== Deserialization Methods ====================

    def _deserialize_asset(self, row: List[Any]) -> Asset:
        name, type, market, data_path, timeframes_str, params_json = row
        timeframe_list = json.loads(timeframes_str) if timeframes_str else []
        params_dict = json.loads(params_json) if params_json else {}
        
        asset = Asset(name=name, type=type, market=market, timeframe=timeframe_list, data_path=data_path)
        for k, v in params_dict.items():
            setattr(asset, k, v)
        return asset

    def _deserialize_asset_portfolio(self, row: List[Any]) -> Asset_Portfolio:
        """Deserializa um portfólio do banco de dados"""
        try:
            name, asset_names_str = row
            asset_names = json.loads(asset_names_str) if asset_names_str else []
            
            # Carrega os assets como um dicionário
            asset_dict = {}
            for asset_name in asset_names:
                asset = self.read(Asset, asset_name)
                if asset:
                    asset_dict[asset_name] = asset
                    
            return Asset_Portfolio(asset_portfolio_params={'name': name, 'assets': asset_dict})
        except Exception as e:
            print(f"Erro ao deserializar portfólio: {e}")
            return None

    def _deserialize_strat(self, row: List[Any]) -> Union[Strat, 'Strategy_Template']:
        """Deserializa uma estratégia do banco de dados"""
        try:
            name, portfolio_name, params_json, strat_type = row
            basic_info = json.loads(params_json)
            
            # Adiciona o diretório de estratégias ao path se necessário
            script_dir = os.path.dirname(os.path.abspath(__file__))
            strategies_dir = os.path.join(script_dir, "Strategies")
            if strategies_dir not in sys.path:
                sys.path.append(strategies_dir)
            
            # Tenta importar a estratégia do arquivo
            try:
                module = __import__(name)
                importlib.reload(module)  # Recarrega o módulo para garantir versão mais recente
                
                # Primeiro tenta obter a instância strategy
                if hasattr(module, 'strategy'):
                    strategy = module.strategy
                    strategy_class = type(strategy)
                    # Adiciona o mapeamento da classe se ainda não existir
                    if strategy_class not in self._table_map:
                        self._table_map[strategy_class] = self._table_map[Strat].copy()
                # Se não encontrar, procura pela classe com o mesmo nome
                elif hasattr(module, name):
                    strategy_class = getattr(module, name)
                    if strategy_class not in self._table_map:
                        self._table_map[strategy_class] = self._table_map[Strat].copy()
                    if hasattr(module, 'StrategyConfig'):
                        config = module.StrategyConfig(name=name)
                        strategy = strategy_class(config)
                    else:
                        strategy = strategy_class()
                else:
                    raise ImportError(f"Não foi possível encontrar a estratégia {name} no módulo")
                
                # Atualiza as informações básicas
                if hasattr(strategy, 'time_rules'):
                    strategy.time_rules.execution_timeframe = basic_info.get('timeframe', 'M15')
                if hasattr(strategy, 'risk_rules'):
                    strategy.risk_rules.trade_risk_default = basic_info.get('risk', 0.01)
                if hasattr(strategy, 'config'):
                    strategy.config.default_timeframe = basic_info.get('timeframe', 'M15')
                    strategy.config.default_risk = basic_info.get('risk', 0.01)
                if hasattr(strategy, 'strat_parameters'):
                    strategy.strat_parameters.time.execution_timeframe = basic_info.get('timeframe', 'M15')
                    strategy.strat_parameters.risk.trade_risk_default = basic_info.get('risk', 0.01)
                
                # Atualiza o portfólio se especificado
                if portfolio_name:
                    portfolio = self.read(Asset_Portfolio, portfolio_name)
                    if portfolio:
                        if hasattr(strategy, 'asset_portfolio'):
                            strategy.asset_portfolio = portfolio
                        if hasattr(strategy, 'strat_parameters'):
                            strategy.strat_parameters.assets = portfolio
                
                return strategy
                
            except (ImportError, AttributeError) as e:
                print(f"Aviso: Não foi possível carregar a estratégia {name} do arquivo: {e}")
                
                # Se não conseguir carregar do arquivo, cria uma versão básica
                portfolio = self.read(Asset_Portfolio, portfolio_name) if portfolio_name else None
                
                # Cria parâmetros básicos da estratégia
                strat_params = Strat_Parameters(
                    name=name,
                    assets=portfolio,
                    time=TimeRules(execution_timeframe=basic_info.get('timeframe', 'M15')),
                    risk=RiskManagementRules(trade_risk_default=basic_info.get('risk', 0.01)),
                    trade=TradeManagementRules(),
                    execution=ExecutionRules()
                )
                
                return Strat(strat_params=strat_params)
                
        except Exception as e:
            print(f"Erro na deserialização: {e}")
            # Retorna uma estratégia básica em caso de erro
            return Strat(Strat_Parameters(name=name or "unknown", assets=None))

    # ==================== UI Helper Methods ====================

    def display_menu(self):
        """Exibe um menu interativo para gerenciar os objetos"""
        try:
            while True:
                print("\n" + "="*50)
                print("Gerenciador de Objetos de Trading")
                print("="*50)
                print("1. Gerenciar Assets")
                print("2. Gerenciar Portfólios de Assets")
                print("3. Gerenciar Estratégias")
                print("4. Sair")
                
                choice = input("\nEscolha uma opção: ")
                
                if choice == '1':
                    self._manage_assets()
                elif choice == '2':
                    self._manage_portfolios()
                elif choice == '3':
                    self._manage_strats()
                elif choice == '4':
                    print("\nEncerrando o programa...")
                    break
                else:
                    print("Opção inválida. Tente novamente.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Realiza a limpeza e fechamento adequado do banco de dados"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.commit()  # Commit de quaisquer transações pendentes
                self.conn.close()
                print("Conexão com o banco de dados fechada com sucesso.")
        except Exception as e:
            print(f"Erro ao fechar conexão com banco de dados: {e}")
        finally:
            print("Programa encerrado.")

    def _manage_assets(self):
        """Menu para gerenciar Assets"""
        while True:
            print("\n" + "-"*50)
            print("Gerenciamento de Assets")
            print("-"*50)
            print("1. Listar Assets")
            print("2. Ver detalhes de um Asset")
            print("3. Criar novo Asset")
            print("4. Modificar Asset")
            print("5. Excluir Asset")
            print("6. Ver Timeframes Disponíveis")
            print("7. Adicionar Assets em Bulk")
            print("8. Voltar")
            
            choice = input("\nEscolha uma opção: ")
            
            if choice == '1':
                assets = self.list_all(Asset)
                print("\nAssets disponíveis:")
                for asset in assets:
                    print(f"- {asset}")
                
            elif choice == '2':
                asset_name = input("Nome do Asset: ")
                details = self.get_details(Asset, asset_name)
                if details:
                    print("\nDetalhes do Asset:")
                    for k, v in details.items():
                        print(f"{k}: {v}")
                else:
                    print("Asset não encontrado.")
                    
            elif choice == '3':
                print("\nCriar novo Asset:")
                name = input("Nome: ")
                type = input("Tipo (futures/currency_pair/stock): ")
                market = input("Mercado (ex: b3, forex, NASDAQ): ")
                data_path = input("Caminho dos dados (opcional): ")
                
                new_asset = Asset(name=name, type=type, market=market, data_path=data_path or None)
                if self.create(new_asset):
                    print("Asset criado com sucesso!")
                else:
                    print("Falha ao criar Asset.")
                    
            elif choice == '4':
                asset_name = input("Nome do Asset a modificar: ")
                if asset_name not in self.list_all(Asset):
                    print("Asset não encontrado.")
                    continue
                    
                print("\nAtributos disponíveis para modificação:")
                details = self.get_details(Asset, asset_name)
                for k in details.keys():
                    print(f"- {k}")
                
                attr = input("Atributo a modificar: ")
                if attr not in details:
                    print("Atributo inválido.")
                    continue
                    
                new_value = input(f"Novo valor para {attr} (atual: {details[attr]}): ")
                try:
                    # Tentar converter para o tipo original
                    if isinstance(details[attr], bool):
                        new_value = new_value.lower() in ('true', '1', 't', 'y', 'yes')
                    elif isinstance(details[attr], int):
                        new_value = int(new_value)
                    elif isinstance(details[attr], float):
                        new_value = float(new_value)
                    elif isinstance(details[attr], list):
                        new_value = [x.strip() for x in new_value.split(',')]
                except ValueError:
                    print("Valor inválido para o tipo do atributo.")
                    continue
                
                if self.modify_attribute(Asset, asset_name, attr, new_value):
                    print("Atributo modificado com sucesso!")
                else:
                    print("Falha ao modificar atributo.")
                    
            elif choice == '5':
                asset_name = input("Nome do Asset a excluir: ")
                if self.delete(Asset, asset_name):
                    print("Asset excluído com sucesso!")
                else:
                    print("Falha ao excluir Asset.")
                    
            elif choice == '6':
                assets = self.list_all(Asset)
                if not assets:
                    print("\nNenhum Asset cadastrado.")
                    continue

                print("\nAssets disponíveis:")
                for i, asset_name in enumerate(assets, 1):
                    print(f"{i}. {asset_name}")

                selected = input("\nEscolha o número do Asset (ou ENTER para ver todos): ").strip()
                
                if selected:
                    try:
                        idx = int(selected) - 1
                        if 0 <= idx < len(assets):
                            asset = self.read(Asset, assets[idx])
                            if asset:
                                print(f"\nTimeframes disponíveis para {asset.name}:")
                                asset.timeframes_load_available()
                                timeframes = asset.timeframes_list()
                                if timeframes:
                                    for tf in timeframes:
                                        print(f"- {tf}")
                                else:
                                    print("Nenhum timeframe encontrado.")
                        else:
                            print("Número inválido.")
                    except ValueError:
                        print("Entrada inválida.")
                else:
                    print("\nTimeframes disponíveis para todos os Assets:")
                    for asset_name in assets:
                        asset = self.read(Asset, asset_name)
                        if asset:
                            print(f"\n{asset.name}:")
                            asset.timeframes_load_available()
                            timeframes = asset.timeframes_list()
                            if timeframes:
                                for tf in timeframes:
                                    print(f"- {tf}")
                            else:
                                print("Nenhum timeframe encontrado.")

            elif choice == '7':
                print("\nAdicionar Assets em Bulk:")
                directory = input("Caminho dos dados (ex: MT5_Dados): ")
                directory = os.path.join(os.getcwd(), directory)
                type = input("Tipo para todos os assets (futures/currency_pair/stock): ")
                market = input("Mercado para todos os assets (ex: b3, forex, NASDAQ): ")

                count = self.bulk_add_assets(directory, type, market)
                if count > 0:
                    print(f"\nForam adicionados {count} assets com sucesso!")
                else:
                    print("\nNenhum asset foi adicionado.")
                    
            elif choice == '8':
                break
            else:
                print("Opção inválida. Tente novamente.")

    def _manage_portfolios(self):
        """Menu para gerenciar Portfólios"""
        while True:
            print("\n" + "-"*50)
            print("Gerenciamento de Portfólios")
            print("-"*50)
            print("1. Listar Portfólios")
            print("2. Ver detalhes de um Portfólio")
            print("3. Criar novo Portfólio")
            print("4. Modificar Portfólio")
            print("5. Excluir Portfólio")
            print("6. Voltar")
            
            choice = input("\nEscolha uma opção: ")
            
            if choice == '1':
                portfolios = self.list_all(Asset_Portfolio)
                print("\nPortfólios disponíveis:")
                for port in portfolios:
                    print(f"- {port}")
                
            elif choice == '2':
                port_name = input("Nome do Portfólio: ")
                details = self.get_details(Asset_Portfolio, port_name)
                if details:
                    print("\nDetalhes do Portfólio:")
                    for k, v in details.items():
                        print(f"{k}: {v}")
                else:
                    print("Portfólio não encontrado.")
                    
            elif choice == '3':
                print("\nCriar novo Portfólio:")
                portfolio_name = input("Nome do Portfólio: ").strip()
                if not portfolio_name:
                    print("Nome do portfólio não pode estar vazio")
                    continue
                
                # Lista assets disponíveis para adicionar
                assets = self.list_all(Asset)
                if not assets:
                    print("Nenhum Asset disponível. Crie Assets primeiro.")
                    continue
                    
                print("\nAssets disponíveis:")
                for i, asset in enumerate(assets, 1):
                    print(f"{i}. {asset}")
                
                selected = input("\nNúmeros dos Assets a incluir (separados por vírgula): ")
                try:
                    indices = [int(x.strip())-1 for x in selected.split(',')]
                    asset_names = [assets[i] for i in indices if 0 <= i < len(assets)]
                except:
                    print("Seleção inválida.")
                    continue
                
                # Carrega os objetos Asset
                asset_objs = {}  # Changed from list to dict
                for name in asset_names:
                    asset = self.read(Asset, name)
                    if asset:
                        asset_objs[name] = asset  # Store as dict with name as key
                
                if not asset_objs:
                    print("Nenhum Asset válido selecionado.")
                    continue
                
                new_port = Asset_Portfolio(asset_portfolio_params={'name': portfolio_name, 'assets': asset_objs})
                if self.create(new_port):
                    print(f"Portfólio '{portfolio_name}' criado com sucesso!")
                else:
                    print("Falha ao criar Portfólio.")
                    
            elif choice == '4':
                port_name = input("Nome do Portfólio a modificar: ")
                if port_name not in self.list_all(Asset_Portfolio):
                    print("Portfólio não encontrado.")
                    continue
                    
                print("\nOpções de modificação:")
                print("1. Adicionar Asset")
                print("2. Remover Asset")
                
                sub_choice = input("Escolha: ")
                
                if sub_choice == '1':
                    # Lista assets disponíveis
                    current_port = self.read(Asset_Portfolio, port_name)
                    current_assets = list(current_port.assets.keys())
                    
                    all_assets = self.list_all(Asset)
                    available = [a for a in all_assets if a not in current_assets]
                    
                    if not available:
                        print("Todos os Assets já estão no portfólio.")
                        continue
                        
                    print("\nAssets disponíveis para adicionar:")
                    for i, asset in enumerate(available, 1):
                        print(f"{i}. {asset}")
                    
                    selected = input("Número do Asset a adicionar: ")
                    try:
                        idx = int(selected)-1
                        if 0 <= idx < len(available):
                            asset_name = available[idx]
                            asset = self.read(Asset, asset_name)
                            if asset:
                                current_port.assets[asset_name] = asset
                                if self.update(current_port):
                                    print("Asset adicionado com sucesso!")
                                else:
                                    print("Falha ao atualizar Portfólio.")
                        else:
                            print("Número inválido.")
                    except:
                        print("Entrada inválida.")
                        
                elif sub_choice == '2':
                    current_port = self.read(Asset_Portfolio, port_name)
                    if not current_port.assets:
                        print("Portfólio vazio.")
                        continue
                        
                    print("\nAssets no portfólio:")
                    asset_names = list(current_port.assets.keys())
                    for i, asset_name in enumerate(asset_names, 1):
                        print(f"{i}. {asset_name}")
                    
                    selected = input("Número do Asset a remover: ")
                    try:
                        idx = int(selected)-1
                        if 0 <= idx < len(asset_names):
                            asset_name = asset_names[idx]
                            removed_asset = current_port.assets.pop(asset_name)
                            if self.update(current_port):
                                print(f"Asset {removed_asset.name} removido com sucesso!")
                            else:
                                print("Falha ao atualizar Portfólio.")
                        else:
                            print("Número inválido.")
                    except:
                        print("Entrada inválida.")
                else:
                    print("Opção inválida.")
                    
            elif choice == '5':
                port_name = input("Nome do Portfólio a excluir: ")
                if self.delete(Asset_Portfolio, port_name):
                    print("Portfólio excluído com sucesso!")
                else:
                    print("Falha ao excluir Portfólio.")
                    
            elif choice == '6':
                break
            else:
                print("Opção inválida. Tente novamente.")

    def _manage_strats(self):
        """Menu para gerenciar Estratégias"""
        while True:
            print("\n" + "-"*50)
            print("Gerenciamento de Estratégias")
            print("-"*50)
            print("1. Listar Estratégias")
            print("2. Ver detalhes de uma Estratégia")
            print("3. Criar nova Estratégia")
            print("4. Modificar Estratégia")
            print("5. Excluir Estratégia")
            print("6. Voltar")
            
            choice = input("\nEscolha uma opção: ")
            
            if choice == '1':
                try:
                    strats = self.list_all(Strat)
                    if not strats:
                        print("\nNenhuma estratégia encontrada.")
                    else:
                        print("\nEstratégias disponíveis:")
                        for strat in strats:
                            try:
                                details = self.get_details(Strat, strat)
                                strat_type = details.get('type', 'traditional')
                                print(f"- {strat} ({strat_type})")
                            except Exception as e:
                                print(f"- {strat} (erro ao obter detalhes: {e})")
                except Exception as e:
                    print(f"Erro ao listar estratégias: {e}")
                    
            elif choice == '2':
                try:
                    strat_name = input("Nome da Estratégia: ")
                    # Busca diretamente no banco de dados
                    self._ensure_connection()
                    self.cursor.execute("SELECT params FROM strats WHERE name = ?", (strat_name,))
                    row = self.cursor.fetchone()
                    if row and row[0]:
                        details = json.loads(row[0])
                        print("\nDetalhes da Estratégia:")
                        for k, v in details.items():
                            print(f"{k}: {v}")
                    else:
                        print("Estratégia não encontrada.")
                except Exception as e:
                    print(f"Erro ao obter detalhes da estratégia: {e}")
                    
            elif choice == '3':
                print("\nCriar nova Estratégia:")
                print("1. Criar do template")
                print("2. Carregar estratégia existente")
                
                creation_choice = input("\nEscolha uma opção: ")
                
                try:
                    name = input("Nome da estratégia: ")
                    if not name:
                        print("Nome da estratégia é obrigatório.")
                        continue
                    
                    if name == "Strategy_Template":
                        print("Erro: 'Strategy_Template' é o nome do template e não pode ser usado como nome de estratégia.")
                        continue
                    
                    # Usa o diretório de estratégias relativo ao script
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    strategies_dir = os.path.join(script_dir, "Strategies")
                    strategy_path = os.path.join(strategies_dir, f"{name}.py")
                    
                    if creation_choice == '1':
                        # Copia o template
                        template_path = os.path.join(strategies_dir, "Strategy_Template.py")
                        if not os.path.exists(template_path):
                            print("Erro: Template não encontrado.")
                            continue
                            
                        with open(template_path, 'r') as source, open(strategy_path, 'w') as target:
                            content = source.read()
                            content = content.replace("Strategy_Template", name)
                            content = content.replace('name: str = "Strategy_Template"', f'name: str = "{name}"')
                            target.write(content)
                            
                    elif creation_choice == '2':
                        path = input("Path completo do arquivo da estratégia: ").strip()
                        if not os.path.exists(path):
                            print("Arquivo não encontrado.")
                            continue
                            
                        # Copia o arquivo para o diretório de estratégias
                        import shutil
                        shutil.copy2(path, strategy_path)
                    else:
                        print("Opção inválida.")
                        continue
                    
                    # Adiciona o diretório ao path se necessário
                    if strategies_dir not in sys.path:
                        sys.path.append(strategies_dir)
                    
                    # Importa a estratégia
                    try:
                        module = __import__(name)
                        importlib.reload(module)  # Recarrega o módulo para garantir que temos a versão mais recente
                        strategy = getattr(module, 'strategy', None)
                        if strategy is None:
                            print(f"Erro: Não foi possível encontrar a estratégia no módulo {name}")
                            continue
                        
                        # Adiciona o mapeamento da classe ao DatabaseManager
                        strategy_class = type(strategy)
                        if strategy_class not in self._table_map:
                            self._table_map[strategy_class] = self._table_map[Strat]
                        
                        # Salva no banco de dados
                        if self.create(strategy):
                            print("Estratégia registrada com sucesso no banco de dados!")
                        else:
                            print("Aviso: Não foi possível registrar a estratégia no banco de dados.")
                            
                    except Exception as e:
                        print(f"Erro ao carregar estratégia: {e}")
                        
                except Exception as e:
                    print(f"Erro ao processar estratégia: {e}")
                    
            elif choice == '4':
                strat_name = input("Nome da Estratégia a modificar: ")
                self._ensure_connection()
                self.cursor.execute("SELECT params FROM strats WHERE name = ?", (strat_name,))
                row = self.cursor.fetchone()
                if not row:
                    print("Estratégia não encontrada.")
                    continue
                
                print("\nOpções de modificação:")
                print("1. Alterar timeframe padrão")
                print("2. Alterar risco padrão")
                print("3. Alterar portfólio")
                print("4. Recarregar do arquivo")
                
                sub_choice = input("\nEscolha uma opção: ")
                
                try:
                    # Carrega a estratégia
                    strat = self.read(Strat, strat_name)
                    if not strat:
                        print("Erro ao carregar estratégia.")
                        continue
                    
                    if sub_choice == '1':
                        new_tf = input("Novo timeframe (ex: M5, M15, H1): ")
                        if hasattr(strat, 'time_rules'):
                            strat.time_rules.execution_timeframe = new_tf
                            # Atualiza também o config se existir
                            if hasattr(strat, 'config'):
                                strat.config.default_timeframe = new_tf
                            # Força atualização dos parâmetros
                            if hasattr(strat, 'strat_parameters'):
                                strat.strat_parameters.time.execution_timeframe = new_tf
                            
                            # Atualiza no banco de dados
                            try:
                                if self.update(strat):
                                    print("Timeframe atualizado com sucesso!")
                                else:
                                    print("Erro ao atualizar timeframe.")
                            except Exception as e:
                                print(f"Erro ao salvar no banco de dados: {e}")
                                
                    elif sub_choice == '2':
                        try:
                            new_risk = float(input("Novo risco padrão (ex: 0.01 para 1%): "))
                            if hasattr(strat, 'risk_rules'):
                                strat.risk_rules.trade_risk_default = new_risk
                                # Atualiza também o config se existir
                                if hasattr(strat, 'config'):
                                    strat.config.default_risk = new_risk
                                # Força atualização dos parâmetros
                                if hasattr(strat, 'strat_parameters'):
                                    strat.strat_parameters.risk.trade_risk_default = new_risk
                                
                                if self.update(strat):
                                    print("Risco atualizado com sucesso!")
                                else:
                                    print("Erro ao atualizar risco.")
                        except ValueError:
                            print("Por favor, insira um número válido")
                            
                    elif sub_choice == '3':
                        portfolios = self.list_all(Asset_Portfolio)
                        if not portfolios:
                            print("Nenhum portfólio disponível.")
                            continue
                            
                        print("\nPortfólios disponíveis:")
                        for i, port in enumerate(portfolios, 1):
                            print(f"{i}. {port}")
                            
                        try:
                            idx = int(input("\nEscolha o número do portfólio: ")) - 1
                            if 0 <= idx < len(portfolios):
                                portfolio = self.read(Asset_Portfolio, portfolios[idx])
                                if portfolio:
                                    # Atualiza o portfólio em todos os lugares necessários
                                    if hasattr(strat, 'asset_portfolio'):
                                        strat.asset_portfolio = portfolio
                                    if hasattr(strat, 'strat_parameters'):
                                        strat.strat_parameters.assets = portfolio
                                    
                                    if self.update(strat):
                                        print("Portfólio atualizado com sucesso!")
                                    else:
                                        print("Erro ao atualizar portfólio.")
                                else:
                                    print("Erro ao carregar portfólio.")
                            else:
                                print("Número inválido")
                        except ValueError:
                            print("Por favor, insira um número válido")
                            
                    elif sub_choice == '4':
                        # Recarrega a estratégia do arquivo
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        strategies_dir = os.path.join(script_dir, "Strategies")
                        if strategies_dir not in sys.path:
                            sys.path.append(strategies_dir)
                            
                        try:
                            # Força recarregar o módulo
                            module = __import__(strat_name)
                            importlib.reload(module)
                            
                            # Primeiro tenta obter a instância strategy do módulo
                            if hasattr(module, 'strategy'):
                                new_strategy = module.strategy
                            # Se não encontrar, tenta criar uma nova instância
                            elif hasattr(module, strat_name):
                                strategy_class = getattr(module, strat_name)
                                if hasattr(module, 'StrategyConfig'):
                                    config = module.StrategyConfig(name=strat_name)
                                    new_strategy = strategy_class(config)
                                else:
                                    new_strategy = strategy_class()
                            else:
                                print(f"Erro: Não foi possível encontrar a estratégia no módulo {strat_name}")
                                continue
                            
                            # Garante que a classe está mapeada
                            strategy_class = type(new_strategy)
                            if strategy_class not in self._table_map:
                                self._table_map[strategy_class] = self._table_map[Strat].copy()
                            
                            # Preserva as configurações atuais antes de sobrescrever
                            if hasattr(strat, 'asset_portfolio') and strat.asset_portfolio:
                                new_strategy.asset_portfolio = strat.asset_portfolio
                                if hasattr(new_strategy, 'strat_parameters'):
                                    new_strategy.strat_parameters.assets = strat.asset_portfolio
                            
                            if hasattr(strat, 'time_rules') and hasattr(new_strategy, 'time_rules'):
                                new_strategy.time_rules.execution_timeframe = strat.time_rules.execution_timeframe
                                if hasattr(new_strategy, 'config'):
                                    new_strategy.config.default_timeframe = strat.time_rules.execution_timeframe
                                if hasattr(new_strategy, 'strat_parameters'):
                                    new_strategy.strat_parameters.time.execution_timeframe = strat.time_rules.execution_timeframe
                            
                            if hasattr(strat, 'risk_rules') and hasattr(new_strategy, 'risk_rules'):
                                new_strategy.risk_rules.trade_risk_default = strat.risk_rules.trade_risk_default
                                if hasattr(new_strategy, 'config'):
                                    new_strategy.config.default_risk = strat.risk_rules.trade_risk_default
                                if hasattr(new_strategy, 'strat_parameters'):
                                    new_strategy.strat_parameters.risk.trade_risk_default = strat.risk_rules.trade_risk_default
                            
                            # Garante que o nome está correto
                            new_strategy.name = strat_name
                            if hasattr(new_strategy, 'config'):
                                new_strategy.config.name = strat_name
                            if hasattr(new_strategy, 'strat_parameters'):
                                new_strategy.strat_parameters.name = strat_name
                            
                            # Remove a estratégia antiga do banco de dados
                            self._ensure_connection()
                            self.cursor.execute("DELETE FROM strats WHERE name = ?", (strat_name,))
                            self.conn.commit()
                            
                            # Salva a nova estratégia no banco de dados
                            if self.create(new_strategy):
                                print("Estratégia recarregada com sucesso!")
                            else:
                                print("Erro ao recarregar estratégia.")
                                
                        except Exception as e:
                            print(f"Erro ao recarregar estratégia: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print("Opção inválida")
                        
                except Exception as e:
                    print(f"Erro ao modificar estratégia: {e}")
                    import traceback
                    traceback.print_exc()
            elif choice == '5':
                strat_name = input("Nome da Estratégia a excluir: ")
                if strat_name == "Strategy_Template":
                    print("Erro: Não é possível excluir o template.")
                    continue
                    
                confirm = input(f"Tem certeza que deseja excluir a estratégia {strat_name}? (s/N): ")
                if confirm.lower() == 's':
                    if self.delete(Strat, strat_name):
                        # Também remove o arquivo .py se existir
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        strategy_path = os.path.join(script_dir, "Strategies", f"{strat_name}.py")
                        try:
                            if os.path.exists(strategy_path):
                                os.remove(strategy_path)
                        except Exception as e:
                            print(f"Aviso: Não foi possível remover o arquivo da estratégia: {e}")
                        print("Estratégia excluída com sucesso!")
                    else:
                        print("Falha ao excluir Estratégia.")
                    
            elif choice == '6':
                break
            else:
                print("Opção inválida. Tente novamente.")

    def close(self):
        """Fecha a conexão com o banco de dados"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def __del__(self):
        """Destrutor que garante que a conexão seja fechada"""
        self.close()

    def save(self, obj: Any) -> bool:
        """Salva um objeto no banco de dados"""
        self._ensure_connection()
        try:
            info = self._get_table_info(obj)
            values = info['serializer'](obj)
            
            # Verifica se o objeto já existe
            self.cursor.execute(f"SELECT 1 FROM {info['table_name']} WHERE {info['pk_column']} = ?", (values[0],))
            exists = self.cursor.fetchone() is not None
            
            if exists:
                # Update
                set_clause = ", ".join(f"{col} = ?" for col in info['columns'][1:])
                self.cursor.execute(
                    f"UPDATE {info['table_name']} SET {set_clause} WHERE {info['pk_column']} = ?",
                    values[1:] + [values[0]]
                )
            else:
                # Insert
                placeholders = ", ".join("?" * len(info['columns']))
                self.cursor.execute(
                    f"INSERT INTO {info['table_name']} ({', '.join(info['columns'])}) VALUES ({placeholders})",
                    values
                )
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Erro ao salvar objeto: {e}")
            return False

    def load(self, obj_type: Type[Any], name: str) -> Optional[Any]:
        """Carrega um objeto do banco de dados pelo nome"""
        try:
            info = self._get_table_info(obj_type)
            
            # Busca os dados do objeto no banco
            query = f"SELECT {','.join(info['columns'])} FROM {info['table_name']} WHERE {info['pk_column']} = ?"
            self.cursor.execute(query, (name,))
            row = self.cursor.fetchone()
            
            if row is None:
                return None
                
            # Deserializa os dados para um objeto
            return info['deserializer'](row)
            
        except Exception as e:
            print(f"Erro ao carregar objeto: {e}")
            return None

    def bulk_add_assets(self, directory: str, asset_type: str, market: str) -> int:
        """
        Adiciona múltiplos assets de uma pasta, todos com o mesmo tipo e mercado.
        Retorna o número de assets adicionados com sucesso.
        """
        try:
            if not os.path.exists(directory):
                print(f"Diretório não encontrado: {directory}")
                return 0

            # Encontra todos os arquivos de dados
            files = os.listdir(directory)
            unique_assets = set()

            # Extrai nomes únicos de ativos dos arquivos
            for file in files:
                if file.endswith(('.csv', '.xlsx', '.xls')):
                    asset_name = file.split('_')[0]  # Assume formato: NOME_TIMEFRAME.extensão
                    unique_assets.add(asset_name)

            if not unique_assets:
                print("Nenhum arquivo de dados encontrado no formato correto.")
                return 0

            # Cria assets para cada nome único encontrado
            success_count = 0
            for asset_name in unique_assets:
                try:
                    new_asset = Asset(
                        name=asset_name,
                        type=asset_type,
                        market=market,
                        data_path=directory
                    )
                    if self.create(new_asset):
                        success_count += 1
                        print(f"Asset {asset_name} criado com sucesso.")
                    else:
                        print(f"Falha ao criar asset {asset_name}.")
                except Exception as e:
                    print(f"Erro ao criar asset {asset_name}: {e}")

            return success_count

        except Exception as e:
            print(f"Erro ao adicionar assets em bulk: {e}")
            return 0


if __name__ == "__main__":

    db = DatabaseManager()
    db.display_menu()



