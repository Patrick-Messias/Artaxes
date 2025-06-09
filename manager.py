import os
import sys
import sqlite3
import json
import importlib
from typing import Dict, List, Union, Any, Type, Optional
from dataclasses import asdict

from classes import Asset, Asset_Portfolio, Strat, Strat_Parameters, Money_Management_Algorithm, BaseClass
from classes import TimeRules, TradeManagementRules, RiskManagementRules, ExecutionRules, DataSettings

# Add Money Management Algo directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Money Management Algo'))
from mm_template import MM_Template

class DatabaseManager:
    def __init__(self, db_name='artaxes_data.db'):
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Use the database in the same directory as the script
        self.db_path = os.path.join(script_dir, db_name)
        self.conn = None
        self.cursor = None
        self._connect()
        
        # Define table mappings
        self._table_map = {
            Asset: {
                'table_name': 'assets',
                'serialize': self._serialize_asset,
                'deserialize': self._deserialize_asset
            },
            Asset_Portfolio: {
                'table_name': 'portfolios',
                'serialize': self._serialize_asset_portfolio,
                'deserialize': self._deserialize_asset_portfolio
            },
            Strat: {
                'table_name': 'strats',
                'serialize': self._serialize_strat,
                'deserialize': self._deserialize_strat
            },
            Money_Management_Algorithm: {
                'table_name': 'money_management',
                'serialize': self._serialize_money_management,
                'deserialize': self._deserialize_money_management
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
            
        # Adiciona mapeamento para MM_Template e outros MMs
        try:
            mm_path = os.path.join(script_dir, "Money Management Algo")
            if mm_path not in sys.path:
                sys.path.append(mm_path)
                
            # Importa todos os MMs disponíveis
            mm_files = [f for f in os.listdir(mm_path) if f.endswith('.py') and f != '__init__.py']
            for mm_file in mm_files:
                mm_name = os.path.splitext(mm_file)[0]
                try:
                    module = __import__(mm_name)
                    importlib.reload(module)  # Recarrega o módulo para garantir versão mais recente
                    
                    # Procura por classes de MM no módulo
                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        if isinstance(item, type) and issubclass(item, Money_Management_Algorithm) and item != Money_Management_Algorithm:
                            self._table_map[item] = self._table_map[Money_Management_Algorithm].copy()
                            
                    # Se existe uma instância de MM, mapeia sua classe também
                    if hasattr(module, 'money_management'):
                        mm = module.money_management
                        mm_class = type(mm)
                        if mm_class not in self._table_map:
                            self._table_map[mm_class] = self._table_map[Money_Management_Algorithm].copy()
                            
                except ImportError as e:
                    print(f"Aviso: Não foi possível importar {mm_name}: {e}")
                    
        except Exception as e:
            print(f"Aviso: Erro ao carregar Money Management: {e}")
            
        self._create_tables()

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
        """Create necessary tables if they don't exist"""
        self._ensure_connection()
        
        with self.conn:
            # Create assets table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS assets (
                    name TEXT PRIMARY KEY,
                    type TEXT,
                    market TEXT,
                    data_path TEXT,
                    params TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create portfolios table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    name TEXT PRIMARY KEY,
                    assets TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create strats table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS strats (
                    name TEXT PRIMARY KEY,
                    template_path TEXT,
                    params TEXT,
                    template_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create money_management table
            self.cursor.execute('''
                DROP TABLE IF EXISTS money_management
            ''')
            self.cursor.execute('''
                CREATE TABLE money_management (
                    name TEXT PRIMARY KEY,
                    params TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()

    # ==================== CRUD Operations ====================

    def create(self, obj: Any) -> bool:
        """Create a new object in the database"""
        try:
            self._ensure_connection()
            
            # Get table info
            table_info = self._get_table_info(obj)
            print(f"Creating object in table {table_info['table_name']}")
            
            # Check if object already exists
            self.cursor.execute(f"SELECT * FROM {table_info['table_name']} WHERE name = ?", (obj.name,))
            if self.cursor.fetchone():
                print(f"Object {obj.name} already exists in {table_info['table_name']}")
                return False
                
            # Serialize and insert
            values = table_info['serialize'](obj)
            print(f"Serialized values: {values}")
            
            placeholders = ','.join(['?' for _ in table_info['columns']])
            query = f"INSERT INTO {table_info['table_name']} ({','.join(table_info['columns'])}) VALUES ({placeholders})"
            print(f"Insert query: {query}")
            print(f"Values to insert: {values}")
            
            self.cursor.execute(query, values)
            self.conn.commit()
            print(f"Object {obj.name} created successfully in {table_info['table_name']}")
            return True
            
        except Exception as e:
            print(f"Error creating object: {e}")
            import traceback
            traceback.print_exc()
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
                obj = table_info['deserialize'](row)
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
        """List all objects of a given type from the database"""
        try:
            self._ensure_connection()
            table_info = self._get_table_info(obj_type)
            
            self.cursor.execute(f"SELECT * FROM {table_info['table_name']}")
            rows = self.cursor.fetchall()
            
            result = []
            for row in rows:
                try:
                    obj = table_info['deserialize'](row[:len(table_info['columns'])])  # Limita ao número correto de colunas
                    if obj:
                        result.append(obj)
                except Exception as e:
                    print(f"Error deserializing row {row}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Display objects in a user-friendly format only for portfolios
            if obj_type == Asset_Portfolio:
                print("\nPortfólios disponíveis:")
                for i, portfolio in enumerate(result):
                    print(f"{i} - {portfolio.name}")
            elif obj_type == Asset:
                print("\nAssets disponíveis:")
                for i, obj in enumerate(result):
                    print(f"{i} - {obj.name}")
            
            # Return only the names for assets, full objects for others
            if obj_type == Asset:
                return [obj.name for obj in result]
            return result
            
        except Exception as e:
            print(f"Error listing objects: {e}")
            import traceback
            traceback.print_exc()
            return []

    # ==================== Detailed View Operations ====================

    def get_details(self, obj_type: Type[Any], name: str) -> Dict[str, Any]:
        """Get detailed information about an object"""
        self._ensure_connection()
        
        try:
            table_info = self._get_table_info(obj_type)
            table_name = table_info['table_name']
            pk_column = table_info['pk_column']
            
            query = f"SELECT * FROM {table_name} WHERE {pk_column} = ?"
            self.cursor.execute(query, (name,))
            row = self.cursor.fetchone()
            
            if row:
                columns = table_info['columns']
                return {columns[i]: row[i] for i in range(len(columns))}
            return None
            
        except Exception as e:
            print(f"Erro ao obter detalhes: {e}")
            return None

    def modify_attribute(self, obj_type: Type[Any], name: str, attribute: str = None, new_value: Any = None) -> bool:
        """Modify a specific attribute of an object"""
        try:
            self._ensure_connection()
            table_info = self._get_table_info(obj_type)
            
            # Verifica se o objeto existe
            self.cursor.execute(f"SELECT * FROM {table_info['table_name']} WHERE name = ?", (name,))
            row = self.cursor.fetchone()
            if not row:
                print(f"Object {name} not found")
                return False
            
            # Carrega o objeto atual
            obj = table_info['deserialize'](row)
            if not obj:
                print(f"Error loading object {name}")
                return False
                
            if obj_type == Asset:
                # Lista os atributos disponíveis
                print("\nAtributos disponíveis para modificação:")
                print("1. type")
                print("2. market")
                print("3. data_path")
                print("4. tick")
                print("5. tick_fin_val")
                print("6. lot_value")
                print("7. min_lot")
                print("8. leverage")
                print("9. comissions")
                print("10. slippage")
                print("11. spread")
                
                if attribute is None:
                    choice = input("\nEscolha o número do atributo a modificar: ")
                    attr_map = {
                        '1': 'type',
                        '2': 'market',
                        '3': 'data_path',
                        '4': 'tick',
                        '5': 'tick_fin_val',
                        '6': 'lot_value',
                        '7': 'min_lot',
                        '8': 'leverage',
                        '9': 'comissions',
                        '10': 'slippage',
                        '11': 'spread'
                    }
                    
                    if choice not in attr_map:
                        print("Opção inválida")
                        return False
                    
                    attribute = attr_map[choice]
                    new_value = input(f"Novo valor para {attribute}: ")
                
                # Converte o valor para o tipo correto
                if attribute in ['tick', 'tick_fin_val', 'lot_value', 'min_lot', 'leverage', 'comissions', 'slippage', 'spread']:
                    try:
                        new_value = float(new_value)
                    except ValueError:
                        print("Valor inválido. Digite um número.")
                        return False
                
                # Atualiza o atributo
                setattr(obj, attribute, new_value)
                
                # Serializa e salva
                values = self._serialize_asset(obj)
                self.cursor.execute(
                    "UPDATE assets SET type = ?, market = ?, data_path = ?, params = ? WHERE name = ?",
                    (values[1], values[2], values[3], values[4], values[0])
                )
                
                self.conn.commit()
                print(f"\nAtributo {attribute} atualizado com sucesso!")
                return True
                
            else:
                # Para outros tipos de objetos
                if attribute is None:
                    print("Atributo não especificado")
                    return False
                
                self.cursor.execute(
                    f"UPDATE {table_info['table_name']} SET {attribute} = ? WHERE name = ?",
                    (new_value, name)
                )
                
                self.conn.commit()
                return True
                
        except Exception as e:
            print(f"Error updating object: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ==================== Utility Methods ====================

    def _get_table_info(self, obj_or_type: Any) -> Dict[str, Any]:
        """Get table information for an object or type"""
        # Se for uma instância, pega seu tipo
        obj_type = obj_or_type if isinstance(obj_or_type, type) else type(obj_or_type)
        
        # Procura a classe base mais próxima que está mapeada
        for base in obj_type.__mro__:
            if base in self._table_map:
                info = self._table_map[base].copy()
                
                # Adiciona informações específicas para cada tipo
                if base == Asset:
                    info['pk_column'] = 'name'
                    info['columns'] = ['name', 'type', 'market', 'data_path', 'params']
                elif base == Asset_Portfolio:
                    info['pk_column'] = 'name'
                    info['columns'] = ['name', 'assets']
                elif base == Strat:
                    info['pk_column'] = 'name'
                    info['columns'] = ['name', 'template_path', 'params', 'template_type']
                elif base == Money_Management_Algorithm:
                    info['pk_column'] = 'name'
                    info['columns'] = ['name', 'params']
                    
                # Adiciona debug para verificar o que está sendo retornado
#                print(f"Table info for {obj_type.__name__}: {info}")
                return info
                
        raise ValueError(f"Unsupported object type: {obj_type}")

    # ==================== Serialization Methods ====================

    def _serialize_asset(self, asset: Asset) -> List[Any]:
        """Serializa um asset para armazenamento no banco de dados"""
        # Serializa os timeframes
        timeframes = json.dumps(asset.timeframes_list())
        
        # Serializa os parâmetros do asset
        params_dict = {
            "tick": asset.tick,
            "tick_fin_val": asset.tick_fin_val,
            "lot_value": asset.lot_value,
            "min_lot": asset.min_lot,
            "leverage": asset.leverage,
            "comissions": asset.comissions,
            "slippage": asset.slippage,
            "spread": asset.spread
        }
        params_json = json.dumps(params_dict)
        
        # Retorna os valores na ordem correta das colunas
        return [
            asset.name,           # name
            asset.type,          # type
            asset.market,        # market
            asset.data_path,     # data_path
            params_json          # params (inclui timeframes e outros parâmetros)
        ]

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

    def _serialize_money_management(self, mm: Money_Management_Algorithm) -> List[Any]:
        """Serializa um Money Management para armazenamento no banco de dados"""
        try:
            # Guarda apenas informações básicas
            basic_info = {
                "name": mm.name,
                "position_sizing_type": mm.position_sizing_type,
                "position_sizing_from": mm.position_sizing_from,
                "position_sizing_method": mm.position_sizing_method,
                "init_capital": mm.init_capital,
                "max_capital_exposure": mm.max_capital_exposure,
                "max_drawdown": mm.max_drawdown,
                "trade_risk_default": mm.trade_risk_default,
                "trade_risk_min": mm.trade_risk_min,
                "trade_risk_max": mm.trade_risk_max,
                "trade_max_num_open": mm.trade_max_num_open,
                "trade_min_num_analysis": mm.trade_min_num_analysis,
                "confidence_level": mm.confidence_level,
                "kelly_weight": mm.kelly_weight
            }
            
            return [mm.name, json.dumps(basic_info)]
        except Exception as e:
            print(f"Erro na serialização: {e}")
            # Retorna valores padrão em caso de erro
            return [mm.name, json.dumps({"name": mm.name})]

    # ==================== Deserialization Methods ====================

    def _deserialize_asset(self, row: List[Any]) -> Asset:
        """Deserializa um asset do banco de dados"""
        try:
            name, type, market, data_path, params_json = row[:5]  # Pega apenas os primeiros 5 campos
            
            # Deserializa os parâmetros
            params_dict = json.loads(params_json) if params_json else {}
            
            # Cria o asset com os parâmetros básicos
            asset = Asset(
                name=name,
                type=type,
                market=market,
                data_path=data_path
            )
            
            # Atualiza os parâmetros específicos
            if params_dict:
                for key, value in params_dict.items():
                    setattr(asset, key, value)
            
            return asset
            
        except Exception as e:
            print(f"Erro ao deserializar asset {row[0] if row else 'unknown'}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _deserialize_asset_portfolio(self, row: List[Any]) -> Asset_Portfolio:
        """Deserializa um portfólio do banco de dados"""
        try:
            if len(row) != 2:  # Verifica se tem exatamente 2 colunas (name e assets)
                raise ValueError(f"Expected 2 columns but got {len(row)}")
                
            name, asset_names_str = row
            asset_names = json.loads(asset_names_str) if asset_names_str else []
            
            # Carrega os assets como um dicionário
            asset_dict = {}
            for asset_name in asset_names:
                asset = self.read(Asset, asset_name)
                if asset:
                    asset_dict[asset_name] = asset
                    
            return Asset_Portfolio({'name': name, 'assets': asset_dict})
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

    def _deserialize_money_management(self, row: List[Any]) -> Money_Management_Algorithm:
        """Deserializa um Money Management do banco de dados"""
        try:
            name, params_json = row
            basic_info = json.loads(params_json)
            
            # Adiciona o diretório de MM ao path se necessário
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mm_dir = os.path.join(script_dir, "Money Management Algo")
            if mm_dir not in sys.path:
                sys.path.append(mm_dir)
            
            # Tenta importar o MM do arquivo
            try:
                module = __import__(name)
                importlib.reload(module)  # Recarrega o módulo para garantir versão mais recente
                
                # Primeiro tenta obter a instância money_management
                if hasattr(module, 'money_management'):
                    mm = module.money_management
                    mm_class = type(mm)
                    # Adiciona o mapeamento da classe se ainda não existir
                    if mm_class not in self._table_map:
                        self._table_map[mm_class] = self._table_map[Money_Management_Algorithm].copy()
                        
                    # Atualiza os parâmetros
                    for key, value in basic_info.items():
                        if hasattr(mm, key):
                            setattr(mm, key, value)
                            
                    return mm
                else:
                    raise ImportError(f"Não foi possível encontrar o MM {name} no módulo")
                
            except (ImportError, AttributeError) as e:
                print(f"Aviso: Não foi possível carregar o MM {name} do arquivo: {e}")
                
                # Se não conseguir carregar do arquivo, cria uma versão básica
                return Money_Management_Algorithm(basic_info)
                
        except Exception as e:
            print(f"Erro na deserialização: {e}")
            # Retorna um MM básico em caso de erro
            return Money_Management_Algorithm({"name": name or "unknown"})

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
                print("4. Gerenciar Money Management")
                print("5. Gerenciar Backtests")
                print("6. Sair")
                
                choice = input("\nEscolha uma opção: ")
                
                if choice == '1':
                    self._manage_assets()
                elif choice == '2':
                    self._manage_portfolios()
                elif choice == '3':
                    self._manage_strats()
                elif choice == '4':
                    self._manage_money_management()
                elif choice == '5':
                    self._manage_backtest()
                elif choice == '6':
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
                # Just call list_all and store the result
                assets = self.list_all(Asset)
                
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
                print("\nModificar Asset:")
                asset_name = input("Nome do Asset: ")
                print("\nO que deseja modificar?")
                print("1. Tipo")
                print("2. Mercado")
                print("3. Caminho dos dados")
                print("4. Parâmetros específicos")
                
                mod_choice = input("\nEscolha uma opção: ")
                
                if mod_choice in ['1', '2', '3']:
                    attribute = {
                        '1': 'type',
                        '2': 'market',
                        '3': 'data_path'
                    }[mod_choice]
                    
                    new_value = input(f"Novo valor para {attribute}: ")
                    if self.modify_attribute(Asset, asset_name, attribute, new_value):
                        print("Asset modificado com sucesso!")
                    else:
                        print("Falha ao modificar Asset.")
                        
                elif mod_choice == '4':
                    print("\nParâmetros disponíveis:")
                    print("- tick")
                    print("- tick_fin_val")
                    print("- lot_value")
                    print("- min_lot")
                    print("- leverage")
                    print("- comissions")
                    print("- slippage")
                    print("- spread")
                    
                    param = input("\nQual parâmetro deseja modificar? ")
                    if param in ['tick', 'tick_fin_val', 'lot_value', 'min_lot', 'leverage', 'comissions', 'slippage', 'spread']:
                        try:
                            new_value = float(input(f"Novo valor para {param}: "))
                            if self.modify_attribute(Asset, asset_name, param, new_value):
                                print("Parâmetro modificado com sucesso!")
                            else:
                                print("Falha ao modificar parâmetro.")
                        except ValueError:
                            print("Valor inválido. Digite um número.")
                    else:
                        print("Parâmetro inválido.")
                        
            elif choice == '5':
                asset_name = input("Nome do Asset a excluir: ")
                if self.delete(Asset, asset_name):
                    print("Asset excluído com sucesso!")
                else:
                    print("Falha ao excluir Asset.")
                    
            elif choice == '6':
                # Lista todos os assets disponíveis
                assets = self.list_all(Asset)  # Agora retorna apenas os nomes
                if not assets:
                    print("\nNenhum Asset cadastrado.")
                    continue

                selected = input("\nEscolha o número do Asset (ou ENTER para ver todos): ").strip()
                
                if selected:
                    try:
                        idx = int(selected) - 1
                        if 0 <= idx < len(assets):
                            asset_name = assets[idx]
                            asset = self.read(Asset, asset_name)
                            if asset:
                                print(f"\nTimeframes disponíveis para {asset.name}:")
                                asset.timeframes_load_available()  # Carrega timeframes disponíveis
                                timeframes = asset.timeframes_list()
                                if timeframes:
                                    for tf in timeframes:
                                        print(f"- {tf}")
                                else:
                                    print("Nenhum timeframe encontrado.")
                            else:
                                print("Erro ao carregar o Asset.")
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
                            asset.timeframes_load_available()  # Carrega timeframes disponíveis
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
                self.list_all(Asset_Portfolio)
                
            elif choice == '2':
                portfolio_name = input("Nome do Portfólio: ")
                details = self.get_details(Asset_Portfolio, portfolio_name)
                if details:
                    print("\nDetalhes do Portfólio:")
                    for k, v in details.items():
                        print(f"{k}: {v}")
                else:
                    print("Portfólio não encontrado.")
                    
            elif choice == '3':
                print("\nCriar novo Portfólio:")
                portfolio_name = input("Nome do Portfólio: ")
                
                # Lista os assets disponíveis
                available_assets = self.list_all(Asset)
                if not available_assets:
                    print("Nenhum Asset disponível para criar o Portfólio.")
                    continue
                    
                selected = input("\nNúmeros dos Assets a incluir (separados por vírgula): ")
                try:
                    indices = [int(x.strip()) for x in selected.split(',')]
                    selected_assets = {}
                    
                    for idx in indices:
                        if 0 <= idx < len(available_assets):
                            asset_name = available_assets[idx]
                            asset = self.read(Asset, asset_name)
                            if asset:
                                selected_assets[asset_name] = asset
                            else:
                                print(f"Erro ao carregar Asset {asset_name}")
                        else:
                            print(f"Índice inválido ignorado: {idx}")
                    
                    if selected_assets:
                        portfolio = Asset_Portfolio({
                            'name': portfolio_name,
                            'assets': selected_assets
                        })
                        if self.create(portfolio):
                            print("Portfólio criado com sucesso!")
                        else:
                            print("Falha ao criar Portfólio.")
                    else:
                        print("Nenhum Asset válido selecionado. Portfólio não criado.")
                except ValueError:
                    print("Entrada inválida. Use números separados por vírgula (ex: 0,1,2)")
                except Exception as e:
                    print(f"Erro ao criar portfólio: {e}")
                    import traceback
                    traceback.print_exc()
                    
            elif choice == '4':
                print("\nModificar Portfólio:")
                portfolio_name = input("Nome do Portfólio: ")
                portfolio = self.read(Asset_Portfolio, portfolio_name)
                
                if not portfolio:
                    print("Portfólio não encontrado.")
                    continue
                    
                print("\nO que deseja fazer?")
                print("1. Adicionar Assets")
                print("2. Remover Assets")
                
                mod_choice = input("\nEscolha uma opção: ")
                
                if mod_choice == '1':
                    # Lista assets disponíveis que não estão no portfólio
                    available_assets = self.list_all(Asset)
                    current_assets = set(portfolio.assets.keys())
                    available_assets = [a for a in available_assets if a not in current_assets]
                    
                    if not available_assets:
                        print("Não há Assets disponíveis para adicionar.")
                        continue
                        
                    print("\nAssets disponíveis:")
                    for i, asset_name in enumerate(available_assets, 1):
                        print(f"{i}. {asset_name}")
                        
                    choice = input("\nEscolha o número do Asset para adicionar: ")
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(available_assets):
                            asset_name = available_assets[idx]
                            asset = self.read(Asset, asset_name)
                            if asset:
                                portfolio.asset_add(asset)
                                if self.update(portfolio):
                                    print(f"Asset {asset_name} adicionado com sucesso!")
                                else:
                                    print("Falha ao atualizar Portfólio.")
                            else:
                                print("Erro ao carregar Asset.")
                        else:
                            print("Número inválido.")
                    except ValueError:
                        print("Entrada inválida.")
                        
                elif mod_choice == '2':
                    if not portfolio.assets:
                        print("Portfólio não possui Assets para remover.")
                        continue
                        
                    print("\nAssets no Portfólio:")
                    assets_list = list(portfolio.assets.keys())
                    for i, asset_name in enumerate(assets_list, 1):
                        print(f"{i}. {asset_name}")
                        
                    choice = input("\nEscolha o número do Asset para remover: ")
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(assets_list):
                            asset_name = assets_list[idx]
                            portfolio.asset_remove(asset_name)
                            if self.update(portfolio):
                                print(f"Asset {asset_name} removido com sucesso!")
                            else:
                                print("Falha ao atualizar Portfólio.")
                        else:
                            print("Número inválido.")
                    except ValueError:
                        print("Entrada inválida.")
                        
            elif choice == '5':
                portfolio_name = input("Nome do Portfólio a excluir: ")
                if self.delete(Asset_Portfolio, portfolio_name):
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

    def _manage_money_management(self):
        """Manage Money Management Algorithms"""
        while True:
            print("\n=== Gerenciamento de Money Management ===")
            print("1. Listar Money Management")
            print("2. Criar Money Management")
            print("3. Deletar Money Management")
            print("4. Editar Money Management")
            print("5. Detalhes do Money Management")
            print("0. Voltar ao menu principal")
            
            choice = input("\nEscolha uma opção: ")
            
            if choice == '1':
                print("\nMoney Management Algorithms:")
                mms = self.list_all(Money_Management_Algorithm)
                if mms:
                    for mm in mms:
                        if isinstance(mm, Money_Management_Algorithm):
                            print(f"- {mm.name}")
                else:
                    print("Nenhum Money Management encontrado.")
                    
            elif choice == '2':
                print("\nCriar novo Money Management:")
                print("1. Criar do template")
                print("2. Carregar MM existente")
                
                create_choice = input("\nEscolha uma opção: ")
                
                if create_choice == '1':
                    name = input("Nome do Money Management: ")
                    
                    # Verifica se já existe
                    if self.read(Money_Management_Algorithm, name):
                        print(f"Money Management '{name}' já existe.")
                        continue
                        
                    try:
                        # Cria o arquivo do MM
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        mm_dir = os.path.join(script_dir, "Money Management Algo")
                        mm_file = os.path.join(mm_dir, f"{name}.py")
                        
                        # Copia o template
                        with open(os.path.join(mm_dir, "mm_template.py"), 'r') as f:
                            template_content = f.read()
                            
                        # Substitui o nome da classe e configuração
                        new_content = template_content.replace("MM_Template", name)
                        new_content = new_content.replace('name: str = "MM_Template"', f'name: str = "{name}"')
                        
                        # Salva o novo arquivo
                        with open(mm_file, 'w') as f:
                            f.write(new_content)
                            
                        # Importa o novo módulo
                        sys.path.append(mm_dir)
                        module = __import__(name)
                        importlib.reload(module)
                        
                        # Cria uma instância do MM
                        if hasattr(module, name):
                            mm_class = getattr(module, name)
                            config = getattr(module, "MoneyManagementConfig")()
                            config.name = name
                            mm = mm_class(config)
                            
                            # Registra no banco de dados
                            if self.create(mm):
                                print(f"Money Management '{name}' criado com sucesso.")
                            else:
                                print("Aviso: Não foi possível registrar o Money Management no banco de dados.")
                        else:
                            print(f"Erro: Classe {name} não encontrada no módulo.")
                            
                    except Exception as e:
                        print(f"Erro ao criar objeto: {e}")
                        
                elif create_choice == '2':
                    print("\nMoney Management disponíveis:")
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    mm_dir = os.path.join(script_dir, "Money Management Algo")
                    mm_files = [f[:-3] for f in os.listdir(mm_dir) if f.endswith('.py') and f != '__init__.py' and f != 'mm_template.py']
                    
                    for i, mm_file in enumerate(mm_files, 1):
                        print(f"{i}. {mm_file}")
                        
                    if not mm_files:
                        print("Nenhum Money Management encontrado.")
                        continue
                        
                    try:
                        idx = int(input("\nEscolha o número do Money Management: ")) - 1
                        if 0 <= idx < len(mm_files):
                            mm_name = mm_files[idx]
                            
                            # Verifica se já existe no banco
                            if self.read(Money_Management_Algorithm, mm_name):
                                print(f"Money Management '{mm_name}' já existe no banco de dados.")
                                continue
                                
                            # Importa o módulo
                            sys.path.append(mm_dir)
                            module = __import__(mm_name)
                            importlib.reload(module)
                            
                            # Cria uma instância do MM
                            if hasattr(module, mm_name):
                                mm_class = getattr(module, mm_name)
                                config = getattr(module, "MoneyManagementConfig")()
                                config.name = mm_name
                                mm = mm_class(config)
                                
                                # Registra no banco de dados
                                if self.create(mm):
                                    print(f"Money Management '{mm_name}' registrado com sucesso.")
                                else:
                                    print("Aviso: Não foi possível registrar o Money Management no banco de dados.")
                            else:
                                print(f"Erro: Classe {mm_name} não encontrada no módulo.")
                                
                    except (ValueError, IndexError):
                        print("Opção inválida.")
                    except Exception as e:
                        print(f"Erro ao carregar Money Management: {e}")
                        
            elif choice == '3':
                name = input("Nome do Money Management para deletar: ")
                if self.delete(Money_Management_Algorithm, name):
                    # Tenta deletar o arquivo também
                    try:
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        mm_file = os.path.join(script_dir, "Money Management Algo", f"{name}.py")
                        if os.path.exists(mm_file):
                            os.remove(mm_file)
                            print(f"Arquivo {name}.py deletado com sucesso.")
                    except Exception as e:
                        print(f"Aviso: Não foi possível deletar o arquivo {name}.py: {e}")
                        
            elif choice == '4':
                name = input("Nome do Money Management para editar: ")
                mm = self.read(Money_Management_Algorithm, name)
                
                if mm:
                    while True:
                        print("\nParâmetros disponíveis:")
                        print("1. Position Sizing")
                        print("2. Capital Management")
                        print("3. Risk Management")
                        print("4. Advanced Parameters")
                        print("0. Voltar")
                        
                        param_choice = input("\nEscolha uma categoria para editar (0-4): ")
                        
                        if param_choice == '0':
                            break
                            
                        elif param_choice == '1':
                            print("\nPosition Sizing Parameters:")
                            print(f"1. Type: {mm.position_sizing_type}")
                            print(f"2. From: {mm.position_sizing_from}")
                            print(f"3. Method: {mm.position_sizing_method}")
                            
                            sub_choice = input("\nEscolha um parâmetro para editar (1-3): ")
                            
                            if sub_choice == '1':
                                value = input("Novo valor (percentage/kelly/confidence): ")
                                if value in ['percentage', 'kelly', 'confidence']:
                                    self.modify_attribute(Money_Management_Algorithm, name, 'position_sizing_type', value)
                            elif sub_choice == '2':
                                value = input("Novo valor (balance/equity): ")
                                if value in ['balance', 'equity']:
                                    self.modify_attribute(Money_Management_Algorithm, name, 'position_sizing_from', value)
                            elif sub_choice == '3':
                                value = input("Novo valor (regular/dynamic): ")
                                if value in ['regular', 'dynamic']:
                                    self.modify_attribute(Money_Management_Algorithm, name, 'position_sizing_method', value)
                                    
                        elif param_choice == '2':
                            print("\nCapital Management Parameters:")
                            print(f"1. Initial Capital: {mm.init_capital}")
                            print(f"2. Max Capital Exposure: {mm.max_capital_exposure}")
                            print(f"3. Max Drawdown: {mm.max_drawdown}")
                            
                            sub_choice = input("\nEscolha um parâmetro para editar (1-3): ")
                            
                            if sub_choice == '1':
                                try:
                                    value = float(input("Novo valor: "))
                                    self.modify_attribute(Money_Management_Algorithm, name, 'init_capital', value)
                                except ValueError:
                                    print("Valor inválido.")
                            elif sub_choice == '2':
                                try:
                                    value = float(input("Novo valor (0-1): "))
                                    if 0 <= value <= 1:
                                        self.modify_attribute(Money_Management_Algorithm, name, 'max_capital_exposure', value)
                                except ValueError:
                                    print("Valor inválido.")
                            elif sub_choice == '3':
                                try:
                                    value = float(input("Novo valor (0-1): "))
                                    if 0 <= value <= 1:
                                        self.modify_attribute(Money_Management_Algorithm, name, 'max_drawdown', value)
                                except ValueError:
                                    print("Valor inválido.")
                                    
                        elif param_choice == '3':
                            print("\nRisk Management Parameters:")
                            print(f"1. Default Risk: {mm.trade_risk_default}")
                            print(f"2. Min Risk: {mm.trade_risk_min}")
                            print(f"3. Max Risk: {mm.trade_risk_max}")
                            print(f"4. Max Open Trades: {mm.trade_max_num_open}")
                            print(f"5. Min Analysis Trades: {mm.trade_min_num_analysis}")
                            
                            sub_choice = input("\nEscolha um parâmetro para editar (1-5): ")
                            
                            if sub_choice in ['1', '2', '3']:
                                try:
                                    value = float(input("Novo valor (0-1): "))
                                    if 0 <= value <= 1:
                                        param_map = {
                                            '1': 'trade_risk_default',
                                            '2': 'trade_risk_min',
                                            '3': 'trade_risk_max'
                                        }
                                        self.modify_attribute(Money_Management_Algorithm, name, param_map[sub_choice], value)
                                except ValueError:
                                    print("Valor inválido.")
                            elif sub_choice in ['4', '5']:
                                try:
                                    value = int(input("Novo valor: "))
                                    if value > 0:
                                        param_map = {
                                            '4': 'trade_max_num_open',
                                            '5': 'trade_min_num_analysis'
                                        }
                                        self.modify_attribute(Money_Management_Algorithm, name, param_map[sub_choice], value)
                                except ValueError:
                                    print("Valor inválido.")
                                    
                        elif param_choice == '4':
                            print("\nAdvanced Parameters:")
                            print(f"1. Confidence Level: {mm.confidence_level}")
                            print(f"2. Kelly Weight: {mm.kelly_weight}")
                            
                            sub_choice = input("\nEscolha um parâmetro para editar (1-2): ")
                            
                            if sub_choice in ['1', '2']:
                                try:
                                    value = float(input("Novo valor (0-1): "))
                                    if 0 <= value <= 1:
                                        param_map = {
                                            '1': 'confidence_level',
                                            '2': 'kelly_weight'
                                        }
                                        self.modify_attribute(Money_Management_Algorithm, name, param_map[sub_choice], value)
                                except ValueError:
                                    print("Valor inválido.")
                else:
                    print(f"Money Management '{name}' não encontrado.")
                    
            elif choice == '5':
                name = input("Nome do Money Management: ")
                details = self.get_details(Money_Management_Algorithm, name)
                if details:
                    print("\nDetalhes do Money Management:")
                    for key, value in details.items():
                        print(f"{key}: {value}")
                else:
                    print(f"Money Management '{name}' não encontrado.")
                    
            elif choice == '0':
                break
                
            else:
                print("Opção inválida.")

    def run_backtest(self, strategy_name: str, asset_name: str = None, stateless: bool = True) -> Optional[Dict[str, Any]]:
        """
        Executa backtest para uma estratégia
        
        Args:
            strategy_name: Nome da estratégia
            asset_name: Nome do ativo específico ou None para todos
            stateless: Se True, não mantém estado entre execuções
        """
        try:
            # Carrega a estratégia
            strategy = self.read(Strat, strategy_name)
            if not strategy:
                print(f"Estratégia {strategy_name} não encontrada")
                return None
            
            # Gera/atualiza sinais
            signals = strategy.generate_signals(asset_name)
            if not signals:
                return None
            
            # Cria instância do Backtest
            backtest = Backtest(stateless=stateless)
            
            # Executa backtest para cada ativo
            results = {}
            if isinstance(signals, dict):
                for name, df in signals.items():
                    print(f"\nExecutando backtest para {name}...")
                    result = backtest.run(df, strategy)
                    results[name] = result
            else:
                name = asset_name or next(iter(strategy.asset_portfolio.assets.keys()))
                print(f"\nExecutando backtest para {name}...")
                results[name] = backtest.run(signals, strategy)
            
            return results
            
        except Exception as e:
            print(f"Erro ao executar backtest: {e}")
            return None

    def run_backtest_setup(self, strat_name: str, portfolio_name: str = None, asset_name: str = None) -> bool:
        """
        Prepara e executa um backtest para uma estratégia
        
        Args:
            strat_name: Nome da estratégia
            portfolio_name: Nome do portfólio (opcional)
            asset_name: Nome do ativo específico no portfólio (opcional)
        """
        try:
            # 1. Carrega a estratégia
            strategy = self.read(Strat, strat_name)
            if not strategy:
                print(f"Estratégia {strat_name} não encontrada")
                return False
            
            # 2. Carrega/cria portfólio
            if portfolio_name:
                portfolio = self.read(Asset_Portfolio, portfolio_name)
                if not portfolio:
                    print(f"Portfólio {portfolio_name} não encontrado")
                    return False
            else:
                # Se não foi especificado um portfólio, verifica se a estratégia já tem um
                if hasattr(strategy, 'asset_portfolio') and strategy.asset_portfolio:
                    portfolio = strategy.asset_portfolio
                else:
                    print("Nenhum portfólio especificado ou associado à estratégia")
                    return False
                
            # 3. Verifica/carrega Money Management
            mm = None
            if hasattr(strategy, 'risk_rules'):
                # Cria um MM genérico baseado nas regras de risco da estratégia
                mm = Money_Management_Algorithm({
                    'name': f"{strat_name}_mm",
                    'position_sizing_type': strategy.risk_rules.position_sizing_type,
                    'position_sizing_from': strategy.risk_rules.position_sizing_from,
                    'position_sizing_method': strategy.risk_rules.position_sizing_method,
                    'trade_risk_default': strategy.risk_rules.trade_risk_default,
                    'trade_risk_min': strategy.risk_rules.trade_risk_min,
                    'trade_risk_max': strategy.risk_rules.trade_risk_max
                })
            
            # 4. Gera sinais
            if asset_name:
                # Para um ativo específico
                if asset_name not in portfolio.assets:
                    print(f"Ativo {asset_name} não encontrado no portfólio")
                    return False
                df = strategy.generate_signals(asset_name)
                
                # 5. Mostra o DataFrame com os sinais
                print(f"\nDataFrame com sinais para {asset_name}:")
                print("\nPrimeiras linhas:")
                print(df.head())
                print("\nÚltimas linhas:")
                print(df.tail())
                
                # Mostra colunas relevantes
                signal_cols = [col for col in df.columns if 'signal' in col.lower()]
                if signal_cols:
                    print("\nColunas de sinais:")
                    print(df[signal_cols].describe())
                
            else:
                # Para todos os ativos no portfólio
                for curr_asset_name in portfolio.assets:
                    df = strategy.generate_signals(curr_asset_name)
                    
                    print(f"\nDataFrame com sinais para {curr_asset_name}:")
                    print("\nPrimeiras linhas:")
                    print(df.head())
                    print("\nÚltimas linhas:")
                    print(df.tail())
                    
                    # Mostra colunas relevantes
                    signal_cols = [col for col in df.columns if 'signal' in col.lower()]
                    if signal_cols:
                        print("\nColunas de sinais:")
                        print(df[signal_cols].describe())
            
            return True
            
        except Exception as e:
            print(f"Erro ao preparar backtest: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _manage_backtest(self):
        """Menu para gerenciar Backtests"""
        while True:
            print("\n" + "-"*50)
            print("Gerenciamento de Backtest")
            print("-"*50)
            print("1. Preparar e Verificar Backtest")
            print("2. Voltar")
            
            choice = input("\nEscolha uma opção: ")
            
            if choice == '1':
                # Lista estratégias disponíveis
                strats = self.list_all(Strat)
                if not strats:
                    print("Nenhuma estratégia disponível")
                    continue
                
                print("\nEstratégias disponíveis:")
                for i, strat in enumerate(strats, 1):
                    print(f"{i}. {strat}")
                
                try:
                    strat_idx = int(input("\nEscolha o número da estratégia: ")) - 1
                    if not (0 <= strat_idx < len(strats)):
                        print("Número inválido")
                        continue
                    
                    strat_name = strats[strat_idx]
                    
                    # Lista portfólios disponíveis
                    portfolios = self.list_all(Asset_Portfolio)
                    if portfolios:
                        print("\nPortfólios disponíveis:")
                        print("0. Usar portfólio da estratégia")
                        for i, port in enumerate(portfolios, 1):
                            print(f"{i}. {port}")
                        
                        port_idx = int(input("\nEscolha o número do portfólio (0 para usar o da estratégia): "))
                        portfolio_name = None if port_idx == 0 else portfolios[port_idx-1]
                    else:
                        portfolio_name = None
                    
                    # Pergunta se quer testar um ativo específico
                    test_specific = input("\nTestar um ativo específico? (s/N): ").lower() == 's'
                    asset_name = None
                    
                    if test_specific:
                        if portfolio_name:
                            portfolio = self.read(Asset_Portfolio, portfolio_name)
                        else:
                            strategy = self.read(Strat, strat_name)
                            portfolio = strategy.asset_portfolio
                            
                        if portfolio:
                            print("\nAtivos disponíveis:")
                            assets = list(portfolio.assets.keys())
                            for i, asset in enumerate(assets, 1):
                                print(f"{i}. {asset}")
                                
                            asset_idx = int(input("\nEscolha o número do ativo: ")) - 1
                            if 0 <= asset_idx < len(assets):
                                asset_name = assets[asset_idx]
                            else:
                                print("Número inválido")
                                continue
                        else:
                            print("Portfólio não encontrado")
                            continue
                    
                    # Executa o backtest
                    if self.run_backtest_setup(strat_name, portfolio_name, asset_name):
                        print("\nPreparação do backtest concluída com sucesso!")
                    else:
                        print("\nErro na preparação do backtest")
                    
                except ValueError:
                    print("Entrada inválida")
                except Exception as e:
                    print(f"Erro: {e}")
            
            elif choice == '2':
                break
            else:
                print("Opção inválida")

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
            values = info['serialize'](obj)
            
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
            return info['deserialize'](row)
            
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



