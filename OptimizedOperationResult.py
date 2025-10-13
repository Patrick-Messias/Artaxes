from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json, pickle
import pandas as pd

"""
# Hierarquia implementada:
portfolio = {
    'models': {
        'model_name': {
            'assets': {...},
            'strats': {
                'strat_name': {
                    'indicators': {...},
                    'asset_mapping': {...},
                    'results': {...}
                }
            },
            'shared_indicators': {...}
        }
    },
    'portfolio_metrics': {...}
}
"""

@dataclass
class ResultMetadata: # Memory optimized metadata
    size_bytes: int
    created_at: str
    last_accessed: str
    compression_type: Optional[str]=None
    file_path: Optional[str]=None

class OptimizedOperationResult: # Class to manage results optmally
    def __init__(self):
        self._structure = {
            'portfolio': {
                'models': {},
                'metadata': {},
                'portfolio_metrics': {}
            }
        }
        self._lazy_loaded={} # lazy-loaded data cache
        self._metadata={} # metadata of each result

    def _get_nested_path(self, path: str) -> tuple: # Converts str path to tuple for navegation
        return tuple(path.split('.'))

    def _navigate_to_path(self, path: tuple, create_if_missing: bool=False) -> Dict[str, Any]: # Navigates to specific path
        current = self._structure

        for key in path[:-1]:
            if key not in current:
                if create_if_missing:
                    current[key]={}
                else:
                    raise KeyError(f"Path not found: {'.'.join(path)}")
            current = current[key]
        return current

    def set_result(self, path: str, data: Any, compress: bool=True) -> None: # Defines a result with automatic optimization
        path_tuple = self._get_nested_path(path)
        parent = self._navigate_to_path(path_tuple, create_if_missing=True)

        # For large quantity of data use compression
        if compress and self._should_compress(data):
            compressed_data = self._compress_data(data)
            parent[path_tuple[-1]] = compressed_data
            self._metadata[path] = ResultMetadata(
                size_bytes=len(pickle.dumps(data)),
                created_at=str(pd.Timestamp.now()),
                last_accessed=str(pd.Timestamp.now()),
                compression_type='pickle_gzip'
            )
        else: parent[path_tuple[-1]] = data

    def get_result(self, path: str, lazy: bool=True) -> Any: # Recupera um resultado com Lazy Landing
        path_tuple = self._get_nested_path(path)
        parent = self._navigate_to_path(path_tuple)

        if path_tuple[-1] not in parent: raise KeyError(f"Result not found: {path}")
        result = parent[path_tuple[-1]]

        # If data is compressed, then decompresses on demand
        if isinstance(result, dict) and result.get("_compressed"):
            if lazy: return self._decompress_data(result)
            else: # Loads in background thread
                import threading
                threading.Thread(target=self._preload_result, args=(path,)).start()
                return self._decompress_data(result)
                
        if path in self._metadata: self._metadata[path].last_accessed = str(pd.Timestamp.now())
        return result

    def _should_compress(self, data: Any) -> bool: # Determines if data must be compressed
        try:
            size = len(pickle.dumps(data))
            return size > 1024*1024 # 1MB threshold
        except:
            return False

    def _compress_data(self, data: Any) -> Dict[str, Any]: # Compresses large data
        import gzip
        compressed = gzip.compress(pickle.dumps(data))
        return {
            '_compressed': True,
            '_data': compressed,
            '_type': type(data).__name__
        }

    def _decompress_data(self, compressed_dict: Dict[str, Any]) -> Any: # Decompresses data
        if not compressed_dict.get('_compressed'): return compressed_dict

        import gzip
        decompressed = gzip.decompress(compressed_dict['_data'])
        return pickle.loads(decompressed)

    def to_dict(self) -> Dict[str, Any]: # Serializes to Dict
        return {
            'structure': self._structure,
            'metadata': asdict(self._metadata) if hasattr(self, '_metadata') else {}
        }

    def save_to_file(self, filepath: str) -> None: # Saves optimized file
        import gzip

        data = self.to_dict()
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'OptimizedOperationResult': # Loads data
        import gzip

        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

        instance = cls()
        instance._structure = data['structure']
        instance._metadata = data.get('metadata', {})
        return instance





