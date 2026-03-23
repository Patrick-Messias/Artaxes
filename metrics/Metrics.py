import polars as pl, numpy as np

# Uses [Daily Returns] or [Trade]

def MonteCarlo(data, runs: int, dataType: str = None, dateType: str = 'Entrada', shuffle: bool = True):
    mcResult = []

    # Verifica se dataType é None (dados já são uma lista/array) ou se é uma coluna do DataFrame
    if dataType is None:
        if isinstance(data, (list, np.ndarray)):
            data_array = np.array(data)  # Converte para array numpy
        else:
            raise TypeError("Se dataType for None, 'data' deve ser uma lista ou array.")
    else:
        if isinstance(data, pd.DataFrame):
            data_array = data[dataType].values  # Extrai a coluna do DataFrame
        else:
            raise TypeError("Se dataType for especificado, 'data' deve ser um DataFrame.")

    # Extrai as datas, se dateType for especificado
    date_array = data[dateType].values if dateType is not None and isinstance(data, pd.DataFrame) else None

    for _ in range(runs):
        if shuffle:
            # Rearranja os dados sem reposição
            indices = np.random.permutation(len(data_array))
        else:
            # Escolhe amostras com reposição
            indices = np.random.choice(len(data_array), size=len(data_array), replace=True)
        
        # Preserva o tipo original dos dados
        shuffled_data = data_array[indices].astype(data_array.dtype)
        
        if dateType is not None and date_array is not None:
            shuffled_dates = date_array[indices]
            mcResult.append(list(zip(shuffled_dates, shuffled_data)))  # Emparelha datas e dados
        else:
            mcResult.append(shuffled_data)  # Apenas os dados

    return mcResult


# Permutation Test














































