import polars as pl, numpy as np, sys
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Storage import Storage
from Asset import Asset
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\indicators')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')



PARAMS = {
    "fast": range(10, 60+1, 10),
    "slow": range(20, 200+1, 30),
    "var_window": range(20, 50+1, 30),
    "var_alpha": [0.05, 0.05],
}

from MA import MA # type: ignore
from VAR import VAR # type: ignore
INDICATORS = {
    "fast_ma": MA(window="fast", ma_type="sma", price_col="close"),
    "slow_ma": MA(window="slow", ma_type="sma", price_col="close"),
    "risk_var": VAR(window="var_window", alpha="var_alpha", var_type="historical"),
}



def long_entry(df, ps, cache):
    fast = get_indicator(cache, "fast_ma", ps, INDICATORS["fast_ma"])
    slow = get_indicator(cache, "slow_ma", ps, INDICATORS["slow_ma"])
    L = (df['dweek']==4) & (df['close'] > df['open']) & (df['hour'] < 10)
    return (L & ~L.shift(1).fill_null(False))

def long_exit(df, ps, cache):
    return (df['close'] < df['low'].shift(1))



def plot_surface(results):

    heatmap = (
        results
        .group_by(
            ["ps_id", "holding_periods"]
        )
        .agg(
            pl.col("pnl")
            .sum()
            .alias("pnl")
        )
        .sort(
            ["ps_id", "holding_periods"]
        )
        .pivot(
            values="pnl",
            index="holding_periods",
            on="ps_id"
        )
        .sort("holding_periods")
    )

    x_labels = (
        heatmap["holding_periods"]
        .to_numpy()
    )

    y_labels = (
        heatmap.columns[1:]
    )

    z = (
        heatmap
        .drop("holding_periods")
        .fill_null(0)
        .to_numpy()
        .T
    )

    fig, ax = plt.subplots(
        figsize=(20, 12),
        facecolor="black"
    )

    ax.set_facecolor("black")

    im = ax.imshow(
        z,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )

    cbar = plt.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "PnL",
        color="white"
    )

    cbar.ax.yaxis.set_tick_params(
        color="white"
    )

    plt.setp(
        plt.getp(
            cbar.ax.axes,
            'yticklabels'
        ),
        color="white"
    )

    # eixo X
    x_step = max(
        1,
        len(x_labels) // 20
    )

    ax.set_xticks(
        np.arange(
            0,
            len(x_labels),
            x_step
        )
    )

    ax.set_xticklabels(
        x_labels[::x_step],
        color="white"
    )

    # eixo Y
    y_step = max(
        1,
        len(y_labels) // 30
    )

    ax.set_yticks(
        np.arange(
            0,
            len(y_labels),
            y_step
        )
    )

    ax.set_yticklabels(
        y_labels[::y_step],
        fontsize=8,
        color="white"
    )

    ax.tick_params(
        axis="both",
        colors="white"
    )

    plt.tight_layout()

    plt.show()


    
def run_backtest(
    df,
    params,
    indicators,
    entry,
    exit=None,
    asset_name=None,
    asset_obj=None,
    direction="long",
    max_horizon=63,
    column_name="close"
):

    param_map = build_param_map(params)

    cache = build_indicator_cache(
        df,
        param_map,
        indicators
    )

    future = build_forward_returns(
        df,
        max_horizon,
        column_name
    )

    datetimes = df["datetime"].to_numpy()
    prices = df[column_name].to_numpy()

    entry_datetime_col = []
    exit_datetime_col = []

    entry_price_col = []
    exit_price_col = []

    holding_periods_col = []

    lot_size_col = []

    ret_pct_col = []
    pnl_col = []

    asset_col = []

    ps_id_col = []

    entry_type_col = []
    exit_type_col = []

    min_lot = (
        getattr(asset_obj, "min_lot", 1.0)
        if asset_obj is not None
        else 1.0
    )

    for ps in param_map.to_dicts():

        entry_signal = entry(
            df,
            ps,
            cache
        )

        entry_idx = np.where(
            entry_signal
            .fill_null(False)
            .to_numpy()
        )[0]

        exit_signal = None

        if callable(exit):

            exit_signal = (
                exit(
                    df,
                    ps,
                    cache
                )
                .fill_null(False)
                .to_numpy()
            )

        ps_id = ps["ps_id"]

        for idx in entry_idx:

            idx = int(idx)

            entry_dt = datetimes[idx]

            entry_price = prices[idx]

            n = 1

            while True:

                exit_idx = idx + n

                if exit_idx >= len(prices):
                    break

                if (
                    max_horizon is not None
                    and n > max_horizon
                ):
                    break

                exit_type = f"n_{n}"

                if (
                    exit_signal is not None
                    and bool(exit_signal[exit_idx])
                ):
                    exit_type = "signal"

                exit_price = prices[exit_idx]

                if direction == "long":

                    lot_size = min_lot

                    ret_pct = (
                        exit_price
                        /
                        entry_price
                    ) - 1.0

                else:

                    lot_size = -min_lot

                    ret_pct = (
                        entry_price
                        /
                        exit_price
                    ) - 1.0

                pnl = ret_pct * abs(lot_size)

                entry_datetime_col.append(
                    entry_dt
                )

                exit_datetime_col.append(
                    datetimes[exit_idx]
                )

                entry_price_col.append(
                    float(entry_price)
                )

                exit_price_col.append(
                    float(exit_price)
                )

                holding_periods_col.append(
                    n
                )

                lot_size_col.append(
                    float(lot_size)
                )

                ret_pct_col.append(
                    float(ret_pct)
                )

                pnl_col.append(
                    float(pnl)
                )

                asset_col.append(
                    asset_name
                )

                ps_id_col.append(
                    ps_id
                )

                entry_type_col.append(
                    direction
                )

                exit_type_col.append(
                    exit_type
                )

                if (
                    exit_signal is not None
                    and bool(exit_signal[exit_idx])
                ):
                    break

                n += 1

    return pl.DataFrame({

        "entry_datetime":
            pl.Series(
                entry_datetime_col,
                dtype=pl.Datetime
            ),

        "exit_datetime":
            pl.Series(
                exit_datetime_col,
                dtype=pl.Datetime
            ),

        "entry_price":
            entry_price_col,

        "exit_price":
            exit_price_col,

        "holding_periods":
            holding_periods_col,

        "lot_size":
            lot_size_col,

        "ret_pct":
            ret_pct_col,

        "pnl":
            pnl_col,

        "asset":
            asset_col,

        "ps_id":
            ps_id_col,

        "entry_type":
            entry_type_col,

        "exit_type":
            exit_type_col,

    })

def build_forward_returns(
    df,
    max_horizon=63,
    column_name="close"
):

    close = df[column_name].to_numpy()

    future = np.empty(
        (len(close), max_horizon),
        dtype=np.float32
    )

    future[:] = np.nan

    for n in range(1, max_horizon + 1):

        future[:-n, n-1] = (
            close[n:] / close[:-n]
        ) - 1.0

    return future

def build_param_map(params):
    names = list(params.keys())
    values = [params[k] for k in names]
    rows = []

    for ps_id, combo in enumerate(product(*values), start=1):
        row = {"ps_id": ps_id}

        for name, value in zip(names, combo):
            row[name] = value

        rows.append(row)

    return pl.DataFrame(rows)

def get_indicator(cache, name, ps, indicator):
    used_params = indicator_used_params(indicator)
    key = tuple(ps[p] for p in used_params)

    return cache[ (name, key) ]

def indicator_used_params(ind):
    return [
        v
        for k, v in ind.params.items()
        if isinstance(v, str) and v in PARAMS
    ]

def build_indicator_cache(df, param_map, indicators):
    cache = {}
    rows = param_map.to_dicts()

    for ind_name, ind in indicators.items():
        used_params = indicator_used_params(ind)
        unique_configs = {}

        for row in rows:
            key = tuple(row[p] for p in used_params)

            if key not in unique_configs:
                cfg = {}

                for param_name, param_value in ind.params.items():
                    if (isinstance(param_value, str) and param_value in PARAMS):
                        cfg[param_name] = row[param_value]
                unique_configs[key] = cfg

        for key, cfg in unique_configs.items():
            expr = ind.get_expression(cfg)
            result = (df.select(expr).to_series())
            cache[(ind_name, key)] = result

    return cache


assets = Asset.load_all()
eurusd = assets.get("EURUSD")
gbpusd = assets.get("GBPUSD")
usdjpy = assets.get("USDJPY")
winfut = assets.get("WIN$")
global_assets = {'EURUSD': eurusd, 'GBPUSD': gbpusd, 'USDJPY': usdjpy, 'WIN$': winfut}
storage = Storage()

df = winfut.load("M15")
df = df.with_columns(pl.col("datetime").dt.weekday().alias("dweek"))
df = df.with_columns(pl.col("datetime").dt.hour().alias("hour"))

results = run_backtest(
    df=df,
    params=PARAMS,
    indicators=INDICATORS,
    entry=long_entry,
    exit=long_exit,
    max_horizon=24,
    column_name="close"
)
print(results)
plot_surface(results)














































