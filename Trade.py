import BaseClass, datetime

class Trade(BaseClass):
    def __init__(self, params: dict): 
        super().__init__()
        self.asset = params.get('asset')
        self.direction = params.get('direction', 'long')
        self.entry_price = params.get('entry_price', 0.0)
        self.entry_time = params.get('entry_time', datetime.datetime.now())
        self.lot_size = params.get('lot_size', 1.0)
        self.stop_loss = params.get('stop_loss')
        self.take_profit = params.get('take_profit')
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.profit = None
        self.profit_r = None

    def close(self, close_params: dict):
        self.exit_price = close_params.get('exit_price', 0.0)
        self.exit_time = close_params.get('exit_time', datetime.datetime.now())
        self.exit_reason = close_params.get('exit_reason', 'unknown')
        
        price_diff = (self.exit_price - self.entry_price) if self.direction == 'long' else (self.entry_price - self.exit_price)
        self.profit = price_diff * self.lot_size
        
        if self.stop_loss:
            risk = abs(self.entry_price - self.stop_loss)
            self.profit_r = price_diff / risk if risk != 0 else 0


    def get_trades_returns(self, trades: dict[Trade], type: str='perc'):
        trade_returns=[]
        
        for trade in trades:
            trade_returns.append(trade.)

        return trade_returns


