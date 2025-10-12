import BaseClass, uuid

class Trade(BaseClass): # Each Trade represents 1 Position too
    def __init__(self, params: dict): 
        super().__init__()
        self.id = params.get('id', str(uuid.uuid4()))
        self.asset = params.get('asset')
        self.status = params.get('status', 'open')
        self.direction = params.get('direction', 'long')
        self.entry_price = params.get('entry_price', 0.0)
        self.entry_time = params.get('entry_time', '00:00:00')
        self.lot_size = params.get('lot_size', 1.0)
        self.stop_loss = params.get('stop_loss')
        self.take_profit = params.get('take_profit')
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.profit = None
        self.profit_r = None

    #def close(self, close_params: dict): # USAR trade.modify_specific_value('status', 'close')
    #    return self.status='close'

    # Open/Close Trade fazem mais sentido em Backtest, não sendo uma função CRUD
