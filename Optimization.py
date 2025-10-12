from dataclasses import dataclass, field
from BaseClass import BaseClass
import uuid

@dataclass
class OptimizationParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    
class Optimization(BaseClass):
    def __init__(self, op_params: OptimizationParams):
        super().__init__()
        self.name = op_params.name


























