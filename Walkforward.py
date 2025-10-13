from dataclasses import dataclass, field
from BaseClass import BaseClass
import uuid

@dataclass
class WalkforwardParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    
class Walkforward(BaseClass):
    def __init__(self, op_params: WalkforwardParams = WalkforwardParams()):
        super().__init__()
        self.name = op_params.name

    def run(self):
        return None
























