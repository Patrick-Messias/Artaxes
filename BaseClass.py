

class BaseClass():
    
    def get_value(self, attr_value: str=None):
        if attr_value is None: return list(self.__dict__.keys())
        elif hasattr(self, attr_value): return getattr(self, attr_value)
        else: raise AttributeError(f"!!! --- Attribute '{attr_value}' not found in instance --- !!!")

    def delete_value(self, attr_value: str=None):
        if attr_value is None: return list(self.__dict__.keys())
        elif hasattr(self, attr_value): delattr(self, attr_value) 
        else: raise AttributeError(f"!!! --- Attribute '{attr_value}' not found in instance --- !!!")

    def modify_specific_value(self, attr_value: str, new_attr_value):
        if attr_value is None: return list(self.__dict__.keys())
        elif hasattr(self, attr_value): setattr(self, attr_value, new_attr_value)
        else: raise AttributeError(f"!!! --- Attribute '{attr_value}' not found in instance --- !!!")

    def list_values(self): 
        return {key: value for key, value in self.__dict__.items()} 
