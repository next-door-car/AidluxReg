import os
import time
from abc import ABC, abstractmethod
from typing import Any, Type, Union, List, Optional, Callable

class BaseConfig(ABC):
    def __init__(self, settings):
        self.config: dict
        self.settings = settings # 获取局部变量的配置
        
    @abstractmethod
    def load_config(self, import_config: str):
        pass
    
    @abstractmethod
    def save_config(self, export_config: str):
        pass
    
    # 设置配置
    @abstractmethod
    def set_config(self, keys: list, value: Any):
        self.config[keys] = value
        pass
    
    @abstractmethod
    # 读取配置参数
    def get_config(self, keys: list) -> Any:
        return self.config[keys]

    @abstractmethod
    # 解析配置参数
    def parse_config(self, config: dict):
        pass
    
    @abstractmethod
    # 读取配置参数
    def setup_config(self, config: dict):
        pass

    @abstractmethod
    # 读取配置参数
    def mkdir_config(self, config: dict):
        pass