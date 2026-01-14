'''
Description: 异常定义
Author: zyq
Date: 2025-12-25 16:23:50
LastEditors: zyq
LastEditTime: 2026-01-09 09:43:43
'''
class BaseAgentsException(Exception):
    pass

class LLMException(BaseAgentsException):
    pass

class MessageException(BaseAgentsException):
    pass

class ConfigException(BaseAgentsException):
    pass

class ToolException(BaseAgentsException):
    pass

class MemoryException(BaseAgentsException):
    pass

class RAGException(BaseAgentsException):
    pass

class ContextException(BaseAgentsException):
    pass