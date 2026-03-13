from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.contracts import (
    AssembleRequest,
    AssembleResult,
    EngineSaveRequest,
    EngineSaveResult,
    EngineSearchRequest,
    EngineSearchResult,
    EvalRequest,
    EvalResult,
    ProcessorOutput,
    RawConversationInput,
    ReflectRequest,
    ReflectResult,
)


# 1) 记忆预处理器：只关心原始输入到标准 MemoryChunk 的转换
class MemoryProcessor(ABC):
    @abstractmethod
    def process(self, payload: RawConversationInput) -> ProcessorOutput:
        raise NotImplementedError


# 2) 记忆引擎黑盒：统一 save/search，内部可自由实现存储+检索耦合逻辑
class MemoryEngine(ABC):
    @abstractmethod
    def save(self, request: EngineSaveRequest) -> EngineSaveResult:
        raise NotImplementedError

    @abstractmethod
    def search(self, request: EngineSearchRequest) -> EngineSearchResult:
        raise NotImplementedError


# 3) 上下文组装器：将检索结果拼接为最终 prompt
class ContextAssembler(ABC):
    @abstractmethod
    def assemble(self, request: AssembleRequest) -> AssembleResult:
        raise NotImplementedError


# 4) 异步反思器：不影响主链路，后台输出高阶洞察
class MemoryReflector(ABC):
    @abstractmethod
    async def reflect(self, request: ReflectRequest) -> ReflectResult:
        raise NotImplementedError


# 5) 评估台：输入当前链路数据，输出多维度评估指标
class EvaluationBench(ABC):
    @abstractmethod
    def evaluate(self, request: EvalRequest) -> EvalResult:
        raise NotImplementedError
