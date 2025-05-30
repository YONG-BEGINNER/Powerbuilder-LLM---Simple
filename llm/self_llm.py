from langchain.llms.base import LLM
from typing import Dict, Any, Mapping
from pydantic import Field

class Self_LLM(LLM):
    url: str = None
    model_name: str = "llama3-70b-8192"
    request_timeout:float = None
    temperature: float = 0.3
    api_key: str = None
    model_kwargs:Dict[str, Any] = Field(default_factory=dict)

    @property
    def _default_params(self) -> Dict[str, Any]:
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
        }
        return {**normal_params}
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {**{"model_name": self.model_name}, **self._default_params}