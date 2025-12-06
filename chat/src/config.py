from dataclasses import dataclass, field
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiSettings(BaseSettings):
    API_KEY: str

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=".env",
        extra="ignore",
    )


class OpenAiSettings(BaseSettings):
    API_KEY: str

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        extra="ignore",
    )


@dataclass
class Settings:
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    openai: OpenAiSettings = field(default_factory=OpenAiSettings)


settings = Settings()
