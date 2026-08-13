from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "demand-forecasting-service"
    log_level: str = "INFO"
    model_path: str = "models/model.joblib"

    class Config:
        env_file = ".env"


settings = Settings()        