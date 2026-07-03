"""Unit tests for configuration management."""

from src.config.settings import (
    DatabricksSettings,
    Settings,
    SnowflakeSettings,
    get_settings,
)


class TestSnowflakeSettings:
    def test_defaults_allow_unconfigured_import(self):
        settings = SnowflakeSettings()
        # Should construct without env vars and report unconfigured
        assert isinstance(settings.account, str)
        assert settings.warehouse == "COMPUTE_WH"

    def test_env_loading(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct123")
        monkeypatch.setenv("SNOWFLAKE_USER", "svc_user")
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "hunter2")
        monkeypatch.setenv("SNOWFLAKE_SCHEMA", "GOLD")

        settings = SnowflakeSettings()
        assert settings.account == "acct123"
        assert settings.schema_name == "GOLD"
        assert settings.is_configured
        # Secrets must not leak in repr
        assert "hunter2" not in repr(settings)

    def test_is_configured_requires_all_credentials(self, monkeypatch):
        monkeypatch.delenv("SNOWFLAKE_PASSWORD", raising=False)
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct123")
        monkeypatch.setenv("SNOWFLAKE_USER", "svc_user")
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "")
        settings = SnowflakeSettings()
        assert not settings.is_configured


class TestDatabricksSettings:
    def test_is_configured(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://dbx.example.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "tok")
        assert DatabricksSettings().is_configured


class TestSettings:
    def test_aggregate_settings_construct(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "test")
        settings = Settings()
        assert settings.environment == "test"
        assert settings.model.random_state == 42
        assert "1h" in settings.features.velocity_windows
        assert settings.api.port == 8000

    def test_get_settings_is_cached(self):
        assert get_settings() is get_settings()
