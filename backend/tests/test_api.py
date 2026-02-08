"""Tests for API routes."""
import pytest
from unittest.mock import patch, AsyncMock


# Test health endpoint works without any dependencies
@pytest.fixture
def app():
    """Create test app with mocked settings."""
    with patch("app.config.Settings") as mock_settings:
        mock_settings.return_value.cors_origin_list = ["http://localhost:3000"]
        mock_settings.return_value.anthropic_api_key = "test"
        mock_settings.return_value.tavily_api_key = "test"
        mock_settings.return_value.hasdata_api_key = "test"
        mock_settings.return_value.supabase_url = "https://test.supabase.co"
        mock_settings.return_value.supabase_anon_key = "test"
        mock_settings.return_value.default_model = "claude-sonnet-4-5-20250929"
        mock_settings.return_value.cors_origins = "http://localhost:3000"

        from app.main import app
        yield app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "secondorder"


def test_list_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 3
    model_ids = [m["id"] for m in data["models"]]
    assert "claude-opus-4-6" in model_ids
    assert "claude-sonnet-4-5-20250929" in model_ids
    assert "claude-haiku-4-5-20251001" in model_ids
