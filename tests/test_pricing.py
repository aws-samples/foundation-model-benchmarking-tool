import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.fmbench.scripts.pricing import get_ec2_pricing, load_and_update_pricing


@pytest.fixture
def mock_pricing_data():
    return {
        "pricing": {
            "instance_based": {
                "ml.c5.xlarge": 0.204,
            },
            "token_based": {
                "ai21.j2-mid-v1": {
                    "input-per-1k-tokens": 0.0125,
                    "output-per-1k-tokens": 0.0125,
                }
            },
        }
    }


@patch("boto3.client")
def test_get_ec2_pricing_success(mock_boto3_client):
    # Mock the Pricing API response
    mock_client = MagicMock()
    mock_client.get_products.return_value = {
        "PriceList": [
            json.dumps(
                {
                    "terms": {
                        "OnDemand": {
                            "id": {
                                "priceDimensions": {
                                    "id": {"pricePerUnit": {"USD": "0.25"}}
                                }
                            }
                        }
                    }
                }
            )
        ]
    }
    mock_boto3_client.return_value = mock_client

    price = get_ec2_pricing("ml.c5.xlarge", "us-east-1")
    assert price == 0.25
    mock_client.get_products.assert_called_once()


@patch("boto3.client")
def test_get_ec2_pricing_unsupported_region(mock_boto3_client):
    with pytest.raises(ValueError, match="Unsupported region code"):
        get_ec2_pricing("ml.c5.xlarge", "invalid-region")


@patch("boto3.client")
def test_get_ec2_pricing_no_pricing_found(mock_boto3_client):
    mock_client = MagicMock()
    mock_client.get_products.return_value = {"PriceList": []}
    mock_boto3_client.return_value = mock_client

    with pytest.raises(ValueError, match="No pricing data found"):
        get_ec2_pricing("ml.c5.xlarge", "us-east-1")


@patch("src.fmbench.utils.load_config")
@patch("src.fmbench.utils.save_config")
@patch("src.fmbench.scripts.pricing.get_ec2_pricing")
def test_load_and_update_pricing_existing_pricing(
    mock_get_ec2_pricing, mock_save_config, mock_load_config, mock_pricing_data
):
    # Mock the load_config to return existing pricing data
    mock_load_config.return_value = mock_pricing_data

    # Run the function
    updated_pricing = load_and_update_pricing(
        PRICING_YAML_PATH=Path("src/fmbench/configs/pricing.yml"),
        PRICING_FALLBACK_YAML_PATH=Path("src/fmbench/configs/pricing_fallback.yml"),
        instances=["ml.c5.xlarge"],
        region_code="us-east-1",
    )

    # Assert pricing was not fetched again
    mock_get_ec2_pricing.assert_not_called()
    assert updated_pricing == mock_pricing_data


@patch("src.fmbench.utils.load_config")
@patch("src.fmbench.utils.save_config")
@patch("src.fmbench.scripts.pricing.get_ec2_pricing")
def test_load_and_update_pricing_fetch_new_pricing(
    mock_get_ec2_pricing, mock_save_config, mock_load_config, mock_pricing_data
):
    # Mock the load_config to return existing pricing data
    mock_load_config.return_value = mock_pricing_data

    # Mock get_ec2_pricing to return a new price
    mock_get_ec2_pricing.return_value = 0.5

    # Run the function
    updated_pricing = load_and_update_pricing(
        PRICING_YAML_PATH=Path("src/fmbench/configs/pricing.yml"),
        PRICING_FALLBACK_YAML_PATH=Path("src/fmbench/configs/pricing_fallback.yml"),
        instances=["ml.g5.xlarge"],
        region_code="us-east-1",
    )

    # Assert pricing was fetched for the new instance
    mock_get_ec2_pricing.assert_called_once_with("ml.g5.xlarge", "us-east-1")
    assert updated_pricing["pricing"]["instance_based"]["ml.g5.xlarge"] == 0.5
    mock_save_config.assert_called_once()


@patch("src.fmbench.utils.load_config")
@patch("src.fmbench.utils.save_config")
@patch("src.fmbench.scripts.pricing.get_ec2_pricing")
def test_load_and_update_pricing_fallback(
    mock_get_ec2_pricing, mock_save_config, mock_load_config
):
    # Mock load_config to raise an exception for the main pricing file
    mock_load_config.side_effect = [
        FileNotFoundError("Main pricing file not found"),
        {"pricing": {"instance_based": {}}},  # Return fallback data
    ]

    # Run the function
    updated_pricing = load_and_update_pricing(
        PRICING_YAML_PATH=Path("src/fmbench/configs/pricing.yml"),
        PRICING_FALLBACK_YAML_PATH=Path("src/fmbench/configs/pricing_fallback.yml"),
        instances=["ml.g5.xlarge"],
        region_code="us-east-1",
    )

    # Assert fallback pricing was loaded
    assert "instance_based" in updated_pricing["pricing"]
    mock_get_ec2_pricing.assert_called_once_with("ml.g5.xlarge", "us-east-1")
    mock_save_config.assert_called_once()
