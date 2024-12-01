import json
import yaml
import boto3
import logging
from pathlib import Path
from fmbench.utils import load_config
from typing import Dict, Union, List

logger = logging.getLogger(__name__)

# YAML map dictionary for region codes to human-readable location names
REGION_MAP = {
    "us-east-1": "US East (N. Virginia)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "eu-west-1": "EU (Ireland)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "sa-east-1": "South America (Sao Paulo)",
}


def get_ec2_pricing(
    instance_type: str,
    region_code: str,
    operating_system: str = "Linux",
    tenancy: str = "Shared",
) -> float:
    """
    Retrieve on-demand pricing for a specified EC2 instance type and region.

    Parameters:
    - instance_type: e.g., "t3.micro"
    - region_code: e.g., "us-east-1", "us-west-2"
    - operating_system: e.g., "Linux" (default)
    - tenancy: e.g., "Shared" (default) or "Dedicated"

    Returns:
    - Price in USD (float) for the specified instance type and region.
    """
    # Convert region_code to human-readable region name
    region_name = REGION_MAP.get(region_code)
    if not region_name:
        raise ValueError(f"Unsupported region code: {region_code}")

    # Filters for the product
    filters = [
        {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
        {"Type": "TERM_MATCH", "Field": "location", "Value": region_name},
        {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": operating_system},
        {"Type": "TERM_MATCH", "Field": "tenancy", "Value": tenancy},
        {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
        {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
    ]

    # Create a Boto3 client for the Pricing service
    client = boto3.client("pricing", region_name="us-east-1")

    try:
        # Call AWS Pricing API
        response = client.get_products(
            ServiceCode="AmazonEC2", Filters=filters, FormatVersion="aws_v1"
        )

        # Parse response to extract pricing
        for price_item in response["PriceList"]:
            price_data = json.loads(price_item)
            for term in price_data["terms"]["OnDemand"].values():
                for price_dimension in term["priceDimensions"].values():
                    price_per_hour = float(price_dimension["pricePerUnit"]["USD"])
                    return price_per_hour

        # Raise an error if no pricing data is found
        raise ValueError(f"No pricing data found for {instance_type} in {region_code}")

    except boto3.exceptions.Boto3Error as e:
        logger.error(
            f"Boto3 client error while fetching pricing for {instance_type}: {e}"
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error while fetching pricing for {instance_type}: {e}"
        )
        raise


def load_and_update_pricing(
    PRICING_YAML_PATH: Union[Path, str],
    PRICING_FALLBACK_YAML_PATH: Union[Path, str],
    instances: List,
    region_code: str = 'us-east-1',
) -> dict:
    """
    EC2 pricing function which checks if the pricing for a given instance type exists in pricing.yml.
    If it does, it skips fetching. If not, it fetches the pricing using `get_ec2_pricing` and updates pricing.yml.
    If fetching fails, it falls back to the fallback pricing file.

    Args:
        PRICING_YAML_PATH (Union[Path, str]): Path to the pricing YAML file or S3 URI.
        PRICING_FALLBACK_YAML_PATH (Union[Path, str]): Path to the fallback YAML file or S3 URI.
        instances (List): List of instances in experiments to fetch pricing for (e.g., "t2.micro").
        region_code (str): The AWS region code (default is us-east-1).
        
    Returns:
        dict: A dictionary containing updated EC2 pricing data.
    """
    try:
        # Load existing pricing data using the `load_config` function
        pricing_data = load_config(PRICING_YAML_PATH)
        logger.info(f"Loaded pricing data from {PRICING_YAML_PATH}")
    except Exception as e:
        logger.error(f"Error loading pricing YAML from {PRICING_YAML_PATH}: {e}")
        logger.warning("Attempting to load fallback pricing YAML.")
        try:
            pricing_data = load_config(PRICING_FALLBACK_YAML_PATH)
            logger.warning(f"Falling back to {PRICING_FALLBACK_YAML_PATH}")
        except Exception as fallback_error:
            logger.critical(f"Failed to load fallback pricing YAML: {fallback_error}")
            raise FileNotFoundError(
                "Pricing data could not be loaded from any source."
            ) from fallback_error

    for instance_type in instances:
        # Check if the instance type already exists under 'pricing > instance_based'
        if "pricing" in pricing_data and "instance_based" in pricing_data["pricing"]:
            if instance_type in pricing_data["pricing"]["instance_based"]:
                logger.info(f"Pricing for {instance_type} already exists. Skipping fetch.")
                continue
        if "token_based" in pricing_data["pricing"] and instance_type in pricing_data["pricing"]["token_based"]:
            logger.info(f"Token-based pricing for {instance_type} already exists. Skipping fetch.")
            continue  # Skip fetching for this instance type

        # Fetch pricing using `get_ec2_pricing`
        logger.info(
            f"Fetching pricing for instance type: {instance_type} in region: {region_code}"
        )
        
        try:
            price = get_ec2_pricing(instance_type, region_code)
            if price:
                pricing_data["pricing"]["instance_based"][instance_type] = price
                logger.info(f"Fetched pricing for {instance_type}: {price} USD")
            else:
                raise ValueError(
                    f"No pricing data found for {instance_type} in {region_code}"
                )
        except Exception as e:
            logger.error(f"Error fetching pricing: {e}")
            logger.warning(
                "Fetching pricing failed. Falling back to fallback pricing data."
            )
            try:
                pricing_data = load_config(PRICING_FALLBACK_YAML_PATH)
                logger.warning(
                    f"Fallback pricing data loaded from {PRICING_FALLBACK_YAML_PATH}"
                )
            except Exception as fallback_error:
                logger.critical(
                    f"Failed to load fallback pricing YAML after fetch error: {fallback_error}"
                )
                raise FileNotFoundError(
                    "Failed to retrieve pricing and fallback pricing data."
                ) from fallback_error

    return pricing_data
