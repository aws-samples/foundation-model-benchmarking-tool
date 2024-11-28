import json
import yaml
import boto3
import logging
from pathlib import Path
from fmbench import globals

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


def save_region_map_to_yaml(yaml_path: Path):
    """
    Save the region map dictionary to a YAML file.
    """
    with yaml_path.open("w") as f:
        yaml.dump(REGION_MAP, f)


def load_region_map_from_yaml(yaml_path: Path) -> dict:
    """
    Load the region map dictionary from a YAML file.
    """
    with yaml_path.open("r") as f:
        return yaml.safe_load(f)


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


def load_pricing(
    PRICING_YAML_PATH: str,
    PRICING_FALLBACK_YAML_PATH: str,
    instance_type: str,
    region_code: str = globals.region_name,
    operating_system: str = "Linux",
    tenancy: str = "Shared",
) -> dict:
    """
    EC2 pricing function which checks if the pricing for a given instance type exists in pricing.yml.
    If it does, it skips fetching. If not, it fetches the pricing using `get_ec2_pricing` and updates pricing.yml.

    Args:
        PRICING_YAML_PATH (str): Path to the pricing YAML file.
        PRICING_FALLBACK_YAML_PATH (str): Path to the fallback YAML file.
        instance_type (str): The EC2 instance type to fetch pricing for (e.g., "t2.micro").
        region_code (str): The AWS region code (default is globals.region_name).
        operating_system (str): The operating system for the instance (default is "Linux").
        tenancy (str): The tenancy type for the instance (default is "Shared").

    Returns:
        dict: A dictionary containing updated EC2 pricing data.
    """
    # Convert string paths to Path objects
    pricing_yaml_path = Path(PRICING_YAML_PATH)
    fallback_yaml_path = Path(PRICING_FALLBACK_YAML_PATH)

    try:
        # Load existing pricing data
        if pricing_yaml_path.exists():
            try:
                with pricing_yaml_path.open("r") as f:
                    pricing_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded existing pricing data from {PRICING_YAML_PATH}")
            except yaml.YAMLError as e:
                logger.error(f"Error reading pricing YAML file: {e}")
                pricing_data = {}
        else:
            pricing_data = {}
            logger.warning(
                f"No existing pricing data found. Creating new file at {PRICING_YAML_PATH}"
            )

        # Check if the instance type already exists
        if instance_type in pricing_data:
            logger.info(f"Pricing for {instance_type} already exists. Skipping fetch.")
            return pricing_data

        # Fetch pricing using get_ec2_pricing function
        logger.info(
            f"Fetching pricing for instance type: {instance_type} in region: {region_code}"
        )
        try:
            price = get_ec2_pricing(
                instance_type, region_code, operating_system, tenancy
            )
            if price:
                pricing_data[instance_type] = price
                logger.info(f"Fetched pricing for {instance_type}: {price} USD")
            else:
                logger.warning(
                    f"No pricing data found for {instance_type} in {region_code}"
                )
        except ValueError as e:
            logger.error(f"Error fetching pricing: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching pricing: {e}")
            raise

        # Save updated pricing data to PRICING_YAML_PATH
        try:
            with pricing_yaml_path.open("w") as f:
                yaml.dump(pricing_data, f)
            logger.info(f"Updated pricing data saved to {PRICING_YAML_PATH}")
        except Exception as e:
            logger.error(f"Error saving pricing data to YAML file: {e}")
            raise

        return pricing_data

    except Exception as e:
        # Log the error and fallback to the pricing fallback file
        logger.error(f"Error processing pricing data for {instance_type}: {e}")
        if fallback_yaml_path.exists():
            try:
                with fallback_yaml_path.open("r") as f:
                    fallback_data = yaml.safe_load(f)
                logger.warning(f"Falling back to {PRICING_FALLBACK_YAML_PATH}")
                return fallback_data
            except yaml.YAMLError as yaml_error:
                logger.error(f"Error reading fallback YAML file: {yaml_error}")
                raise
        else:
            logger.critical("Fallback pricing file not found.")
            raise FileNotFoundError("Fallback pricing file not found.") from e
