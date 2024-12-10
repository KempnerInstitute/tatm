import argparse
import pathlib
import os

import tatm.data

from utils import metadata_files, tokenized_datasets

# Connection Imports
from metadata.generated.schema.entity.services.connections.metadata.openMetadataConnection import (
    OpenMetadataConnection,
)
from metadata.generated.schema.security.client.openMetadataJWTClientConfig import (
    OpenMetadataJWTClientConfig,
)
from metadata.ingestion.ometa.ometa_api import OpenMetadata

# Tag and Classification Imports
from metadata.generated.schema.api.classification.createClassification import (
    CreateClassificationRequest,
)
from metadata.generated.schema.api.classification.createTag import CreateTagRequest
from metadata.generated.schema.type.tagLabel import TagLabel
from metadata.generated.schema.entity.classification.classification import (
    Classification,
)

# Custom Property Imports
from metadata.ingestion.models.custom_properties import OMetaCustomProperties
from metadata.ingestion.models.custom_properties import CustomPropertyDataTypes
from metadata.generated.schema.type.customProperty import PropertyType
from metadata.generated.schema.api.data.createCustomProperty import (
    CreateCustomPropertyRequest,
)

# DatabaseService Imports, used to represent the general tatm data service
from metadata.generated.schema.entity.services.databaseService import (
    DatabaseService,
    DatabaseServiceType,
)
from metadata.generated.schema.api.services.createDatabaseService import (
    CreateDatabaseServiceRequest,
)

# Database Imports, used to represent the specific data sets
from metadata.generated.schema.entity.data.database import Database
from metadata.generated.schema.api.data.createDatabase import CreateDatabaseRequest

# DatabaseSchema Imports, used to represent tokenized data within the data sets, could also be used to represent corpuses? TBD
from metadata.generated.schema.entity.data.databaseSchema import DatabaseSchema
from metadata.generated.schema.api.data.createDatabaseSchema import (
    CreateDatabaseSchemaRequest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Open Metadata backend")
    parser.add_argument(
        "--host", type=str, help="Host of the metadata store", default=None
    )
    parser.add_argument(
        "--port", type=int, help="Port of the metadata store", default=None
    )
    parser.add_argument(
        "--api_key", type=str, help="API key for the metadata store", default=None
    )
    parser.add_argument(
        "--dirs",
        type=str,
        help="Comma seperated list of directories containing metadata files",
    )

    return parser.parse_args()


def init_open_metadata_backend(host: str, port: int, api_key: str):
    """Initialize the Open Metadata backend connection and test the connection

    Args:
        host: address of the Open Metadata backend, defaults to OPEN_METADATA_HOST environment variable if not provided via CLI
        port: port of the Open Metadata backend, defaults to OPEN_METADATA_PORT environment variable if not provided via CLI. If not set, port 80 is used.
        api_key: API key for the Open Metadata backend, defaults to OPEN_METADATA_API_KEY environment variable if not provided via CLI


    Returns:
        OpenMetadata: Open Metadata connection object that can be used to interact with the Open Metadata backend
    """
    if api_key is None:
        api_key = os.getenv("OPEN_METADATA_API_KEY")
    if api_key is None:
        raise ValueError(
            "API key not provided and OPEN_METADATA_API_KEY environment variable not set"
        )

    address = construct_address(host, port)
    connection_config = OpenMetadataConnection(
        hostPort=address,
        authProvider="openmetadata",
        securityConfig=OpenMetadataJWTClientConfig(jwtToken=api_key),
    )

    connection = OpenMetadata(connection_config)
    if not connection.health_check():
        raise ValueError("Could not connect to Open Metadata backend at {address}")

    return connection


def construct_address(host: str, port: int) -> str:
    """Construct the address of the Open Metadata backend

    Args:
        host: address of the Open Metadata backend, either an IP address or a domain name
        port: port of the Open Metadata backend, if not provided, port 80 is used (implicit HTTP)

    Returns:
        str: formatted address of the Open Metadata server
    """
    if host is None:
        host = os.getenv("OPEN_METADATA_HOST")
    if host is None:
        host = "localhost"

    if host[0:4] != "http":
        address = f"http://{host}"
    else:
        address = host

    if port is not None:
        address = f"{address}:{port}"
    return f"{address}/api"


# Classification/Tag Functions


def setup_classifications(connection: OpenMetadata):
    """Make API calls to create the necessary classifications in the Open Metadata backend



    Args:
        connection: Initialized Open Metadata connection object
    """

    classifications = {
        "DataFocus": "General purpose of the data in the dataset (i.e. code, math, general text, vision, etc)",
        "Tokenizer": "The HF name of the tokenizer used to generate the data",
    }
    for class_name, description in classifications.items():
        create_classification(connection, class_name, description)


def create_classification(
    connection: OpenMetadata, classifcation_name: str, description: str
) -> Classification:
    """Create a classification (TagType) in the Open Metadata backend

    Args:
        connection: Initialized Open Metadata connection object
        classifcation_name: Name of the classification
        description: Description of the classification

    Returns:
        Classification: The created classification object
    """
    request = CreateClassificationRequest(
        name=classifcation_name, description=description
    )
    return connection.create_or_update(data=request)


def create_tag_value(
    connection: OpenMetadata,
    classification_name: str,
    tag_name: str,
    description: str = "",
):
    """Create a tag value within a classification in the Open Metadata backend

    Args:
        connection: Initialized Open Metadata connection object
        classification_name: Name of the classification to create the tag value in
        tag_name: Name of the tag value to create
        description: Description of the particular tag value. Defaults to "".
    """
    classification = connection.get_by_name(Classification, classification_name)
    request = CreateTagRequest(
        classification=classification.fullyQualifiedName,
        name=tag_name,
        description=description,
    )
    connection.create_or_update(data=request)


def tag_label(parent: str, value: str) -> TagLabel:
    """Create a TagLabel object for a given classification and tag value

    Args:
        parent: Classification name
        value: value of the tag

    Returns:
        TagLabel: TagLabel object representing the classification and tag value
    """
    return TagLabel(
        tagFQN=f"{parent}.{value}",
        source="Classification",
        labelType="Manual",
        state="Confirmed",
    )


# Custom Property Functions


def setup_custom_properties(connection: OpenMetadata):
    """Create the necessary custom properties in the Open Metadata backend for the Database and Database Schema entities used to represent the data

    Args:
        connection: Initialized Open Metadata connection object
    """
    custom_prop_str_type = connection.get_custom_property_type(
        CustomPropertyDataTypes.STRING
    )

    pt = PropertyType(id=custom_prop_str_type.id, type="type")
    create_tatm_metadata_request = CreateCustomPropertyRequest(
        name="TatmMetadata", propertyType=pt, description="The metadata of the dataset"
    )

    for resource in [Database, DatabaseSchema]:
        ometa_request = OMetaCustomProperties(
            entity_type=resource,
            createCustomPropertyRequest=create_tatm_metadata_request,
        )
        connection.create_or_update_custom_property(ometa_request)


# Data Processing Functions


def process_dataset(
    connection: OpenMetadata,
    service_entity: DatabaseService,
    data_genre: str,
    data_dir: str | pathlib.Path,
    tatm_metadata: tatm.data.TatmDataMetadata,
):
    """Process a dataset and load it into the Open Metadata backend, creating the necessary entities and relationships along the way

    Args:
        connection: Initialized Open Metadata connection object
        service_entity: DatabaseService object representing the general Tatm Data Service
        data_genre: Genre of the data in the dataset (i.e. code, math, general text, vision, etc)
        data_dir: Path to the directory containing the dataset
        tatm_metadata: TatmDataMetadata object representing the metadata of the dataset
    """

    create_tag_value(connection, "DataFocus", data_genre)

    create_database_request = CreateDatabaseRequest(
        name=tatm_metadata.name,
        service=service_entity.fullyQualifiedName,
        description=tatm_metadata.description,
        extension={"TatmMetadata": tatm_metadata.as_json()},
        tags=[tag_label("DataFocus", data_genre)],
    )
    db_entity = connection.create_or_update(data=create_database_request)

    for corpus in tatm_metadata.corpuses:
        create_db_schema_request = CreateDatabaseSchemaRequest(
            name=corpus,
            database=db_entity.fullyQualifiedName,
            description="corpus within the dataset representing a specific subset of the data",
        )

        connection.create_or_update(data=create_db_schema_request)

    for data_path, tokenized_data in tokenized_datasets(data_dir):
        process_tokenized_dataset(
            connection, db_entity, tatm_metadata, data_path, tokenized_data
        )


def process_tokenized_dataset(
    connection: OpenMetadata,
    db_entity: Database,
    parent_metadata: tatm.data.TatmDataMetadata,
    data_path: pathlib.Path,
    tokenized_data: tatm.data.TatmDataMetadata,
):
    """Process a tokenized dataset and load it into the Open Metadata backend, creating the necessary entities and relationships along

    Args:
        connection: Initialized Open Metadata connection object
        db_entity: Database object representing the parent dataset
        parent_metadata: Metadata of the parent dataset
        data_path: Path to the directory containing the tokenized data
        tokenized_data: Metadata of the tokenized data
    """
    if not isinstance(data_path, pathlib.Path):
        data_path = pathlib.Path(data_path)
    create_tag_value(connection, "Tokenizer", tokenized_data.tokenized_info.tokenizer)
    create_db_schema_request = CreateDatabaseSchemaRequest(
        name=f"{parent_metadata.name}-tokenized-{tokenized_data.tokenized_info.tokenizer}-{data_path.parts[-1]}",
        database=db_entity.fullyQualifiedName,
        description="tokenized data within the dataset",
        extension={"TatmMetadata": tokenized_data.as_json()},
        tags=[tag_label("Tokenizer", tokenized_data.tokenized_info.tokenizer)],
    )
    connection.create_or_update(data=create_db_schema_request)


def create_tatm_data_service(
    connection: OpenMetadata, description: str
) -> DatabaseService:
    """Create the Tatm Data Service in the Open Metadata backend

    Args:
        connection: Initialized Open Metadata connection object
        description: Description of the Tatm Data Service

    Raises:
        RuntimeError: If the Tatm Data Service could not be created

    Returns:
        DatabaseService: The created Tatm Data Service
    """
    create_database_service_request = CreateDatabaseServiceRequest(
        name="tatm-data-service",
        serviceType=DatabaseServiceType.CustomDatabase,
        description=description,
    )
    service_entity = connection.create_or_update(data=create_database_service_request)
    if not service_entity:
        raise RuntimeError("Could not create the Tatm Data Service")
    return service_entity


def main():
    args = parse_args()
    metadata_connection = init_open_metadata_backend(args.host, args.port, args.api_key)

    setup_classifications(metadata_connection)
    setup_custom_properties(metadata_connection)

    data_service = create_tatm_data_service(
        metadata_connection, "Top Level Tatm Data Service"
    )
    metadata_dirs = args.dirs.split(",")
    for dir in metadata_dirs:
        for genre, path, metadata in metadata_files(dir):
            print(f"Processing {metadata.name} located at {path}")
            process_dataset(metadata_connection, data_service, genre, path, metadata)


if __name__ == "__main__":
    main()
