import logging

from metadata.generated.schema.entity.data.database import Database
from metadata.generated.schema.entity.data.databaseSchema import DatabaseSchema

# Connection Imports
from metadata.generated.schema.entity.services.connections.metadata.openMetadataConnection import (
    OpenMetadataConnection,
)
from metadata.generated.schema.security.client.openMetadataJWTClientConfig import (
    OpenMetadataJWTClientConfig,
)
from metadata.ingestion.ometa.ometa_api import OpenMetadata

from tatm.data.metadata_store.metadata_backend import TatmMetadataStoreBackend

LOGGER = logging.getLogger(__name__)


class OpenMetadataTatmMetadataStoreBackend(TatmMetadataStoreBackend):
    """Metadata store backend that stores metadata as JSON files."""

    def __init__(
        self,
        address: str,
        api_key: str,
        data_service_name="tatm-data-service",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.address = address
        self.api_key = api_key
        self.data_service_name = data_service_name
        self.connection = self._get_connection()

    def lookup(self, name: str) -> str:
        """Lookup metadata for a dataset by name.
        Should return a string containing a json representation of the metadata.

        Args:
            name: Name of the dataset to lookup.

        Raises:
            KeyError: If metadata is not found for the dataset.

        Returns:
            str: JSON representation of the metadata.
        """
        lookup_response = self._lookup_database(name)
        if lookup_response is None:
            LOGGER.info(
                f"top level Dataset Metadata not found for {name}, trying to lookup tokenized metadata"
            )
            lookup_response = self._lookup_database_schema(name)
        if lookup_response is None:
            raise KeyError(f"Metadata not found for dataset: {name}")
        else:
            return lookup_response

    def _get_connection(self) -> OpenMetadata:
        """Connect to the Open Metadata backend.

        Raises:
            ValueError: If the connection could not be established.

        Returns:
            OpenMetadata: Connection to the Open Metadata
        """

        connection_config = OpenMetadataConnection(
            hostPort=self.address,
            authProvider="openmetadata",
            securityConfig=OpenMetadataJWTClientConfig(jwtToken=self.api_key),
        )
        connection = OpenMetadata(connection_config)

        if not connection.health_check():
            raise ValueError("Could not connect to Open Metadata backend at {address}")

        return connection

    def _lookup_database(self, name: str) -> str:
        """Lookup metadata for a database by name.

        Args:
            name: Name of the database to lookup.

        Returns:
            str: JSON representation of the metadata.
        """
        fqn = self._construct_fqn(name, Database)
        out = self.connection.get_by_name(Database, fqn=fqn, fields=["extension"])
        if out is None:
            return None
        try:
            return out.extension.root["TatmMetadata"]
        except Exception as e:
            LOGGER.error(f"Error parsing metadata for {name}: {e}")
            return None

    def _lookup_database_schema(self, name: str) -> str:
        """Lookup metadata for a database schema by name.

        Args:
            name: Name of the database to lookup.
        Returns:
            str: JSON representation of the metadata.
        """
        fqn = self._construct_fqn(name, DatabaseSchema)
        out = self.connection.get_by_name(DatabaseSchema, fqn=fqn, fields=["extension"])
        if out is None:
            return None
        try:
            return out.extension.root["TatmMetadata"]
        except Exception as e:
            LOGGER.error(f"Error parsing metadata for {name}: {e}")
            return None

    def _construct_fqn(self, name: str, entity_type: type) -> str:
        """Construct a fully qualified name for a database or schema.

        Args:
            name: Name of the database or schema.
            entity_type: Type of the openmetadata entity being queried.

        Returns:
            str: Fully qualified name.
        """
        if entity_type is Database:
            return f"{self.data_service_name}.{name}"
        elif entity_type is DatabaseSchema:
            database_name = name.split("_")[0].split("-tokenized")[0]
            return f"{self.data_service_name}.{database_name}.{name}"
