from unittest.mock import patch

import pytest
from metadata.generated.schema.entity.data.database import Database
from metadata.generated.schema.entity.data.databaseSchema import DatabaseSchema

from tatm.data.metadata_store.open_metadata_backend import (
    OpenMetadataTatmMetadataStoreBackend,
)


class MockResponse:
    def __init__(self, response="test"):
        self.extension = MockExtension(response)


class MockExtension:
    def __init__(self, response="test"):
        self.root = {"TatmMetadata": response}


class MockConnection:
    def __init__(self, response="test"):
        self.response = response

    def get_by_name(self, *args, **kwargs):
        return MockResponse(self.response)


class MockNullConnection:
    def __init__(self, response="test"):
        self.response = response

    def get_by_name(self, *args, **kwargs):
        return None


class TestOpenMetadataBackend:
    @patch("tatm.data.metadata_store.open_metadata_backend.OpenMetadataConnection")
    @patch("tatm.data.metadata_store.open_metadata_backend.OpenMetadataJWTClientConfig")
    @patch("tatm.data.metadata_store.open_metadata_backend.OpenMetadata")
    def test_object_creation(self, mock_connection, mock_config, mock_api):
        _ = OpenMetadataTatmMetadataStoreBackend(
            address="http://example.com",
            api_key="1234",
        )

    @patch.object(OpenMetadataTatmMetadataStoreBackend, "_get_connection")
    def test_fqn_creation(self, mock_method):
        backend = OpenMetadataTatmMetadataStoreBackend(
            address="http://example.com",
            api_key="1234",
        )

        database_name = backend._construct_fqn("database", Database)
        assert database_name == "tatm-data-service.database"

        tokenized_set_name = backend._construct_fqn(
            "database-tokenized_hello", DatabaseSchema
        )
        assert (
            tokenized_set_name == "tatm-data-service.database.database-tokenized_hello"
        )

    @patch.object(OpenMetadataTatmMetadataStoreBackend, "_get_connection")
    def test_database_lookup(self, mock_method1):
        backend = OpenMetadataTatmMetadataStoreBackend(
            address="http://example.com",
            api_key="1234",
        )
        backend.connection = MockConnection()
        assert backend.lookup("test") == "test"

    @patch.object(OpenMetadataTatmMetadataStoreBackend, "_get_connection")
    @patch.object(
        OpenMetadataTatmMetadataStoreBackend, "_lookup_database", return_value=None
    )
    def test_database_schema_lookup(self, mock_method1, mock_method2):
        backend = OpenMetadataTatmMetadataStoreBackend(
            address="http://example.com",
            api_key="1234",
        )
        backend.connection = MockConnection()
        assert backend.lookup("test") == "test"
        backend._lookup_database.assert_called_once_with("test")

    @patch.object(OpenMetadataTatmMetadataStoreBackend, "_get_connection")
    def test_database_lookup_fail(self, mock_method1):
        backend = OpenMetadataTatmMetadataStoreBackend(
            address="http://example.com",
            api_key="1234",
        )
        backend.connection = MockNullConnection()
        with pytest.raises(KeyError):
            backend.lookup("test")
