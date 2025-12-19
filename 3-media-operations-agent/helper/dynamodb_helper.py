# Copyright 2024 Amazon.com and its affiliates; all rights reserved.
# This file is AWS Content and may not be duplicated or distributed without permission

"""
This module contains a helper class for working with DynamoDB tables.
The DynamoDBHelper class provides a convenient interface for creating, loading, 
deleting, and querying DynamoDB tables.
"""

import boto3
from typing import List
from boto3.dynamodb.conditions import Key


class DynamoDBHelper:
    """Provides an easy to use wrapper for DynamoDB operations."""

    def __init__(self, region_name: str = None):
        """Constructs an instance.
        
        Args:
            region_name (str, optional): AWS region name. If not provided, uses default region.
        """
        self._region = region_name or boto3.Session().region_name
        self._dynamodb_client = boto3.client("dynamodb", region_name=self._region)
        self._dynamodb_resource = boto3.resource("dynamodb", region_name=self._region)

    def create_table(self, table_name: str, pk: str, sk: str) -> None:
        """Creates a DynamoDB table with the specified partition key and sort key.

        Args:
            table_name (str): Name of the table to create
            pk (str): Partition key attribute name
            sk (str): Sort key attribute name
        """
        try:
            table = self._dynamodb_resource.create_table(
                TableName=table_name,
                KeySchema=[
                    {"AttributeName": pk, "KeyType": "HASH"},
                    {"AttributeName": sk, "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": pk, "AttributeType": "S"},
                    {"AttributeName": sk, "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",  # Use on-demand capacity mode
            )

            # Wait for the table to be created
            print(f"Creating table {table_name}...")
            table.wait_until_exists()
            print(f"Table {table_name} created successfully!")
        except self._dynamodb_client.exceptions.ResourceInUseException:
            print(f"Table {table_name} already exists, skipping table creation step")

    def load_dynamodb(self, table_name: str, table_items: List[dict]) -> None:
        """Loads items into a DynamoDB table.

        Args:
            table_name (str): Name of the table to load items into
            table_items (List[dict]): List of items to insert into the table
        """
        try:
            table = self._dynamodb_resource.Table(table_name)
            for item in table_items:
                table.put_item(Item=item)
            print(f"Successfully loaded {len(table_items)} items into table {table_name}")
        except Exception as e:
            print(f"Error loading items into table {table_name}: {e}")

    def delete_table(self, table_name: str) -> None:
        """Deletes a DynamoDB table.

        Args:
            table_name (str): Name of the table to delete
        """
        try:
            table = self._dynamodb_resource.Table(table_name)
            table.delete()

            # Wait for the table to be deleted
            print(f"Deleting table {table_name}...")
            table.wait_until_not_exists()
            print(f"Table {table_name} deleted successfully!")
        except Exception as e:
            print(f"Error deleting table {table_name}: {e}")

    def query_table(
        self,
        table_name: str,
        pk_field: str,
        pk_value: str,
        sk_field: str = None,
        sk_value: str = None,
    ) -> List[dict]:
        """Queries a DynamoDB table.

        Args:
            table_name (str): Name of the table to query
            pk_field (str): Partition key field name
            pk_value (str): Partition key value to query
            sk_field (str, optional): Sort key field name
            sk_value (str, optional): Sort key value prefix to query

        Returns:
            List[dict]: List of items matching the query
        """
        try:
            table = self._dynamodb_resource.Table(table_name)
            
            # Create key condition expression
            if sk_field and sk_value:
                key_expression = Key(pk_field).eq(pk_value) & Key(sk_field).begins_with(sk_value)
            else:
                key_expression = Key(pk_field).eq(pk_value)

            query_data = table.query(KeyConditionExpression=key_expression)
            return query_data["Items"]
        except Exception as e:
            print(f"Error querying table {table_name}: {e}")
            return []
