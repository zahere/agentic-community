"""
Database Tool for Agentic Community Edition

This tool provides database connectivity and operations for various database systems
including PostgreSQL, MySQL, SQLite, and MongoDB.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import re
from enum import Enum

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# For production, import actual database drivers
# import psycopg2
# import mysql.connector
# import sqlite3
# import pymongo


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"


class QueryType(Enum):
    """Types of database queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"


class DatabaseConfig(BaseModel):
    """Configuration for database connection."""
    db_type: DatabaseType = Field(description="Type of database")
    host: Optional[str] = Field(default="localhost", description="Database host")
    port: Optional[int] = Field(default=None, description="Database port")
    database: str = Field(description="Database name")
    username: Optional[str] = Field(default=None, description="Database username")
    password: Optional[str] = Field(default=None, description="Database password")
    connection_string: Optional[str] = Field(default=None, description="Full connection string")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional connection options")


class DatabaseTool(BaseTool):
    """Tool for database operations."""
    
    name: str = "database_tool"
    description: str = """
    A tool for database operations including:
    - Executing SQL queries (SELECT, INSERT, UPDATE, DELETE)
    - Creating and managing tables
    - Data analysis and aggregation
    - Database schema inspection
    - Data import/export
    - Transaction management
    - Safe query building
    """
    
    config: Optional[DatabaseConfig] = None
    connection: Any = None
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        super().__init__()
        self.config = config or self._load_config_from_env()
        self._connection = None
        self._query_history: List[Dict[str, Any]] = []
        
    def _load_config_from_env(self) -> DatabaseConfig:
        """Load database configuration from environment variables."""
        db_type = os.getenv("DB_TYPE", "sqlite")
        
        return DatabaseConfig(
            db_type=DatabaseType(db_type.lower()),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "0")) if os.getenv("DB_PORT") else None,
            database=os.getenv("DB_NAME", "agentic.db"),
            username=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connection_string=os.getenv("DB_CONNECTION_STRING")
        )
        
    def _run(self, query: str) -> str:
        """Run the database tool based on the query."""
        query_lower = query.lower()
        
        # Determine query type
        if query_lower.startswith("select") or "find" in query_lower or "show" in query_lower:
            return self._handle_select_query(query)
        elif query_lower.startswith("insert") or "add" in query_lower or "create record" in query_lower:
            return self._handle_insert_query(query)
        elif query_lower.startswith("update") or "modify" in query_lower or "change" in query_lower:
            return self._handle_update_query(query)
        elif query_lower.startswith("delete") or "remove" in query_lower:
            return self._handle_delete_query(query)
        elif "create table" in query_lower or "create database" in query_lower:
            return self._handle_create_query(query)
        elif "describe" in query_lower or "schema" in query_lower:
            return self._handle_describe_query(query)
        elif "analyze" in query_lower or "statistics" in query_lower:
            return self._handle_analyze_query(query)
        else:
            return self._handle_natural_language_query(query)
            
    async def _arun(self, query: str) -> str:
        """Async version of run."""
        return self._run(query)
        
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                # Simulated SQLite connection
                self._connection = {"type": "sqlite", "connected": True}
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                # In production: psycopg2.connect(...)
                self._connection = {"type": "postgresql", "connected": True}
            elif self.config.db_type == DatabaseType.MYSQL:
                # In production: mysql.connector.connect(...)
                self._connection = {"type": "mysql", "connected": True}
            elif self.config.db_type == DatabaseType.MONGODB:
                # In production: pymongo.MongoClient(...)
                self._connection = {"type": "mongodb", "connected": True}
            else:
                raise ValueError(f"Unsupported database type: {self.config.db_type}")
                
            return True
            
        except Exception as e:
            self._connection = None
            return False
            
    def disconnect(self):
        """Close database connection."""
        if self._connection:
            # In production, properly close the connection
            self._connection = None
            
    def execute_query(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
        fetch_all: bool = True
    ) -> Dict[str, Any]:
        """Execute a database query safely."""
        try:
            # Ensure connection
            if not self._connection:
                if not self.connect():
                    return {
                        "success": False,
                        "error": "Failed to connect to database"
                    }
                    
            # Validate query for safety
            if not self._is_query_safe(query):
                return {
                    "success": False,
                    "error": "Query contains potentially unsafe operations"
                }
                
            # Log query
            self._query_history.append({
                "query": query,
                "params": params,
                "timestamp": datetime.now().isoformat()
            })
            
            # Simulate query execution
            # In production, use actual database cursor
            
            if query.lower().startswith("select"):
                # Simulate SELECT results
                results = [
                    {"id": 1, "name": "John Doe", "email": "john@example.com"},
                    {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
                ]
                
                return {
                    "success": True,
                    "data": results,
                    "row_count": len(results),
                    "columns": ["id", "name", "email"] if results else []
                }
                
            elif query.lower().startswith("insert"):
                return {
                    "success": True,
                    "row_count": 1,
                    "last_insert_id": 3,
                    "message": "Record inserted successfully"
                }
                
            elif query.lower().startswith("update"):
                return {
                    "success": True,
                    "row_count": 1,
                    "message": "Record updated successfully"
                }
                
            elif query.lower().startswith("delete"):
                return {
                    "success": True,
                    "row_count": 1,
                    "message": "Record deleted successfully"
                }
                
            else:
                return {
                    "success": True,
                    "message": "Query executed successfully"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table."""
        try:
            # Simulate schema retrieval
            # In production, query information_schema or equivalent
            
            schema = {
                "table_name": table_name,
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False, "primary_key": True},
                    {"name": "name", "type": "VARCHAR(100)", "nullable": False, "primary_key": False},
                    {"name": "email", "type": "VARCHAR(255)", "nullable": True, "primary_key": False},
                    {"name": "created_at", "type": "TIMESTAMP", "nullable": False, "primary_key": False}
                ],
                "indexes": [
                    {"name": "idx_email", "columns": ["email"], "unique": True}
                ],
                "row_count": 1000
            }
            
            return {
                "success": True,
                "schema": schema
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def build_safe_query(
        self,
        query_type: QueryType,
        table: str,
        conditions: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Tuple[str, Optional[tuple]]:
        """Build a safe parameterized query."""
        params = []
        
        if query_type == QueryType.SELECT:
            col_list = ", ".join(columns) if columns else "*"
            query = f"SELECT {col_list} FROM {table}"
            
            if conditions:
                where_clauses = []
                for key, value in conditions.items():
                    where_clauses.append(f"{key} = %s")
                    params.append(value)
                query += " WHERE " + " AND ".join(where_clauses)
                
            if order_by:
                query += f" ORDER BY {order_by}"
                
            if limit:
                query += f" LIMIT {limit}"
                
        elif query_type == QueryType.INSERT:
            if not data:
                raise ValueError("INSERT requires data")
                
            columns = list(data.keys())
            values = list(data.values())
            
            col_list = ", ".join(columns)
            placeholders = ", ".join(["%s"] * len(columns))
            
            query = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
            params = values
            
        elif query_type == QueryType.UPDATE:
            if not data:
                raise ValueError("UPDATE requires data")
                
            set_clauses = []
            for key, value in data.items():
                set_clauses.append(f"{key} = %s")
                params.append(value)
                
            query = f"UPDATE {table} SET " + ", ".join(set_clauses)
            
            if conditions:
                where_clauses = []
                for key, value in conditions.items():
                    where_clauses.append(f"{key} = %s")
                    params.append(value)
                query += " WHERE " + " AND ".join(where_clauses)
                
        elif query_type == QueryType.DELETE:
            query = f"DELETE FROM {table}"
            
            if conditions:
                where_clauses = []
                for key, value in conditions.items():
                    where_clauses.append(f"{key} = %s")
                    params.append(value)
                query += " WHERE " + " AND ".join(where_clauses)
                
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
            
        return query, tuple(params) if params else None
        
    def analyze_data(
        self,
        table: str,
        column: str,
        operation: str = "stats"
    ) -> Dict[str, Any]:
        """Analyze data in a table."""
        try:
            # Simulate data analysis
            # In production, run actual aggregate queries
            
            if operation == "stats":
                stats = {
                    "count": 1000,
                    "min": 1,
                    "max": 1000,
                    "avg": 500.5,
                    "sum": 500500,
                    "distinct_count": 950
                }
                
                return {
                    "success": True,
                    "table": table,
                    "column": column,
                    "statistics": stats
                }
                
            elif operation == "distribution":
                distribution = [
                    {"value": "A", "count": 300, "percentage": 30.0},
                    {"value": "B", "count": 450, "percentage": 45.0},
                    {"value": "C", "count": 250, "percentage": 25.0}
                ]
                
                return {
                    "success": True,
                    "table": table,
                    "column": column,
                    "distribution": distribution
                }
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown analysis operation: {operation}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def _is_query_safe(self, query: str) -> bool:
        """Check if a query is safe to execute."""
        # Basic safety checks
        dangerous_keywords = [
            "drop database", "drop schema", "truncate",
            "exec", "execute", "xp_", "sp_executesql"
        ]
        
        query_lower = query.lower()
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                return False
                
        # Check for SQL injection patterns
        if re.search(r';\s*(drop|delete|truncate|update)', query_lower):
            return False
            
        return True
        
    def _natural_language_to_sql(self, query: str) -> str:
        """Convert natural language query to SQL."""
        # Simple pattern matching - in production use NLP
        query_lower = query.lower()
        
        # Extract table name
        table_match = re.search(r'from\s+(\w+)', query_lower)
        if not table_match:
            # Try to find table name in common patterns
            if "users" in query_lower:
                table = "users"
            elif "orders" in query_lower:
                table = "orders"
            elif "products" in query_lower:
                table = "products"
            else:
                table = "data"
        else:
            table = table_match.group(1)
            
        # Build SQL based on patterns
        if "count" in query_lower:
            return f"SELECT COUNT(*) FROM {table}"
        elif "average" in query_lower or "avg" in query_lower:
            column_match = re.search(r'(?:average|avg)\s+(\w+)', query_lower)
            column = column_match.group(1) if column_match else "value"
            return f"SELECT AVG({column}) FROM {table}"
        elif "sum" in query_lower or "total" in query_lower:
            column_match = re.search(r'(?:sum|total)\s+(\w+)', query_lower)
            column = column_match.group(1) if column_match else "amount"
            return f"SELECT SUM({column}) FROM {table}"
        elif "maximum" in query_lower or "max" in query_lower:
            column_match = re.search(r'(?:maximum|max)\s+(\w+)', query_lower)
            column = column_match.group(1) if column_match else "value"
            return f"SELECT MAX({column}) FROM {table}"
        elif "minimum" in query_lower or "min" in query_lower:
            column_match = re.search(r'(?:minimum|min)\s+(\w+)', query_lower)
            column = column_match.group(1) if column_match else "value"
            return f"SELECT MIN({column}) FROM {table}"
        else:
            # Default to SELECT *
            limit = 10
            limit_match = re.search(r'(?:top|first|limit)\s+(\d+)', query_lower)
            if limit_match:
                limit = int(limit_match.group(1))
                
            return f"SELECT * FROM {table} LIMIT {limit}"
            
    def _handle_select_query(self, query: str) -> str:
        """Handle SELECT query."""
        result = self.execute_query(query)
        
        if not result["success"]:
            return f"❌ Query failed: {result['error']}"
            
        if not result.get("data"):
            return "No results found"
            
        # Format results as table
        data = result["data"]
        columns = result["columns"]
        
        # Create header
        response = "📊 Query Results:\n\n"
        response += " | ".join(columns) + "\n"
        response += "-" * (len(" | ".join(columns)) + 10) + "\n"
        
        # Add rows
        for row in data[:10]:  # Limit display to 10 rows
            row_values = [str(row.get(col, "")) for col in columns]
            response += " | ".join(row_values) + "\n"
            
        if len(data) > 10:
            response += f"\n... and {len(data) - 10} more rows"
            
        response += f"\n\nTotal rows: {result['row_count']}"
        
        return response
        
    def _handle_insert_query(self, query: str) -> str:
        """Handle INSERT query."""
        result = self.execute_query(query)
        
        if result["success"]:
            return f"✅ {result['message']}\n🔑 New record ID: {result.get('last_insert_id', 'N/A')}"
        else:
            return f"❌ Insert failed: {result['error']}"
            
    def _handle_update_query(self, query: str) -> str:
        """Handle UPDATE query."""
        result = self.execute_query(query)
        
        if result["success"]:
            return f"✅ {result['message']}\n📝 Rows affected: {result['row_count']}"
        else:
            return f"❌ Update failed: {result['error']}"
            
    def _handle_delete_query(self, query: str) -> str:
        """Handle DELETE query."""
        # Add safety confirmation for DELETE
        if "where" not in query.lower():
            return "⚠️ WARNING: DELETE without WHERE clause will remove all records. Please add a WHERE clause to specify which records to delete."
            
        result = self.execute_query(query)
        
        if result["success"]:
            return f"✅ {result['message']}\n🗑️ Rows deleted: {result['row_count']}"
        else:
            return f"❌ Delete failed: {result['error']}"
            
    def _handle_create_query(self, query: str) -> str:
        """Handle CREATE query."""
        result = self.execute_query(query)
        
        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ Create failed: {result['error']}"
            
    def _handle_describe_query(self, query: str) -> str:
        """Handle schema description query."""
        # Extract table name
        table_match = re.search(r'(?:describe|desc|show)\s+(?:table\s+)?(\w+)', query, re.IGNORECASE)
        if not table_match:
            return "Please specify a table name to describe"
            
        table_name = table_match.group(1)
        result = self.get_table_schema(table_name)
        
        if not result["success"]:
            return f"❌ Failed to get schema: {result['error']}"
            
        schema = result["schema"]
        
        response = f"📋 Schema for table '{schema['table_name']}':\n\n"
        response += "Columns:\n"
        
        for col in schema["columns"]:
            pk = " 🔑 PRIMARY KEY" if col["primary_key"] else ""
            nullable = "" if col["nullable"] else " NOT NULL"
            response += f"  • {col['name']} ({col['type']}){nullable}{pk}\n"
            
        if schema.get("indexes"):
            response += "\nIndexes:\n"
            for idx in schema["indexes"]:
                unique = " UNIQUE" if idx.get("unique") else ""
                response += f"  • {idx['name']} on ({', '.join(idx['columns'])}){unique}\n"
                
        response += f"\nRow count: {schema.get('row_count', 'Unknown')}"
        
        return response
        
    def _handle_analyze_query(self, query: str) -> str:
        """Handle data analysis query."""
        # Extract table and column
        match = re.search(r'analyze\s+(\w+)\.(\w+)', query, re.IGNORECASE)
        if not match:
            match = re.search(r'analyze\s+(\w+)', query, re.IGNORECASE)
            if match:
                table = match.group(1)
                column = "*"
            else:
                return "Please specify table or table.column to analyze"
        else:
            table = match.group(1)
            column = match.group(2)
            
        result = self.analyze_data(table, column)
        
        if not result["success"]:
            return f"❌ Analysis failed: {result['error']}"
            
        response = f"📊 Analysis of {table}.{column}:\n\n"
        
        if "statistics" in result:
            stats = result["statistics"]
            response += "Statistics:\n"
            response += f"  • Count: {stats['count']:,}\n"
            response += f"  • Min: {stats['min']}\n"
            response += f"  • Max: {stats['max']}\n"
            response += f"  • Average: {stats['avg']:.2f}\n"
            response += f"  • Sum: {stats['sum']:,}\n"
            response += f"  • Distinct values: {stats['distinct_count']:,}\n"
            
        if "distribution" in result:
            dist = result["distribution"]
            response += "\nValue Distribution:\n"
            for item in dist:
                response += f"  • {item['value']}: {item['count']:,} ({item['percentage']:.1f}%)\n"
                
        return response
        
    def _handle_natural_language_query(self, query: str) -> str:
        """Handle natural language query by converting to SQL."""
        try:
            sql_query = self._natural_language_to_sql(query)
            
            response = f"🔄 Interpreted as SQL: {sql_query}\n\n"
            response += self._handle_select_query(sql_query)
            
            return response
            
        except Exception as e:
            return f"❌ Could not interpret query: {str(e)}\n\nTry using SQL syntax or be more specific."
