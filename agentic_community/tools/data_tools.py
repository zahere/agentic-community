"""
Data processing tools for CSV and JSON manipulation.

This module provides tools for processing, transforming, and analyzing data.
"""

import json
import csv
import statistics
from typing import Any, Dict, List, Optional, Union, Callable
from collections import defaultdict, Counter
import logging

from agentic_community.core.base import BaseTool
from agentic_community.core.exceptions import (
    ToolExecutionError, ValidationError, handle_error
)
from agentic_community.core.utils.validation import validate_not_empty

logger = logging.getLogger(__name__)


class DataProcessorTool(BaseTool):
    """
    Tool for processing and transforming data in various formats.
    
    Supports filtering, sorting, aggregation, and transformation operations.
    """
    
    name = "data_processor"
    description = "Process and transform data (filter, sort, aggregate, transform)"
    
    def filter_data(self, data: List[Dict], conditions: Dict[str, Any]) -> List[Dict]:
        """
        Filter data based on conditions.
        
        Args:
            data: List of dictionaries to filter
            conditions: Dictionary of field:value conditions
            
        Returns:
            Filtered data
        """
        if not data:
            return []
        
        filtered = []
        for item in data:
            match = True
            for field, condition in conditions.items():
                if field not in item:
                    match = False
                    break
                
                # Handle different condition types
                if isinstance(condition, dict):
                    # Complex conditions like {"$gt": 5, "$lt": 10}
                    item_value = item[field]
                    for op, value in condition.items():
                        if op == "$gt" and not (item_value > value):
                            match = False
                        elif op == "$gte" and not (item_value >= value):
                            match = False
                        elif op == "$lt" and not (item_value < value):
                            match = False
                        elif op == "$lte" and not (item_value <= value):
                            match = False
                        elif op == "$ne" and not (item_value != value):
                            match = False
                        elif op == "$in" and item_value not in value:
                            match = False
                        elif op == "$contains" and value not in str(item_value):
                            match = False
                else:
                    # Simple equality
                    if item[field] != condition:
                        match = False
            
            if match:
                filtered.append(item)
        
        return filtered
    
    def sort_data(self, data: List[Dict], field: str, reverse: bool = False) -> List[Dict]:
        """
        Sort data by a field.
        
        Args:
            data: List of dictionaries to sort
            field: Field to sort by
            reverse: Sort in descending order
            
        Returns:
            Sorted data
        """
        if not data:
            return []
        
        try:
            return sorted(data, key=lambda x: x.get(field, ''), reverse=reverse)
        except Exception as e:
            raise ToolExecutionError(self.name, f"Failed to sort data: {str(e)}")
    
    def aggregate_data(self, data: List[Dict], group_by: str, operations: Dict[str, str]) -> List[Dict]:
        """
        Aggregate data with group by operations.
        
        Args:
            data: List of dictionaries to aggregate
            group_by: Field to group by
            operations: Dictionary of field:operation (sum, avg, count, min, max)
            
        Returns:
            Aggregated data
        """
        if not data:
            return []
        
        # Group data
        groups = defaultdict(list)
        for item in data:
            key = item.get(group_by, 'undefined')
            groups[key].append(item)
        
        # Perform aggregations
        results = []
        for group_key, group_items in groups.items():
            result = {group_by: group_key}
            
            for field, operation in operations.items():
                values = []
                for item in group_items:
                    if field in item and item[field] is not None:
                        try:
                            values.append(float(item[field]))
                        except (ValueError, TypeError):
                            pass
                
                if operation == "sum":
                    result[f"{field}_{operation}"] = sum(values) if values else 0
                elif operation == "avg" or operation == "mean":
                    result[f"{field}_{operation}"] = statistics.mean(values) if values else 0
                elif operation == "count":
                    result[f"{field}_{operation}"] = len(values)
                elif operation == "min":
                    result[f"{field}_{operation}"] = min(values) if values else None
                elif operation == "max":
                    result[f"{field}_{operation}"] = max(values) if values else None
                elif operation == "median":
                    result[f"{field}_{operation}"] = statistics.median(values) if values else 0
                elif operation == "stdev":
                    result[f"{field}_{operation}"] = statistics.stdev(values) if len(values) > 1 else 0
            
            results.append(result)
        
        return results
    
    def transform_data(self, data: List[Dict], transformations: Dict[str, Union[str, Callable]]) -> List[Dict]:
        """
        Transform data fields.
        
        Args:
            data: List of dictionaries to transform
            transformations: Dictionary of field:transformation
            
        Returns:
            Transformed data
        """
        if not data:
            return []
        
        transformed = []
        for item in data:
            new_item = item.copy()
            
            for field, transformation in transformations.items():
                if isinstance(transformation, str):
                    # String transformations
                    if field in new_item:
                        value = str(new_item[field])
                        if transformation == "upper":
                            new_item[field] = value.upper()
                        elif transformation == "lower":
                            new_item[field] = value.lower()
                        elif transformation == "trim":
                            new_item[field] = value.strip()
                        elif transformation.startswith("split:"):
                            delimiter = transformation.split(":", 1)[1]
                            new_item[field] = value.split(delimiter)
                        elif transformation.startswith("replace:"):
                            old, new = transformation.split(":", 2)[1:]
                            new_item[field] = value.replace(old, new)
                elif callable(transformation):
                    # Custom transformation function
                    if field in new_item:
                        new_item[field] = transformation(new_item[field])
            
            transformed.append(new_item)
        
        return transformed
    
    def get_statistics(self, data: List[Dict], field: str) -> Dict[str, float]:
        """
        Get statistics for a numeric field.
        
        Args:
            data: List of dictionaries
            field: Field to analyze
            
        Returns:
            Dictionary of statistics
        """
        if not data:
            return {}
        
        values = []
        for item in data:
            if field in item and item[field] is not None:
                try:
                    values.append(float(item[field]))
                except (ValueError, TypeError):
                    pass
        
        if not values:
            return {"error": "No numeric values found"}
        
        stats = {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        
        if len(values) > 1:
            stats["stdev"] = statistics.stdev(values)
            stats["variance"] = statistics.variance(values)
        
        return stats
    
    def pivot_data(self, data: List[Dict], index: str, columns: str, values: str, aggfunc: str = "sum") -> Dict[str, Dict[str, Any]]:
        """
        Create a pivot table from data.
        
        Args:
            data: List of dictionaries
            index: Field to use as index (rows)
            columns: Field to use as columns
            values: Field to aggregate
            aggfunc: Aggregation function (sum, mean, count, etc.)
            
        Returns:
            Pivot table as nested dictionary
        """
        if not data:
            return {}
        
        # Collect unique column values
        column_values = set()
        for item in data:
            if columns in item:
                column_values.add(item[columns])
        
        # Initialize pivot table
        pivot = defaultdict(lambda: {col: None for col in column_values})
        
        # Group data
        groups = defaultdict(lambda: defaultdict(list))
        for item in data:
            row_key = item.get(index)
            col_key = item.get(columns)
            if row_key and col_key and values in item:
                try:
                    value = float(item[values])
                    groups[row_key][col_key].append(value)
                except (ValueError, TypeError):
                    pass
        
        # Aggregate
        for row_key, col_data in groups.items():
            for col_key, values_list in col_data.items():
                if values_list:
                    if aggfunc == "sum":
                        pivot[row_key][col_key] = sum(values_list)
                    elif aggfunc == "mean" or aggfunc == "avg":
                        pivot[row_key][col_key] = statistics.mean(values_list)
                    elif aggfunc == "count":
                        pivot[row_key][col_key] = len(values_list)
                    elif aggfunc == "min":
                        pivot[row_key][col_key] = min(values_list)
                    elif aggfunc == "max":
                        pivot[row_key][col_key] = max(values_list)
        
        return dict(pivot)
    
    def _invoke(self, params: str) -> str:
        """
        Tool invocation method used by agents.
        
        Expected format: JSON with operation and parameters
        
        Args:
            params: JSON string with operation details
            
        Returns:
            Processed data as JSON string
        """
        try:
            # Parse parameters
            params_dict = json.loads(params)
            operation = params_dict.get('operation', '').lower()
            data = params_dict.get('data', [])
            
            if operation == 'filter':
                conditions = params_dict.get('conditions', {})
                result = self.filter_data(data, conditions)
                return json.dumps({"filtered": result, "count": len(result)}, indent=2)
                
            elif operation == 'sort':
                field = params_dict.get('field', '')
                reverse = params_dict.get('reverse', False)
                result = self.sort_data(data, field, reverse)
                return json.dumps({"sorted": result}, indent=2)
                
            elif operation == 'aggregate':
                group_by = params_dict.get('group_by', '')
                operations = params_dict.get('operations', {})
                result = self.aggregate_data(data, group_by, operations)
                return json.dumps({"aggregated": result}, indent=2)
                
            elif operation == 'transform':
                transformations = params_dict.get('transformations', {})
                result = self.transform_data(data, transformations)
                return json.dumps({"transformed": result}, indent=2)
                
            elif operation == 'statistics':
                field = params_dict.get('field', '')
                result = self.get_statistics(data, field)
                return json.dumps({"statistics": result}, indent=2)
                
            elif operation == 'pivot':
                index = params_dict.get('index', '')
                columns = params_dict.get('columns', '')
                values = params_dict.get('values', '')
                aggfunc = params_dict.get('aggfunc', 'sum')
                result = self.pivot_data(data, index, columns, values, aggfunc)
                return json.dumps({"pivot": result}, indent=2)
                
            else:
                return f"Unknown operation: {operation}. Supported: filter, sort, aggregate, transform, statistics, pivot"
                
        except json.JSONDecodeError as e:
            return f"Invalid JSON parameters: {str(e)}"
        except Exception as e:
            handle_error(e, f"processing data")
            return f"Error processing data: {str(e)}"


class CSVProcessorTool(BaseTool):
    """
    Specialized tool for CSV file processing.
    """
    
    name = "csv_processor"
    description = "Process CSV files with various operations"
    
    def merge_csv(self, csv_data1: List[Dict], csv_data2: List[Dict], key: str) -> List[Dict]:
        """
        Merge two CSV datasets on a common key.
        
        Args:
            csv_data1: First dataset
            csv_data2: Second dataset
            key: Common key field
            
        Returns:
            Merged data
        """
        # Create lookup from second dataset
        lookup = {row[key]: row for row in csv_data2 if key in row}
        
        # Merge
        merged = []
        for row in csv_data1:
            if key in row:
                merged_row = row.copy()
                if row[key] in lookup:
                    # Merge fields from second dataset
                    for field, value in lookup[row[key]].items():
                        if field != key:
                            merged_row[f"{field}_2"] = value
                merged.append(merged_row)
        
        return merged
    
    def clean_csv(self, data: List[Dict], operations: List[str]) -> List[Dict]:
        """
        Clean CSV data with various operations.
        
        Args:
            data: CSV data to clean
            operations: List of cleaning operations
            
        Returns:
            Cleaned data
        """
        cleaned = []
        
        for row in data:
            clean_row = {}
            skip_row = False
            
            for field, value in row.items():
                # Apply cleaning operations
                if "remove_empty" in operations and not value:
                    continue
                
                if "trim" in operations and isinstance(value, str):
                    value = value.strip()
                
                if "remove_duplicates" in operations:
                    # This would need to track seen values
                    pass
                
                if "normalize_case" in operations and isinstance(value, str):
                    value = value.lower()
                
                if "remove_special_chars" in operations and isinstance(value, str):
                    # Remove non-alphanumeric except spaces
                    value = ''.join(c for c in value if c.isalnum() or c.isspace())
                
                clean_row[field] = value
            
            if not skip_row and clean_row:
                cleaned.append(clean_row)
        
        return cleaned
    
    def _invoke(self, params: str) -> str:
        """
        Tool invocation method for CSV operations.
        """
        try:
            params_dict = json.loads(params)
            operation = params_dict.get('operation', '')
            
            if operation == 'merge':
                data1 = params_dict.get('data1', [])
                data2 = params_dict.get('data2', [])
                key = params_dict.get('key', '')
                result = self.merge_csv(data1, data2, key)
                return json.dumps({"merged": result, "count": len(result)}, indent=2)
                
            elif operation == 'clean':
                data = params_dict.get('data', [])
                operations = params_dict.get('operations', [])
                result = self.clean_csv(data, operations)
                return json.dumps({"cleaned": result, "count": len(result)}, indent=2)
                
            else:
                return f"Unknown operation: {operation}. Supported: merge, clean"
                
        except Exception as e:
            handle_error(e, "processing CSV")
            return f"Error processing CSV: {str(e)}"
