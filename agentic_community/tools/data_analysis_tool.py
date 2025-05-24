"""
Data Analysis Tool for Community Edition

Provides data analysis capabilities including statistics, visualization,
and basic machine learning operations.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

from agentic_community.core.base import BaseTool
from agentic_community.core.exceptions import ToolError


class DataAnalysisTool(BaseTool):
    """
    Tool for performing data analysis operations.
    
    Features:
    - Statistical analysis
    - Data transformation
    - Basic visualization preparation
    - Time series analysis
    - Correlation analysis
    """
    
    def __init__(self):
        super().__init__(
            name="DataAnalysis",
            description="Perform data analysis operations"
        )
        self.supported_operations = [
            "describe", "correlate", "aggregate", "transform",
            "time_series", "outliers", "distribution"
        ]
    
    async def analyze(self, 
                     data: Union[List[Dict], pd.DataFrame],
                     operation: str,
                     **kwargs) -> Dict[str, Any]:
        """
        Perform data analysis operation.
        
        Args:
            data: Input data (list of dicts or DataFrame)
            operation: Analysis operation to perform
            **kwargs: Operation-specific parameters
            
        Returns:
            Analysis results
        """
        if operation not in self.supported_operations:
            raise ToolError(f"Unsupported operation: {operation}")
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Perform operation
        if operation == "describe":
            return await self._describe_data(df, **kwargs)
        elif operation == "correlate":
            return await self._calculate_correlations(df, **kwargs)
        elif operation == "aggregate":
            return await self._aggregate_data(df, **kwargs)
        elif operation == "transform":
            return await self._transform_data(df, **kwargs)
        elif operation == "time_series":
            return await self._analyze_time_series(df, **kwargs)
        elif operation == "outliers":
            return await self._detect_outliers(df, **kwargs)
        elif operation == "distribution":
            return await self._analyze_distribution(df, **kwargs)
    
    async def _describe_data(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Get descriptive statistics for the data.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        description = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric statistics
        if numeric_cols:
            numeric_stats = df[numeric_cols].describe()
            description["numeric_summary"] = numeric_stats.to_dict()
        
        # Categorical statistics
        for col in categorical_cols:
            description["categorical_summary"][col] = {
                "unique_values": df[col].nunique(),
                "most_common": df[col].value_counts().head(5).to_dict()
            }
        
        return description
    
    async def _calculate_correlations(self, 
                                    df: pd.DataFrame,
                                    method: str = "pearson",
                                    **kwargs) -> Dict[str, Any]:
        """
        Calculate correlations between numeric columns.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns for correlation"}
        
        corr_matrix = df[numeric_cols].corr(method=method)
        
        # Find strongest correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Threshold for strong correlation
                    strong_correlations.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": float(corr_value)
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": sorted(
                strong_correlations,
                key=lambda x: abs(x["correlation"]),
                reverse=True
            )
        }
    
    async def _aggregate_data(self, 
                            df: pd.DataFrame,
                            group_by: Union[str, List[str]],
                            agg_func: Union[str, Dict[str, str]],
                            **kwargs) -> Dict[str, Any]:
        """
        Aggregate data by groups.
        """
        try:
            if isinstance(agg_func, str):
                result = df.groupby(group_by).agg(agg_func)
            else:
                result = df.groupby(group_by).agg(agg_func)
            
            return {
                "aggregated_data": result.to_dict(),
                "group_counts": df.groupby(group_by).size().to_dict()
            }
        except Exception as e:
            raise ToolError(f"Aggregation failed: {str(e)}")
    
    async def _transform_data(self,
                            df: pd.DataFrame,
                            transformations: List[Dict[str, Any]],
                            **kwargs) -> Dict[str, Any]:
        """
        Apply transformations to the data.
        
        Transformations format:
        [{"column": "col_name", "operation": "normalize|scale|log|diff", "params": {...}}]
        """
        df_transformed = df.copy()
        applied_transforms = []
        
        for transform in transformations:
            col = transform["column"]
            op = transform["operation"]
            params = transform.get("params", {})
            
            if col not in df_transformed.columns:
                continue
            
            if op == "normalize":
                # Min-max normalization
                min_val = df_transformed[col].min()
                max_val = df_transformed[col].max()
                if max_val != min_val:
                    df_transformed[f"{col}_normalized"] = (
                        (df_transformed[col] - min_val) / (max_val - min_val)
                    )
                    applied_transforms.append(f"Normalized {col}")
            
            elif op == "scale":
                # Standard scaling
                mean = df_transformed[col].mean()
                std = df_transformed[col].std()
                if std != 0:
                    df_transformed[f"{col}_scaled"] = (
                        (df_transformed[col] - mean) / std
                    )
                    applied_transforms.append(f"Scaled {col}")
            
            elif op == "log":
                # Log transformation
                if (df_transformed[col] > 0).all():
                    df_transformed[f"{col}_log"] = np.log(df_transformed[col])
                    applied_transforms.append(f"Log transformed {col}")
            
            elif op == "diff":
                # Differencing
                periods = params.get("periods", 1)
                df_transformed[f"{col}_diff"] = df_transformed[col].diff(periods)
                applied_transforms.append(f"Differenced {col} (periods={periods})")
        
        return {
            "transformed_data": df_transformed.to_dict(),
            "applied_transformations": applied_transforms,
            "new_columns": [col for col in df_transformed.columns if col not in df.columns]
        }
    
    async def _analyze_time_series(self,
                                 df: pd.DataFrame,
                                 time_column: str,
                                 value_column: str,
                                 frequency: Optional[str] = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        Analyze time series data.
        """
        try:
            # Convert time column to datetime
            df[time_column] = pd.to_datetime(df[time_column])
            df_sorted = df.sort_values(time_column)
            
            # Basic time series statistics
            ts_stats = {
                "start_date": str(df_sorted[time_column].min()),
                "end_date": str(df_sorted[time_column].max()),
                "duration_days": (df_sorted[time_column].max() - df_sorted[time_column].min()).days,
                "num_observations": len(df_sorted),
                "mean_value": float(df_sorted[value_column].mean()),
                "std_value": float(df_sorted[value_column].std()),
                "trend": self._calculate_trend(df_sorted[value_column].values)
            }
            
            # Moving averages
            if len(df_sorted) > 7:
                df_sorted['ma_7'] = df_sorted[value_column].rolling(window=7).mean()
                ts_stats["moving_avg_7"] = df_sorted['ma_7'].iloc[-1]
            
            if len(df_sorted) > 30:
                df_sorted['ma_30'] = df_sorted[value_column].rolling(window=30).mean()
                ts_stats["moving_avg_30"] = df_sorted['ma_30'].iloc[-1]
            
            return ts_stats
            
        except Exception as e:
            raise ToolError(f"Time series analysis failed: {str(e)}")
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """
        Calculate simple trend direction.
        """
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    async def _detect_outliers(self,
                             df: pd.DataFrame,
                             columns: Optional[List[str]] = None,
                             method: str = "iqr",
                             **kwargs) -> Dict[str, Any]:
        """
        Detect outliers in the data.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == "iqr":
                # Interquartile range method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_indices = df[outlier_mask].index.tolist()
                
                outliers[col] = {
                    "method": "IQR",
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "num_outliers": len(outlier_indices),
                    "outlier_indices": outlier_indices[:10],  # Limit to first 10
                    "outlier_values": df.loc[outlier_indices, col].head(10).tolist()
                }
            
            elif method == "zscore":
                # Z-score method
                threshold = kwargs.get("threshold", 3)
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > threshold
                outlier_indices = df[outlier_mask].index.tolist()
                
                outliers[col] = {
                    "method": "Z-score",
                    "threshold": threshold,
                    "num_outliers": len(outlier_indices),
                    "outlier_indices": outlier_indices[:10],
                    "outlier_values": df.loc[outlier_indices, col].head(10).tolist()
                }
        
        return outliers
    
    async def _analyze_distribution(self,
                                  df: pd.DataFrame,
                                  column: str,
                                  bins: int = 10,
                                  **kwargs) -> Dict[str, Any]:
        """
        Analyze the distribution of a column.
        """
        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")
        
        data = df[column].dropna()
        
        # Basic statistics
        dist_stats = {
            "count": len(data),
            "mean": float(data.mean()),
            "median": float(data.median()),
            "mode": float(data.mode().iloc[0]) if not data.mode().empty else None,
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "skewness": float(data.skew()),
            "kurtosis": float(data.kurtosis())
        }
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90]
        dist_stats["percentiles"] = {
            f"p{p}": float(data.quantile(p/100))
            for p in percentiles
        }
        
        # Histogram
        hist, bin_edges = np.histogram(data, bins=bins)
        dist_stats["histogram"] = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
        }
        
        # Distribution type assessment
        if abs(dist_stats["skewness"]) < 0.5:
            dist_type = "approximately_normal"
        elif dist_stats["skewness"] > 0.5:
            dist_type = "right_skewed"
        else:
            dist_type = "left_skewed"
        
        dist_stats["distribution_type"] = dist_type
        
        return dist_stats
    
    async def process(self, input_data: str) -> str:
        """
        Process data analysis request.
        """
        try:
            # Parse input as JSON
            request = json.loads(input_data)
            
            data = request.get("data", [])
            operation = request.get("operation", "describe")
            params = request.get("params", {})
            
            result = await self.analyze(data, operation, **params)
            
            return json.dumps(result, indent=2)
            
        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            return f"Error: {str(e)}"
