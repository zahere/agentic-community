"""
Basic Calculator Tool - Community Edition
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

from typing import Type
from pydantic import BaseModel, Field
import ast
import operator

from agentic_community.core.base import BaseTool, ToolConfig
from agentic_community.core.utils import get_logger

logger = get_logger(__name__)


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate")


class CalculatorTool(BaseTool):
    """Basic calculator tool for community edition."""
    
    # Allowed operators for safety
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    
    def __init__(self):
        """Initialize calculator tool."""
        config = ToolConfig(
            name="calculator",
            description="Perform basic mathematical calculations",
            return_direct=True
        )
        super().__init__(config)
        
    @property
    def input_schema(self) -> Type[BaseModel]:
        """Return input schema."""
        return CalculatorInput
        
    def _safe_eval(self, node):
        """Safely evaluate mathematical expressions."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            return self.ALLOWED_OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand)
            return self.ALLOWED_OPERATORS[type(node.op)](operand)
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
            
    def execute(self, expression: str) -> str:
        """
        Execute a calculation.
        
        Args:
            expression: Mathematical expression
            
        Returns:
            Calculation result as string
        """
        logger.info(f"Calculating: {expression}")
        
        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            
            # Evaluate safely
            result = self._safe_eval(tree.body)
            
            return f"{expression} = {result}"
            
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return f"Error calculating {expression}: {str(e)}"
