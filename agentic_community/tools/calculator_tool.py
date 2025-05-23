"""
Basic Calculator Tool - Community Edition
Copyright (c) 2025 Zaher Khateeb
Licensed under Apache License 2.0
"""

import ast
import operator
import re

from agentic_community.core.base import BaseTool
from agentic_community.core.utils import get_logger

logger = get_logger(__name__)


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
        super().__init__(
            name="calculator",
            description="Perform basic mathematical calculations",
            keywords=["calculate", "compute", "sum", "add", "subtract", "multiply", "divide", "math"]
        )
        
    def _extract_expression(self, input_text: str) -> str:
        """Extract mathematical expression from input text."""
        # Try to find mathematical expressions
        patterns = [
            r"calculate\s+(.+)",
            r"compute\s+(.+)",
            r"what is\s+(.+)",
            r"(\d+[\s\+\-\*/\^]+\d+.*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, input_text.lower())
            if match:
                return match.group(1).strip()
                
        # If no pattern matches, assume the whole input is the expression
        return input_text.strip()
        
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
            
    def run(self, input_text: str) -> str:
        """
        Execute a calculation.
        
        Args:
            input_text: Text containing mathematical expression
            
        Returns:
            Calculation result as string
        """
        # Extract expression from input text
        expression = self._extract_expression(input_text)
        
        # Clean up the expression
        expression = expression.replace("^", "**")  # Convert ^ to Python power operator
        
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
