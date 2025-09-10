import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator

class TestAIGenerator(unittest.TestCase):
    """Test cases for AIGenerator class with sequential tool calling"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock Anthropic client
        self.mock_client = MagicMock()
        
        # Create AIGenerator with mocked client
        with patch('anthropic.Anthropic', return_value=self.mock_client):
            self.ai_generator = AIGenerator(api_key="test_key", model="test_model", base_url="test_url")
            self.ai_generator.client = self.mock_client
        
        # Mock tool manager
        self.mock_tool_manager = MagicMock()
        
        # Sample tools for testing
        self.test_tools = [
            {
                "name": "test_tool_1",
                "description": "Test tool 1",
                "input_schema": {"type": "object", "properties": {}}
            },
            {
                "name": "test_tool_2",
                "description": "Test tool 2",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
    
    def test_direct_response_no_tools(self):
        """Test direct response without tool use"""
        # Mock response with direct content
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Test response")]
        self.mock_client.messages.create.return_value = mock_response
        
        # Call generate_response
        response = self.ai_generator.generate_response("Test query")
        
        # Verify response
        self.assertEqual(response, "Test response")
        self.mock_client.messages.create.assert_called_once()
    
    def test_single_tool_call(self):
        """Test single tool call and response"""
        # Mock initial response with tool use
        mock_initial_response = MagicMock()
        mock_initial_response.stop_reason = "tool_use"
        
        # Create tool use content block
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "test_tool_1"
        mock_tool_block.id = "tool_1_id"
        mock_tool_block.input = {"param": "value"}
        
        mock_initial_response.content = [mock_tool_block]
        
        # Mock final response after tool use
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock(text="Final response after tool use")]
        
        # Set up client to return different responses
        self.mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        # Mock tool execution result
        self.mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        # Call generate_response with tools
        response = self.ai_generator.generate_response(
            "Test query with tool", 
            tools=self.test_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify response
        self.assertEqual(response, "Final response after tool use")
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        self.mock_tool_manager.execute_tool.assert_called_once_with("test_tool_1", param="value")
    
    def test_sequential_tool_calls(self):
        """Test sequential tool calls with multiple rounds"""
        # Mock first response with tool use
        mock_first_response = MagicMock()
        mock_first_response.stop_reason = "tool_use"
        
        # Create first tool use content block
        mock_tool_block1 = MagicMock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "test_tool_1"
        mock_tool_block1.id = "tool_1_id"
        mock_tool_block1.input = {"param1": "value1"}
        
        mock_first_response.content = [mock_tool_block1]
        
        # Mock second response with another tool use
        mock_second_response = MagicMock()
        mock_second_response.stop_reason = "tool_use"
        
        # Create second tool use content block
        mock_tool_block2 = MagicMock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "test_tool_2"
        mock_tool_block2.id = "tool_2_id"
        mock_tool_block2.input = {"param2": "value2"}
        
        mock_second_response.content = [mock_tool_block2]
        
        # Mock final response after tool use
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock(text="Final response after sequential tool use")]
        
        # Set up client to return different responses
        self.mock_client.messages.create.side_effect = [mock_first_response, mock_second_response, mock_final_response]
        
        # Mock tool execution results
        self.mock_tool_manager.execute_tool.side_effect = ["Tool 1 result", "Tool 2 result"]
        
        # Call generate_response with tools
        response = self.ai_generator.generate_response(
            "Test query with sequential tools", 
            tools=self.test_tools,
            tool_manager=self.mock_tool_manager,
            max_tool_rounds=2
        )
        
        # Verify response
        self.assertEqual(response, "Final response after sequential tool use")
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        
        # Verify tool calls
        self.mock_tool_manager.execute_tool.assert_any_call("test_tool_1", param1="value1")
        self.mock_tool_manager.execute_tool.assert_any_call("test_tool_2", param2="value2")
    
    def test_max_rounds_limit(self):
        """Test that max_rounds limit is respected"""
        # Mock responses with tool use
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        
        # Create tool use content block
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "test_tool_1"
        mock_tool_block.id = "tool_id"
        mock_tool_block.input = {"param": "value"}
        
        mock_response.content = [mock_tool_block]
        
        # Mock final response
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock(text="Final response after max rounds")]
        
        # Set up client to return tool use responses for first two calls, then final response
        self.mock_client.messages.create.side_effect = [mock_response, mock_response, mock_final_response]
        
        # Mock tool execution result
        self.mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        # Call generate_response with max_tool_rounds=1
        response = self.ai_generator.generate_response(
            "Test query with tool", 
            tools=self.test_tools,
            tool_manager=self.mock_tool_manager,
            max_tool_rounds=1
        )
        
        # Verify response and that only one round of tool calls was made
        self.assertEqual(response, "Final response after max rounds")
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 1)
    
    def test_tool_execution_error_handling(self):
        """Test graceful handling of tool execution errors"""
        # Mock response with tool use
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        
        # Create tool use content block
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "test_tool_1"
        mock_tool_block.id = "tool_id"
        mock_tool_block.input = {"param": "value"}
        
        mock_response.content = [mock_tool_block]
        
        # Mock final response
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock(text="Final response after error")]
        
        # Set up client to return different responses
        self.mock_client.messages.create.side_effect = [mock_response, mock_final_response]
        
        # Mock tool execution to raise an exception
        self.mock_tool_manager.execute_tool.side_effect = Exception("Test tool execution error")
        
        # Call generate_response with tools
        response = self.ai_generator.generate_response(
            "Test query with tool error", 
            tools=self.test_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify response
        self.assertEqual(response, "Final response after error")
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

if __name__ == "__main__":
    unittest.main()