import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Search Tool Usage:
- Use the search_course_content tool for questions about specific course content or detailed educational materials
- Use the get_course_outline tool for questions about course structure, lesson lists, or course overviews
- **You can make up to 2 tool calls per query** when needed for complex questions
- Use sequential tool calls when you need to:
  1. First gather information with one tool (e.g., get a course outline)
  2. Then use that information with another tool (e.g., search for related content)
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Course outline questions**: Use the course outline tool and include all course details (title, link, lesson numbers and titles)
- **Complex multi-part questions**: Use sequential tool calls when one search is insufficient
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
  - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str,base_url:str):
        self.client = anthropic.Anthropic(api_key=api_key,base_url=base_url)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_tool_rounds: int = 2) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling with multiple rounds.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool calling rounds (default: 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed, with support for sequential rounds
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager, tools, max_tool_rounds)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager, tools: Optional[List] = None, max_rounds: int = 2):
        """
        Handle execution of tool calls and get follow-up response with support for sequential tool calling.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            tools: Available tools the AI can use for subsequent rounds
            max_rounds: Maximum number of sequential tool calling rounds
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Track the number of tool calling rounds
        current_round = 1
        
        # Process tool calls sequentially until max_rounds is reached or no more tool calls
        while current_round <= max_rounds:
            # Add AI's tool use response to messages
            messages.append({"role": "assistant", "content": initial_response.content})
            
            # Execute all tool calls and collect results
            tool_results = []
            has_tool_calls = False
            
            for content_block in initial_response.content:
                if content_block.type == "tool_use":
                    has_tool_calls = True
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name, 
                            **content_block.input
                        )
                    except Exception as e:
                        # Handle tool execution errors gracefully
                        tool_result = f"Error executing tool: {str(e)}"
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
            
            # Add tool results as single message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            
            # If no tool calls or reached max rounds, prepare final response without tools
            if not has_tool_calls or current_round >= max_rounds:
                # Prepare final API call without tools
                final_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"]
                }
                
                # Get final response
                final_response = self.client.messages.create(**final_params)
                return final_response.content[0].text
            
            # Otherwise, prepare for next round with tools
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"]
            }
            
            # Add tools for next round
            if tools:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}
            
            # Get next response from Claude
            initial_response = self.client.messages.create(**next_params)
            
            # If no more tool calls, return the response
            if initial_response.stop_reason != "tool_use":
                return initial_response.content[0].text
            
            # Increment round counter
            current_round += 1
        
        # This should not be reached due to the loop conditions, but as a fallback
        return "Maximum tool calling rounds reached without final response."