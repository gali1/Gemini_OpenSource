"""
Model interface for Gemini OpenSource
Bridges the CLI with the PyTorch Gemini model implementation
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import re
import asyncio
from pathlib import Path
import numpy as np
from datetime import datetime
import logging

# Import the existing Gemini model
from gemini_torch.model import Gemini
from gemini_torch.tokenizer import MultimodalSentencePieceTokenizer
from gemini_torch.long_gemini import LongGemini

from .core.exceptions import ModelError, GeminiCLIError


class GeminiModelInterface:
    """Interface for interacting with the Gemini PyTorch model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.logger = logging.getLogger(__name__)

        # Model components
        self.model = None
        self.tokenizer = None
        self.device = None

        # Model parameters
        self.model_name = config.get("model", "gemini-torch")
        self.max_tokens = config.get("maxTokens", 4096)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("topP", 0.9)
        self.top_k = config.get("topK", 40)

        # Model dimensions (will be set based on config)
        self.model_params = {
            "num_tokens": 50432,
            "max_seq_len": self.max_tokens,
            "dim": 2560,
            "depth": 32,
            "dim_head": 128,
            "heads": 24,
            "use_abs_pos_emb": False,
            "attn_flash": True,
            "attn_kv_heads": 2,
            "qk_norm": True,
            "attn_qk_norm": True,
            "attn_qk_norm_dim_scale": True
        }

        # Override with config values
        model_config = config.get("modelConfig", {})
        self.model_params.update(model_config)

        # Tool calling support
        self.tool_calling_enabled = True
        self.available_tools = []

        # Conversation management
        self.conversation_history = []
        self.system_prompt = self._get_system_prompt()

    async def initialize(self):
        """Initialize the model and tokenizer."""
        self.logger.info("Initializing Gemini model...")

        try:
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")

            # Initialize tokenizer
            await self._initialize_tokenizer()

            # Initialize model
            await self._initialize_model()

            self.logger.info("Model initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise ModelError(f"Model initialization failed: {e}")

    async def _initialize_tokenizer(self):
        """Initialize the tokenizer."""
        try:
            tokenizer_name = self.config.get("tokenizerName", "hf-internal-testing/llama-tokenizer")
            self.tokenizer = MultimodalSentencePieceTokenizer(tokenizer_name=tokenizer_name)
            self.logger.info(f"Tokenizer initialized: {tokenizer_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            raise ModelError(f"Tokenizer initialization failed: {e}")

    async def _initialize_model(self):
        """Initialize the Gemini model."""
        try:
            # Choose model type based on configuration
            model_type = self.config.get("modelType", "standard")

            if model_type == "long":
                self.model = LongGemini(**self.model_params)
            else:
                self.model = Gemini(**self.model_params)

            self.model.to(self.device)
            self.model.eval()

            # Load pretrained weights if available
            model_path = self.config.get("modelPath")
            if model_path and Path(model_path).exists():
                await self._load_model_weights(model_path)

            self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise ModelError(f"Model initialization failed: {e}")

    async def _load_model_weights(self, model_path: str):
        """Load pretrained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.logger.info(f"Loaded model weights from: {model_path}")

        except Exception as e:
            self.logger.warning(f"Could not load model weights: {e}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the model."""
        return """You are Gemini, an advanced AI assistant created by Google DeepMind. You are now running in an open-source implementation with access to various tools and capabilities.

You can help with:
- Code analysis and generation
- File operations (reading, writing, editing)
- Shell command execution
- Web searches and content fetching
- Memory management across conversations
- Multi-modal understanding (text, images, audio, video)

When you need to use tools, request them by generating a tool call in the following JSON format:
{
  "tool_calls": [
    {
      "name": "tool_name",
      "arguments": {
        "param1": "value1",
        "param2": "value2"
      }
    }
  ]
}

Always be helpful, accurate, and safe. Ask for clarification when needed and explain your reasoning when using tools.
"""

    async def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response from the model."""
        try:
            # Update available tools
            if available_tools:
                self.available_tools = available_tools

            # Prepare the input
            messages = self._prepare_messages(prompt, conversation_history or [])

            # Tokenize input
            input_text = self._format_messages_for_model(messages)
            input_tokens = self._tokenize_input(input_text)

            # Generate response
            with torch.no_grad():
                output_tokens = await self._generate_tokens(input_tokens)

            # Decode response
            response_text = self._decode_output(output_tokens)

            # Parse tool calls if any
            tool_calls, cleaned_response = self._extract_tool_calls(response_text)

            # Calculate token usage
            usage = {
                "input_tokens": len(input_tokens[0]) if input_tokens.dim() > 1 else len(input_tokens),
                "output_tokens": len(output_tokens[0]) if output_tokens.dim() > 1 else len(output_tokens),
                "total_tokens": 0
            }
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

            result = {
                "content": cleaned_response,
                "usage": usage,
                "model": self.model_name,
                "timestamp": datetime.now().isoformat()
            }

            if tool_calls:
                result["tool_calls"] = tool_calls

            return result

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise ModelError(f"Response generation failed: {e}")

    def _prepare_messages(self, prompt: str, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Prepare messages for the model."""
        messages = []

        # Add system prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })

        # Add tool information if available
        if self.available_tools:
            tool_info = self._format_tool_information()
            messages.append({
                "role": "system",
                "content": f"Available tools:\n{tool_info}"
            })

        # Add conversation history
        for message in conversation_history[-10:]:  # Keep last 10 messages
            if message.get("role") in ["user", "assistant"]:
                messages.append({
                    "role": message["role"],
                    "content": message.get("content", "")
                })

        # Add current prompt
        messages.append({
            "role": "user",
            "content": prompt
        })

        return messages

    def _format_tool_information(self) -> str:
        """Format tool information for the model."""
        if not self.available_tools:
            return "No tools available."

        tool_descriptions = []
        for tool in self.available_tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description")
            parameters = tool.get("parameters", {})

            tool_desc = f"- {name}: {description}"
            if parameters.get("properties"):
                props = []
                for prop_name, prop_info in parameters["properties"].items():
                    prop_type = prop_info.get("type", "any")
                    required = prop_name in parameters.get("required", [])
                    req_marker = " (required)" if required else ""
                    props.append(f"  - {prop_name} ({prop_type}){req_marker}: {prop_info.get('description', '')}")

                if props:
                    tool_desc += "\n" + "\n".join(props)

            tool_descriptions.append(tool_desc)

        return "\n".join(tool_descriptions)

    def _format_messages_for_model(self, messages: List[Dict[str, str]]) -> str:
        """Format messages as a single string for the model."""
        formatted_parts = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")

        # Add final assistant prompt
        formatted_parts.append("Assistant:")

        return "\n\n".join(formatted_parts)

    def _tokenize_input(self, text: str) -> torch.Tensor:
        """Tokenize input text."""
        try:
            # Use the tokenizer to encode the text
            tokens = self.tokenizer.encode(text, modality="text")

            # Convert to tensor and add batch dimension
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

            # Truncate if too long
            max_len = self.model_params["max_seq_len"] - 512  # Leave room for generation
            if token_tensor.shape[1] > max_len:
                token_tensor = token_tensor[:, -max_len:]

            return token_tensor

        except Exception as e:
            self.logger.error(f"Tokenization failed: {e}")
            raise ModelError(f"Tokenization failed: {e}")

    async def _generate_tokens(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """Generate tokens using the model."""
        try:
            batch_size, seq_len = input_tokens.shape
            max_new_tokens = min(512, self.max_tokens - seq_len)

            generated_tokens = input_tokens.clone()

            for _ in range(max_new_tokens):
                # Forward pass
                with torch.no_grad():
                    if hasattr(self.model, 'generate'):
                        # If model has a generate method
                        outputs = self.model.generate(
                            generated_tokens,
                            max_length=generated_tokens.shape[1] + 1,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            do_sample=True
                        )
                        generated_tokens = outputs
                    else:
                        # Manual generation
                        logits = self.model(generated_tokens)

                        # Apply temperature
                        if self.temperature != 1.0:
                            logits = logits / self.temperature

                        # Apply top-k filtering
                        if self.top_k > 0:
                            indices_to_remove = logits < torch.topk(logits, self.top_k, dim=-1)[0][..., -1, None]
                            logits[indices_to_remove] = -float('inf')

                        # Apply top-p filtering
                        if self.top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                            sorted_indices_to_remove = cumulative_probs > self.top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0

                            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                            logits[indices_to_remove] = -float('inf')

                        # Sample next token
                        probs = F.softmax(logits[:, -1, :], dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                        # Append to sequence
                        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

                # Check for end of sequence or special tokens
                if next_token.item() in [self.tokenizer.eos_id, self.tokenizer.pad_id]:
                    break

                # Yield control to allow other operations
                if (_ + 1) % 10 == 0:
                    await asyncio.sleep(0)

            return generated_tokens

        except Exception as e:
            self.logger.error(f"Token generation failed: {e}")
            raise ModelError(f"Token generation failed: {e}")

    def _decode_output(self, output_tokens: torch.Tensor) -> str:
        """Decode output tokens to text."""
        try:
            # Remove batch dimension and convert to list
            if output_tokens.dim() > 1:
                tokens = output_tokens[0].tolist()
            else:
                tokens = output_tokens.tolist()

            # Decode using tokenizer
            decoded_text = self.tokenizer.decode(tokens)

            # Clean up the response
            decoded_text = self._clean_generated_text(decoded_text)

            return decoded_text

        except Exception as e:
            self.logger.error(f"Decoding failed: {e}")
            raise ModelError(f"Decoding failed: {e}")

    def _clean_generated_text(self, text: str) -> str:
        """Clean up generated text."""
        # Remove special tokens and artifacts
        text = text.replace("<|endoftext|>", "")
        text = text.replace("<|im_end|>", "")
        text = text.replace("<|im_start|>", "")

        # Extract only the assistant's response
        if "Assistant:" in text:
            parts = text.split("Assistant:")
            if len(parts) > 1:
                text = parts[-1].strip()

        # Remove any remaining system/human prompts
        lines = text.split('\n')
        cleaned_lines = []
        skip_line = False

        for line in lines:
            if line.strip().startswith(("System:", "Human:", "Assistant:")):
                skip_line = True
                continue
            if skip_line and line.strip() == "":
                continue
            skip_line = False
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()

    def _extract_tool_calls(self, text: str) -> Tuple[List[Dict[str, Any]], str]:
        """Extract tool calls from generated text."""
        tool_calls = []
        cleaned_text = text

        try:
            # Look for JSON-formatted tool calls
            json_pattern = r'\{[^{}]*"tool_calls"[^{}]*\}'
            matches = re.finditer(json_pattern, text, re.DOTALL)

            for match in matches:
                try:
                    json_str = match.group(0)
                    data = json.loads(json_str)

                    if "tool_calls" in data:
                        tool_calls.extend(data["tool_calls"])
                        # Remove the JSON from the text
                        cleaned_text = cleaned_text.replace(json_str, "").strip()

                except json.JSONDecodeError:
                    continue

            # Also look for simpler tool call patterns
            simple_pattern = r'(?:use|call|execute)\s+(?:tool\s+)?(\w+)\s*(?:with|:)?\s*\{([^}]+)\}'
            simple_matches = re.finditer(simple_pattern, text, re.IGNORECASE)

            for match in simple_matches:
                tool_name = match.group(1)
                args_str = match.group(2)

                try:
                    # Parse arguments
                    args = {}
                    for arg_pair in args_str.split(','):
                        if ':' in arg_pair:
                            key, value = arg_pair.split(':', 1)
                            key = key.strip().strip('"\'')
                            value = value.strip().strip('"\'')
                            args[key] = value

                    tool_calls.append({
                        "name": tool_name,
                        "arguments": args
                    })

                    # Remove from text
                    cleaned_text = cleaned_text.replace(match.group(0), "").strip()

                except Exception:
                    continue

        except Exception as e:
            self.logger.warning(f"Error extracting tool calls: {e}")

        return tool_calls, cleaned_text

    async def compress_conversation(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress conversation history to save tokens."""
        try:
            if len(conversation) <= 2:
                return conversation

            # Keep first and last messages, summarize the middle
            first_msg = conversation[0]
            last_msg = conversation[-1]
            middle_msgs = conversation[1:-1]

            if not middle_msgs:
                return conversation

            # Create summary prompt
            summary_prompt = "Summarize the following conversation history concisely:\n\n"
            for msg in middle_msgs:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                summary_prompt += f"{role.title()}: {content}\n\n"

            # Generate summary
            summary_response = await self.generate_response(
                summary_prompt,
                conversation_history=[],  # No history for summary
                available_tools=[]  # No tools for summary
            )

            summary_msg = {
                "role": "system",
                "content": f"Previous conversation summary: {summary_response.get('content', '')}",
                "timestamp": datetime.now().isoformat()
            }

            return [first_msg, summary_msg, last_msg]

        except Exception as e:
            self.logger.warning(f"Failed to compress conversation: {e}")
            return conversation  # Return original if compression fails

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "name": self.model_name,
            "type": self.config.get("modelType", "standard"),
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "device": str(self.device) if self.device else "unknown",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k
        }

    def get_version(self) -> str:
        """Get model version."""
        return "1.0.0"  # Version of this model interface

    async def health_check(self) -> bool:
        """Check if the model is healthy and responsive."""
        try:
            if not self.model or not self.tokenizer:
                return False

            # Simple test generation
            test_tokens = torch.tensor([[1, 2, 3]], device=self.device)
            with torch.no_grad():
                _ = self.model(test_tokens)

            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        try:
            if self.tokenizer:
                tokens = self.tokenizer.encode(text, modality="text")
                return len(tokens)
            else:
                # Rough estimation: ~4 characters per token
                return len(text) // 4
        except Exception:
            return len(text) // 4

    def can_handle_modality(self, modality: str) -> bool:
        """Check if the model can handle a specific modality."""
        supported_modalities = ["text", "image", "audio", "video"]
        return modality.lower() in supported_modalities

    async def process_multimodal_input(
        self,
        text: Optional[str] = None,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process multimodal input through the model."""
        try:
            # Convert text to tokens if provided
            text_tokens = None
            if text:
                text_tokens = self._tokenize_input(text)

            # Forward through model with multimodal inputs
            with torch.no_grad():
                if hasattr(self.model, 'forward') and text_tokens is not None:
                    output = self.model(
                        text=text_tokens,
                        img=image,
                        audio=audio,
                        video=video
                    )
                else:
                    # Fallback to text-only
                    output = self.model(text_tokens)

            return output

        except Exception as e:
            self.logger.error(f"Multimodal processing failed: {e}")
            raise ModelError(f"Multimodal processing failed: {e}")