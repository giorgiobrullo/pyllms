from typing import AsyncGenerator, Dict, Generator, List, Optional, Union, Any
import tiktoken
import os

from openai import AsyncOpenAI, OpenAI
import json

from ..results.result import AsyncStreamResult, Result, StreamResult
from .base_provider import BaseProvider
from ..utils.image_utils import (
    encode_image_to_base64,
    normalize_images_input,
)
try:
    from ..utils.pydantic_utils import (
        pydantic_to_json_schema,
        is_pydantic_model,
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class OpenAIProvider(BaseProvider):
    # cost is per million tokens
    MODEL_INFO = {
        "gpt-3.5-turbo": {"prompt": 2.0, "completion": 2.0, "token_limit": 16_385, "is_chat": True, "output_limit": 4_096},
        "gpt-3.5-turbo-1106": {"prompt": 2.0, "completion": 2.0, "token_limit": 16_385, "is_chat": True, "output_limit": 4_096},
        "gpt-3.5-turbo-instruct": {"prompt": 2.0, "completion": 2.0, "token_limit": 4096, "is_chat": False},
        "gpt-4": {"prompt": 30.0, "completion": 60.0, "token_limit": 8192, "is_chat": True},
        "gpt-4-1106-preview": {"prompt": 10.0, "completion": 30.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "gpt-4-turbo-preview": {"prompt": 10.0, "completion": 30.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "gpt-4o": {"prompt": 2.5, "completion": 10.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "gpt-4o-2024-08-06": {"prompt": 2.50, "completion": 10.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "gpt-4.1": {"prompt": 10.0, "completion": 30.0, "token_limit": 128000, "is_chat": True, "output_limit": 16384},
        "gpt-4.1-mini": {"prompt": 2.0, "completion": 8.0, "token_limit": 128000, "is_chat": True, "output_limit": 16384},
        "gpt-4.1-nano": {"prompt": 0.5, "completion": 2.0, "token_limit": 128000, "is_chat": True, "output_limit": 16384},
        "gpt-4.5-preview": {"prompt": 75, "completion": 150.0, "token_limit": 128000, "is_chat": True, "output_limit": 16384},
        "chatgpt-4o-latest": {"prompt": 5, "completion": 15.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096},
        "o1-preview": {"prompt": 15.0, "completion": 60.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096, "use_max_completion_tokens": True},
        "o1-mini": {"prompt": 3.0, "completion": 12.0, "token_limit": 128000, "is_chat": True, "output_limit": 4_096, "use_max_completion_tokens": True},
        "o1": {"prompt": 15.0, "completion": 60.0, "token_limit": 200000, "is_chat": True, "output_limit": 100000, "use_max_completion_tokens": True},
        "o1-pro": {"prompt": 150.0, "completion": 600.0, "token_limit": 200000, "is_chat": True, "output_limit": 100000, "use_max_completion_tokens": True, "use_responses_api": True},
        "o3-mini": {"prompt": 1.1, "completion": 4.40, "token_limit": 128000, "is_chat": True, "output_limit": 4_096, "use_max_completion_tokens": True},
        "o3": {"prompt": 20.0, "completion": 80.0, "token_limit": 200000, "is_chat": True, "output_limit": 100000, "use_max_completion_tokens": True},
        "o3-pro": {"prompt": 200.0, "completion": 800.0, "token_limit": 200000, "is_chat": True, "output_limit": 100000, "use_max_completion_tokens": True, "use_responses_api": True},
        "o4-mini": {"prompt": 0.8, "completion": 3.2, "token_limit": 128000, "is_chat": True, "output_limit": 4_096, "use_max_completion_tokens": True},
        "gpt-5": {"prompt": 1.25, "completion": 10.0, "token_limit": 256000, "is_chat": True, "output_limit": 128000, "use_max_completion_tokens": True, "use_responses_api": True},
        "gpt-5.1": {"prompt": 1.25, "completion": 10.0, "token_limit": 272000, "is_chat": True, "output_limit": 128000, "use_max_completion_tokens": True, "use_responses_api": True},
        "gpt-5.2": {"prompt": 1.75, "completion": 14.0, "token_limit": 400000, "is_chat": True, "output_limit": 128000, "use_max_completion_tokens": True, "use_responses_api": True},
        "gpt-5-mini": {"prompt": 0.25, "completion": 2.0, "token_limit": 256000, "is_chat": True, "output_limit": 128000, "use_max_completion_tokens": True, "use_responses_api": True},
        "gpt-5-nano": {"prompt": 0.05, "completion": 0.40, "token_limit": 256000, "is_chat": True, "output_limit": 128000, "use_max_completion_tokens": True, "use_responses_api": True},
    }

    def __init__(
        self,
        api_key: Union[str, None] = None,
        model: Union[str, None] = None,
        client_kwargs: Union[dict, None] = None,
        async_client_kwargs: Union[dict, None] = None,
    ):
        if model is None:
            model = list(self.MODEL_INFO.keys())[0]
        self.model = model
        if client_kwargs is None:
            client_kwargs = {}
        self.client = OpenAI(api_key=api_key, **client_kwargs)
        if async_client_kwargs is None:
            async_client_kwargs = {}
        self.async_client = AsyncOpenAI(api_key=api_key, **async_client_kwargs)

    @property
    def is_chat_model(self) -> bool:
        return self.MODEL_INFO[self.model]['is_chat']
        
    @property
    def uses_responses_api(self) -> bool:
        return self.MODEL_INFO[self.model].get('use_responses_api', False)

    def _format_content_with_images(self, prompt: Union[str, dict, list], images: Optional[Union[str, list]] = None, for_responses_api: bool = False) -> Union[str, list]:
        """Format content with images for OpenAI vision API."""
        # Already formatted content - pass through
        if isinstance(prompt, (dict, list)):
            return prompt
            
        # No images - return plain text
        images_list = normalize_images_input(images)
        if not images_list:
            return prompt
            
        # Different formats for Responses API (GPT-5) vs Chat API
        if for_responses_api:
            # GPT-5 Responses API format: input_text + input_image
            content = [{"type": "input_text", "text": prompt}]
            
            for image in images_list:
                if image.startswith(('http://', 'https://')):
                    # Responses API wants just the URL string
                    image_obj = {"type": "input_image", "image_url": image}
                elif os.path.exists(image):
                    # Local file - encode to base64 data URL
                    base64_data, media_type = encode_image_to_base64(image)
                    url = f"data:{media_type};base64,{base64_data}"
                    image_obj = {"type": "input_image", "image_url": url}
                elif image.startswith('data:image/'):
                    # Already a data URL
                    image_obj = {"type": "input_image", "image_url": image}
                else:
                    # Raw base64 - wrap in data URL
                    url = f"data:image/jpeg;base64,{image}"
                    image_obj = {"type": "input_image", "image_url": url}
                
                content.append(image_obj)
        else:
            # Standard Chat API format: text + image_url objects
            content = [{"type": "text", "text": prompt}]
            
            for image in images_list:
                if image.startswith(('http://', 'https://')):
                    # Chat API accepts URLs in object format
                    image_obj = {"type": "image_url", "image_url": {"url": image}}
                elif os.path.exists(image):
                    # Local file - encode to base64 data URL
                    base64_data, media_type = encode_image_to_base64(image)
                    url = f"data:{media_type};base64,{base64_data}"
                    image_obj = {"type": "image_url", "image_url": {"url": url}}
                elif image.startswith('data:image/'):
                    # Already a data URL
                    image_obj = {"type": "image_url", "image_url": {"url": image}}
                else:
                    # Raw base64 - wrap in data URL
                    url = f"data:image/jpeg;base64,{image}"
                    image_obj = {"type": "image_url", "image_url": {"url": url}}
                
                content.append(image_obj)
                    
        return content

    def _process_response_format(self, response_format: Optional[Union[Dict, Any]]) -> Optional[Dict]:
        """Process response_format, converting Pydantic models if necessary."""
        if response_format is None:
            return None

        # If it's a Pydantic model, convert it to JSON schema
        if PYDANTIC_AVAILABLE and is_pydantic_model(response_format):
            return pydantic_to_json_schema(response_format)

        # Otherwise, assume it's already a dict in the correct format
        return response_format

    def count_tokens(self, content: Union[str, List[dict]]) -> int:
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # For new models not yet in tiktoken, use gpt-4 as fallback
            enc = tiktoken.encoding_for_model("gpt-4")
        
        if isinstance(content, list):
            # When field name is present, ChatGPT will ignore the role token.
            # Adopted from OpenAI cookbook
            # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            formatting_token_count = 4

            messages = content
            messages_text = ["".join(message.values()) for message in messages]
            tokens = [enc.encode(t, disallowed_special=()) for t in messages_text]

            n_tokens_list = []
            for token, message in zip(tokens, messages):
                n_tokens = len(token) + formatting_token_count
                if "name" in message:
                    n_tokens += -1
                n_tokens_list.append(n_tokens)
            return sum(n_tokens_list)
        else:
            return len(enc.encode(content, disallowed_special=()))

    def _prepare_model_inputs(
        self,
        prompt: Union[str, dict, list],
        history: Optional[List[dict]] = None,
        system_message: Union[str, List[dict], None] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stream: bool = False,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        images: Optional[Union[str, list]] = None,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        if self.is_chat_model:
            # Format content with images if provided
            # Use different format for Responses API (GPT-5, etc.)
            content = self._format_content_with_images(prompt, images, for_responses_api=self.uses_responses_api)
            messages = [{"role": "user", "content": content}]

            if history:
                messages = [*history, *messages]

            if isinstance(system_message, str):
                messages = [{"role": "system", "content": system_message}, *messages]

            # users can input multiple full system message in dict form
            elif isinstance(system_message, list):
                messages = [*system_message, *messages]

            model_inputs = {
                "messages": messages,
                "stream": stream,
                **({'reasoning_effort': reasoning_effort} if reasoning_effort else {}),
                **({'verbosity': verbosity} if verbosity else {}),
                **({'response_format': self._process_response_format(response_format)} if response_format else {}),
                **kwargs,
            }

            # Use max_completion_tokens for models that require it
            if self.MODEL_INFO[self.model].get("use_max_completion_tokens", False):
                model_inputs["max_completion_tokens"] = max_tokens
            else:
                model_inputs["max_tokens"] = max_tokens
                model_inputs["temperature"] = temperature

        else:
            if history:
                raise ValueError(
                    f"history argument is not supported for {self.model} model"
                )

            if system_message:
                raise ValueError(
                    f"system_message argument is not supported for {self.model} model"
                )

            model_inputs = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                **kwargs,
            }
        return model_inputs

    def complete(
        self,
        prompt: Union[str, dict, list],
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        images: Optional[Union[str, list]] = None,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> Result:
        """
        Args:
            prompt: Text prompt, or formatted content dict/list for vision models
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
            images: Optional image(s) to include - can be file paths, URLs, or base64 strings
        """

        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            images=images,
            response_format=response_format,
            **kwargs,
        )

        with self.track_latency():
            if self.uses_responses_api:
                # Convert messages format for Responses API
                input_messages = model_inputs.pop("messages")
                # Handle any reasoning_effort parameter
                reasoning = {}
                if "reasoning_effort" in model_inputs:
                    reasoning["effort"] = model_inputs.pop("reasoning_effort")
                
                # Initialize text parameters dict for Responses API
                text_params = {}
                if "verbosity" in model_inputs:
                    text_params["verbosity"] = model_inputs.pop("verbosity")
                
                # Prepare parameters for Responses API
                responses_params = {
                    "model": self.model,
                    "input": input_messages
                }
                
                # Temperature is not supported for some models with Responses API
                # Only add it if the model supports it
                
                # For Responses API, max_tokens should be converted to max_output_tokens
                if max_tokens is not None:
                    responses_params["max_output_tokens"] = max_tokens
                
                # Add response_format as text.format for Responses API
                if "response_format" in model_inputs and model_inputs["response_format"]:
                    # For Responses API, structured output uses text.format
                    text_params["format"] = model_inputs["response_format"]
                
                # Add any other supported parameters
                for key, value in model_inputs.items():
                    if key not in ["messages", "max_completion_tokens", "max_tokens", "temperature", "reasoning_effort", "verbosity", "response_format"]:
                        responses_params[key] = value
                
                # Add reasoning if present
                if reasoning:
                    responses_params["reasoning"] = reasoning
                
                # Add text parameters if present (includes format and/or verbosity)
                if text_params:
                    responses_params["text"] = text_params
                
                response = self.client.responses.create(**responses_params)
            elif self.is_chat_model:
                response = self.client.chat.completions.create(model=self.model, **model_inputs)
            else:
                response = self.client.completions.create(model=self.model, **model_inputs)

        function_call = {}
        completion = ""
        
        if self.uses_responses_api:
            # Extract text from Responses API
            # Find the output_text in the response
            for item in response.output:
                if item.type == "message" and hasattr(item, "content"):
                    for content_item in item.content:
                        if content_item.type == "output_text":
                            completion = content_item.text.strip()
                            break
            
            # Handle function calls if present
            if hasattr(response, 'output') and hasattr(response.output, 'function_calls'):
                function_call = {
                    "name": response.output.function_calls[0].name,
                    "arguments": response.output.function_calls[0].arguments
                }
            
            # Usage has different field names in Responses API
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens
            }
        else:
            is_func_call = response.choices[0].finish_reason == "function_call"
            if self.is_chat_model:
                if is_func_call:
                    function_call = {
                        "name": response.choices[0].message.function_call.name,
                        "arguments": json.loads(response.choices[0].message.function_call.arguments)
                    }
                else:
                    completion = response.choices[0].message.content.strip()
            else:
                completion = response.choices[0].text.strip()
            usage = response.usage

        meta = {
            "tokens_prompt": usage["prompt_tokens"] if isinstance(usage, dict) else usage.prompt_tokens,
            "tokens_completion": usage["completion_tokens"] if isinstance(usage, dict) else usage.completion_tokens,
            "latency": self.latency,
        }
        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            meta=meta,
            function_call=function_call,
        )

    async def acomplete(
        self,
        prompt: Union[str, dict, list],
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        images: Optional[Union[str, list]] = None,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> Result:
        """
        Args:
            prompt: Text prompt, or formatted content dict/list for vision models
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
            images: Optional image(s) to include - can be file paths, URLs, or base64 strings
        """
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            images=images,
            response_format=response_format,
            **kwargs,
        )

        with self.track_latency():
            if self.uses_responses_api:
                # Convert messages format for Responses API
                input_messages = model_inputs.pop("messages")
                # Handle any reasoning_effort parameter
                reasoning = {}
                if "reasoning_effort" in model_inputs:
                    reasoning["effort"] = model_inputs.pop("reasoning_effort")
                
                # Initialize text parameters dict for Responses API
                text_params = {}
                if "verbosity" in model_inputs:
                    text_params["verbosity"] = model_inputs.pop("verbosity")
                
                # Prepare parameters for Responses API
                responses_params = {
                    "model": self.model,
                    "input": input_messages
                }
                
                # Temperature is not supported for some models with Responses API
                # Only add it if the model supports it
                
                # For Responses API, max_tokens should be converted to max_output_tokens
                if max_tokens is not None:
                    responses_params["max_output_tokens"] = max_tokens
                
                # Add response_format as text.format for Responses API
                if "response_format" in model_inputs and model_inputs["response_format"]:
                    # For Responses API, structured output uses text.format
                    text_params["format"] = model_inputs["response_format"]
                
                # Add any other supported parameters
                for key, value in model_inputs.items():
                    if key not in ["messages", "max_completion_tokens", "max_tokens", "temperature", "reasoning_effort", "verbosity", "response_format"]:
                        responses_params[key] = value
                
                # Add reasoning if present
                if reasoning:
                    responses_params["reasoning"] = reasoning
                
                # Add text parameters if present (includes format and/or verbosity)
                if text_params:
                    responses_params["text"] = text_params
                
                response = await self.async_client.responses.create(**responses_params)
                # Find the output_text in the response
                completion = ""
                for item in response.output:
                    if item.type == "message" and hasattr(item, "content"):
                        for content_item in item.content:
                            if content_item.type == "output_text":
                                completion = content_item.text.strip()
                                break
                
                # Usage has different field names in Responses API
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            elif self.is_chat_model:
                response = await self.async_client.chat.completions.create(model=self.model, **model_inputs)
                completion = response.choices[0].message.content.strip()
                usage = response.usage
            else:
                response = await self.async_client.completions.create(model=self.model, **model_inputs)
                completion = response.choices[0].text.strip()
                usage = response.usage

        # Handle usage consistently
        if isinstance(usage, dict):
            meta = {
                "tokens_prompt": usage["prompt_tokens"],
                "tokens_completion": usage["completion_tokens"],
                "latency": self.latency,
            }
        else:
            meta = {
                "tokens_prompt": usage.prompt_tokens,
                "tokens_completion": usage.completion_tokens,
                "latency": self.latency,
            }
        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            meta=meta,
        )

    def complete_stream(
        self,
        prompt: Union[str, dict, list],
        history: Optional[List[dict]] = None,
        system_message: Union[str, List[dict], None] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        images: Optional[Union[str, list]] = None,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> StreamResult:
        """
        Args:
            prompt: Text prompt, or formatted content dict/list for vision models
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
            images: Optional image(s) to include - can be file paths, URLs, or base64 strings
        """
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            images=images,
            response_format=response_format,
            **kwargs,
        )

        if self.uses_responses_api:
            # Responses API doesn't support streaming in the same way
            # For now, we'll use the chat completions API for streaming
            response = self.client.chat.completions.create(model=self.model, **model_inputs)
        elif self.is_chat_model:
            response = self.client.chat.completions.create(model=self.model, **model_inputs)
        else:
            response = self.client.completions.create(model=self.model, **model_inputs)
        stream = self._process_stream(response)

        return StreamResult(stream=stream, model_inputs=model_inputs, provider=self)

    def _process_stream(self, response: Generator) -> Generator:
        if self.is_chat_model:
            chunk_generator = (
                chunk.choices[0].delta.content for chunk in response
            )
        else:
            chunk_generator = (
                chunk.choices[0].text for chunk in response
            )

        while not (first_text := next(chunk_generator)):
            continue
        yield first_text.lstrip()
        for chunk in chunk_generator:
            if chunk is not None:
                yield chunk

    async def acomplete_stream(
            self,
            prompt: Union[str, dict, list],
            history: Optional[List[dict]] = None,
            system_message: Union[str, List[dict], None] = None,
            temperature: float = 0,
            max_tokens: int = 300,
            reasoning_effort: Optional[str] = None,
            verbosity: Optional[str] = None,
            images: Optional[Union[str, list]] = None,
            response_format: Optional[Dict] = None,
            **kwargs,
    ) -> AsyncStreamResult:
        """
        Args:
            prompt: Text prompt, or formatted content dict/list for vision models
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
            images: Optional image(s) to include - can be file paths, URLs, or base64 strings
        """
        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            images=images,
            response_format=response_format,
            **kwargs,
        )

        with self.track_latency():
            if self.uses_responses_api:
                # Responses API doesn't support streaming in the same way
                # For now, we'll use the chat completions API for streaming
                response = await self.async_client.chat.completions.create(model=self.model, **model_inputs)
            elif self.is_chat_model:
                response = await self.async_client.chat.completions.create(model=self.model, **model_inputs)
            else:
                response = await self.async_client.completions.create(model=self.model, **model_inputs)
        stream = self._aprocess_stream(response)
        return AsyncStreamResult(
            stream=stream, model_inputs=model_inputs, provider=self
        )

    async def _aprocess_stream(self, response: AsyncGenerator) -> AsyncGenerator:
        if self.is_chat_model:
            while True:
                first_completion = (await response.__anext__()).choices[0].delta.content
                if first_completion:
                    yield first_completion.lstrip()
                    break

            async for chunk in response:
                completion = chunk.choices[0].delta.content
                if completion is not None:
                    yield completion
        else:
            while True:
                first_completion = (await response.__anext__()).choices[0].text
                if first_completion:
                    yield first_completion.lstrip()
                    break

            async for chunk in response:
                completion = chunk.choices[0].text
                if completion is not None:
                    yield completion
