# llms/providers/anthropic.py

from typing import AsyncGenerator, Dict, Generator, List, Optional, Union
import os

import anthropic

from ..results.result import AsyncStreamResult, Result, StreamResult
from .base_provider import BaseProvider
from ..utils.image_utils import (
    encode_image_to_base64,
    download_and_encode_image,
    normalize_images_input,
    parse_base64_data_url
)


class AnthropicProvider(BaseProvider):
    MODEL_INFO = {
        # Legacy model
        "claude-2.1": {"prompt": 8.00, "completion": 24.00, "token_limit": 200_000, "output_limit": 4_096},
        
        # Claude 3 family
        "claude-3-5-sonnet-20240620": {"prompt": 3.00, "completion": 15, "token_limit": 200_000, "output_limit": 4_096},
        "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15, "token_limit": 200_000, "output_limit": 4_096},
        "claude-3-5-haiku-20241022": {"prompt": 0.80, "completion": 4, "token_limit": 200_000, "output_limit": 4_096},
        
        # Claude 3.7 family
        "claude-3-7-sonnet-20250219": {"prompt": 3.00, "completion": 15, "token_limit": 200_000, "output_limit": 4_096},
        
        # Claude 4 family (latest)
        "claude-sonnet-4-20250514": {"prompt": 3.00, "completion": 15, "token_limit": 200_000, "output_limit": 4_096},
        "claude-opus-4-20250514": {"prompt": 15.00, "completion": 75, "token_limit": 200_000, "output_limit": 4_096},
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
        self.client = anthropic.Anthropic(api_key=api_key, **client_kwargs)
        if async_client_kwargs is None:
            async_client_kwargs = {}
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key, **async_client_kwargs)

    def _format_content_with_images(self, prompt: Union[str, dict, list], images: Optional[Union[str, list]] = None) -> Union[str, list]:
        """Format content with images for Anthropic vision API."""
        # Already formatted content - pass through
        if isinstance(prompt, (dict, list)):
            return prompt
            
        # No images - return plain text
        images_list = normalize_images_input(images)
        if not images_list:
            return prompt
            
        # Build Anthropic vision format: image objects then text
        # Anthropic prefers images before text for better performance
        content = []
        
        for image in images_list:
            if image.startswith(('http://', 'https://')):
                # Anthropic requires downloading URLs and encoding
                base64_data, media_type = download_and_encode_image(image)
            elif os.path.exists(image):
                # Local file - encode to base64
                base64_data, media_type = encode_image_to_base64(image)
            elif image.startswith('data:image/'):
                # Parse data URL
                base64_data, media_type = parse_base64_data_url(image)
            else:
                # Raw base64 string
                base64_data = image
                media_type = "image/jpeg"
            
            # Anthropic image format
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data
                }
            })
        
        # Add text after images
        content.append({"type": "text", "text": prompt})
                    
        return content

    def count_tokens(self, content: str | Dict) -> int:
        """Count tokens using Anthropic's native token counting API."""
        
        if isinstance(content, str):
            # For string content, format as a single user message
            messages = [{"role": "user", "content": content}]
        elif isinstance(content, list):
            # If it's already a list of messages, use directly
            messages = content
        elif isinstance(content, dict):
            # If it's a single message dict, wrap in list
            messages = [content]
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
        
        try:
            response = self.client.messages.count_tokens(
                model=self.model,
                messages=messages
            )
            return response.input_tokens
        except Exception as e:
            # Fallback to tiktoken approximation if API fails
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
            if isinstance(content, str):
                return len(enc.encode(content, disallowed_special=()))
            
            # Handle message format
            formatting_token_count = 4
            total = 0
            for message in messages:
                if isinstance(message.get("content"), str):
                    total += len(enc.encode(message["content"], disallowed_special=())) + formatting_token_count
            return total



    def _prepare_message_inputs(
        self,
        prompt: Union[str, dict, list],
        history: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        system_message: Union[str, None] = None,
        images: Optional[Union[str, list]] = None,
        **kwargs,
    ) -> Dict:
        history = history or []
        system_message = system_message or ""
        max_tokens = kwargs.pop("max_tokens_to_sample", max_tokens)
        
        # Format content with images if provided
        content = self._format_content_with_images(prompt, images)
        messages = [*history, {"role": "user", "content": content}]
        if ai_prompt:
            messages.append({"role": "assistant", "content": ai_prompt})

        if system_message and self.model.startswith("claude-instant"):
            raise ValueError("System message is not supported for Claude instant")
        model_inputs = {
            "messages": messages,
            "system": system_message,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop_sequences": stop_sequences,
        }
        
        # Add thinking mode if specified
        thinking = kwargs.pop('thinking', None)
        if thinking is not None:
            model_inputs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking if isinstance(thinking, int) else 32000
            }
        return model_inputs

    def _prepare_model_inputs(
        self,
        prompt: Union[str, dict, list],
        history: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        system_message: Union[str, None] = None,
        stream: bool = False,
        images: Optional[Union[str, list]] = None,
        **kwargs,
    ) -> Dict:
        return self._prepare_message_inputs(
            prompt=prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            ai_prompt=ai_prompt,
            system_message=system_message,
            images=images,
            stream=stream,
            **kwargs,
        )

    def complete(
        self,
        prompt: Union[str, dict, list],
        history: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        system_message: Union[str, None] = None,
        images: Optional[Union[str, list]] = None,
        **kwargs,
    ) -> Result:
        """
        Args:
            prompt: Text prompt, or formatted content dict/list for vision models
            history: messages in OpenAI format,
              each dict must include role and content key.
            ai_prompt: prefix of AI response, for finer control on the output.
            images: Optional image(s) to include - can be file paths, URLs, or base64 strings
        """

        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            ai_prompt=ai_prompt,
            system_message=system_message,
            images=images,
            **kwargs,
        )

        meta = {}
        with self.track_latency():
            response = self.client.messages.create(model=self.model, **model_inputs)
            if "thinking" in model_inputs:
                text_block = next((b for b in response.content if b.type == "text"), None)
                completion = text_block.text if text_block else ""
            else:
                completion = response.content[0].text
            meta["tokens_prompt"] = response.usage.input_tokens
            meta["tokens_completion"] = response.usage.output_tokens

        meta["latency"] = self.latency
        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            meta=meta,
        )

    async def acomplete(
        self,
        prompt: Union[str, dict, list],
        history: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        system_message: Union[str, None] = None,
        images: Optional[Union[str, list]] = None,
        **kwargs,
    ):
        """
        Args:
            prompt: Text prompt, or formatted content dict/list for vision models
            history: messages in OpenAI format,
              each dict must include role and content key.
            ai_prompt: prefix of AI response, for finer control on the output.
            images: Optional image(s) to include - can be file paths, URLs, or base64 strings
        """

        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            ai_prompt=ai_prompt,
            system_message=system_message,
            images=images,
            **kwargs,
        )

        with self.track_latency():
            response = await self.async_client.messages.create(model=self.model, **model_inputs)
            if "thinking" in model_inputs:
                text_block = next((b for b in response.content if b.type == "text"), None)
                completion = text_block.text if text_block else ""
            else:
                completion = response.content[0].text

        return Result(
            text=completion,
            model_inputs=model_inputs,
            provider=self,
            meta={"latency": self.latency},
        )

    def complete_stream(
        self,
        prompt: Union[str, dict, list],
        history: Optional[List[dict]] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        stop_sequences: Optional[List[str]] = None,
        ai_prompt: str = "",
        system_message: Union[str, None] = None,
        images: Optional[Union[str, list]] = None,
        **kwargs,
    ) -> StreamResult:
        """
        Args:
            prompt: Text prompt, or formatted content dict/list for vision models
            history: messages in OpenAI format,
              each dict must include role and content key.
            ai_prompt: prefix of AI response, for finer control on the output.
            images: Optional image(s) to include - can be file paths, URLs, or base64 strings
        """

        model_inputs = self._prepare_model_inputs(
            prompt=prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            ai_prompt=ai_prompt,
            system_message=system_message,
            stream=True,
            images=images,
            **kwargs,
        )
        with self.track_latency():
            response = self.client.messages.stream(model=self.model, **model_inputs)
            stream = self._process_message_stream(response)

        return StreamResult(stream=stream, model_inputs=model_inputs, provider=self)

    def _process_message_stream(self, response) -> Generator:
        with response as stream_manager:
            for text in stream_manager.text_stream:
                yield text

    def _process_stream(self, response: Generator) -> Generator:
        first_completion = next(response).completion
        yield first_completion.lstrip()

        for data in response:
            yield data.completion
