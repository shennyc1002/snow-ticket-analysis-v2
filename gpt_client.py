"""Custom GPT Client with OAuth2 authentication for organization wrapper API."""

import requests
import time
from typing import Optional
from dataclasses import dataclass
import config


class GPTClientError(Exception):
    """Raised when GPT API call fails."""
    pass


class AuthenticationError(Exception):
    """Raised when OAuth2 authentication fails."""
    pass


@dataclass
class TokenInfo:
    """Stores JWT token and expiry information."""
    access_token: str
    expires_at: float  # Unix timestamp when token expires
    token_type: str = "Bearer"

    def is_expired(self) -> bool:
        """Check if token is expired (with 60 second buffer)."""
        return time.time() >= (self.expires_at - 60)


# Global token cache - persists across Streamlit reruns
_global_token_cache: Optional[TokenInfo] = None


class GPTWrapperClient:
    """
    GPT Client that authenticates via OAuth2 and calls organization's wrapper API.

    Flow:
    1. Authenticate with client_id and client_secret to get JWT token
    2. Use JWT token to call GPT wrapper API
    3. Automatically refresh token when expired
    """

    def __init__(
        self,
        auth_url: str = None,
        client_id: str = None,
        client_secret: str = None,
        gpt_wrapper_base_url: str = None,
        api_version: str = None
    ):
        self.auth_url = auth_url or config.AUTH_URL
        self.client_id = client_id or config.CLIENT_ID
        self.client_secret = client_secret or config.CLIENT_SECRET
        self.gpt_wrapper_base_url = gpt_wrapper_base_url or config.GPT_WRAPPER_BASE_URL
        self.api_version = api_version or config.GPT_API_VERSION

        self._token: Optional[TokenInfo] = None
        self._session = requests.Session()

        # Validate configuration
        if not all([self.auth_url, self.client_id, self.client_secret, self.gpt_wrapper_base_url]):
            raise GPTClientError(
                "Missing required configuration. Please set AUTH_URL, CLIENT_ID, "
                "CLIENT_SECRET, and GPT_WRAPPER_BASE_URL in your .env file."
            )

    def _build_url(self, model: str) -> str:
        """Build the full API URL with model embedded.

        Format: {base_url}/{model}/chat/completions?api-version={version}
        """
        return f"{self.gpt_wrapper_base_url}/{model}/chat/completions?api-version={self.api_version}"

    def _authenticate(self) -> TokenInfo:
        """
        Authenticate using OAuth2 client_credentials flow.

        Returns:
            TokenInfo with access token and expiry
        """
        try:
            response = self._session.post(
                self.auth_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                timeout=30
            )

            if response.status_code != 200:
                raise AuthenticationError(
                    f"Authentication failed with status {response.status_code}: {response.text}"
                )

            data = response.json()

            # Calculate expiry time
            expires_in = data.get("expires_in", 3600)  # Default 1 hour
            expires_at = time.time() + expires_in

            return TokenInfo(
                access_token=data["access_token"],
                expires_at=expires_at,
                token_type=data.get("token_type", "Bearer")
            )

        except requests.RequestException as e:
            raise AuthenticationError(f"Authentication request failed: {e}")

    def _get_token(self) -> str:
        """Get valid access token, refreshing if necessary."""
        global _global_token_cache

        # Check global cache first
        if _global_token_cache is not None and not _global_token_cache.is_expired():
            self._token = _global_token_cache
            return self._token.access_token

        # Need to authenticate
        if self._token is None or self._token.is_expired():
            self._token = self._authenticate()
            _global_token_cache = self._token  # Cache globally

        return self._token.access_token

    def chat_completion(
        self,
        messages: list[dict],
        model: str = None,
        max_tokens: int = None,
        temperature: float = 0.7,
        response_format: dict = None
    ) -> dict:
        """
        Call the GPT wrapper API for chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (defaults to config.MODEL_NAME) - embedded in URL
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            response_format: Optional format specification (e.g., {"type": "json_object"})

        Returns:
            API response as dictionary
        """
        token = self._get_token()
        model_name = model or config.MODEL_NAME

        # Build URL with model embedded
        api_url = self._build_url(model_name)

        # Build payload - messages is already in format [{"role": "user", "content": "..."}]
        payload = {
            "messages": messages
        }

        # Optional parameters - comment out if your wrapper doesn't support them
        payload["max_tokens"] = max_tokens or config.MAX_TOKENS
        payload["temperature"] = temperature
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Convert payload to proper JSON string
        import json as json_module
        json_body = json_module.dumps(payload, ensure_ascii=False)

        # Debug: Print the actual JSON being sent (flush=True ensures immediate output)
        print(f"[DEBUG] Request URL: {api_url}", flush=True)
        print(f"[DEBUG] Request Body:\n{json_body}", flush=True)

        try:
            print(f"[DEBUG] Sending request...", flush=True)
            response = self._session.post(
                api_url,
                data=json_body,  # Send as string, not dict
                headers=headers,
                timeout=120  # Longer timeout for GPT responses
            )
            print(f"[DEBUG] Request completed.", flush=True)

            if response.status_code == 401:
                # Token might be invalid, clear cache and retry
                global _global_token_cache
                self._token = None
                _global_token_cache = None
                token = self._get_token()
                headers["Authorization"] = f"Bearer {token}"
                response = self._session.post(
                    api_url,
                    data=json_body,  # Send as string, not dict
                    headers=headers,
                    timeout=120
                )

            # Debug: Print response status (flush=True ensures immediate output)
            print(f"[DEBUG] Response Status: {response.status_code}", flush=True)
            print(f"[DEBUG] Response Body: {response.text[:500]}...", flush=True)  # First 500 chars

            if response.status_code != 200:
                raise GPTClientError(
                    f"GPT API call failed with status {response.status_code}: {response.text}"
                )

            return response.json()

        except requests.RequestException as e:
            print(f"[ERROR] Request exception: {e}", flush=True)
            raise GPTClientError(f"GPT API request failed: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected exception: {type(e).__name__}: {e}", flush=True)
            raise

    def get_completion_text(
        self,
        messages: list[dict],
        model: str = None,
        max_tokens: int = None,
        response_format: dict = None
    ) -> str:
        """
        Convenience method to get just the text content from a completion.

        Returns:
            The assistant's message content as a string
        """
        response = self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            response_format=response_format
        )

        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise GPTClientError(f"Unexpected response format: {e}\nResponse: {response}")


class DirectOpenAIClient:
    """Fallback client that uses OpenAI directly (for testing without wrapper)."""

    def __init__(self, api_key: str = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)

    def chat_completion(
        self,
        messages: list[dict],
        model: str = None,
        max_tokens: int = None,
        temperature: float = 0.7,
        response_format: dict = None
    ) -> dict:
        """Call OpenAI API directly."""
        kwargs = {
            "model": model or config.MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens or config.MAX_TOKENS,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)

        # Convert to dict format matching wrapper response
        return {
            "choices": [{
                "message": {
                    "content": response.choices[0].message.content
                }
            }]
        }

    def get_completion_text(
        self,
        messages: list[dict],
        model: str = None,
        max_tokens: int = None,
        response_format: dict = None
    ) -> str:
        """Get just the text content from a completion."""
        response = self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            response_format=response_format
        )
        return response["choices"][0]["message"]["content"]


def get_gpt_client():
    """
    Factory function to get the appropriate GPT client.

    Returns GPTWrapperClient if wrapper credentials are configured,
    otherwise falls back to DirectOpenAIClient.
    """
    if config.USE_WRAPPER_API:
        print("[INFO] Using organization's GPT wrapper API")
        return GPTWrapperClient()
    elif config.OPENAI_API_KEY:
        print("[INFO] Using direct OpenAI API")
        return DirectOpenAIClient()
    else:
        raise GPTClientError(
            "No API credentials configured. Please set either:\n"
            "1. CLIENT_ID, CLIENT_SECRET, AUTH_URL, GPT_WRAPPER_URL (for wrapper API)\n"
            "2. OPENAI_API_KEY (for direct OpenAI access)"
        )
