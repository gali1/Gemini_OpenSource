import os
from logging import getLogger
from typing import List, Optional
from pathlib import Path

import requests
from sentencepiece import SentencePieceProcessor

logger = getLogger()

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}


class MultimodalSentencePieceTokenizer:
    """Multimodal SentencePiece tokenizer.

    Args:
        model_path (str, optional): Path to the SentencePiece model file. Defaults to None.
        tokenizer_name (str, optional): Name of the tokenizer to download. Defaults to None.

    Methods:
        encode(s: str, modality: str, bos: bool = True, eos: bool = True) -> List[int]: Encodes a string into a list of token IDs.
        decode(tokens: List[int]) -> str: Decodes a list of token IDs into a string.

    Examples:
        >>> tokenizer_name = "hf-internal-testing/llama-tokenizer"
        >>> tokenizer = MultimodalSentencePieceTokenizer(tokenizer_name=tokenizer_name)
        >>> encoded_audio = tokenizer.encode("Audio description", modality='audio')
        >>> decoded_audio = tokenizer.decode(encoded_audio)
        >>> print("Encoded audio:", encoded_audio)
        >>> print("Decoded audio:", decoded_audio)
    """

    def __init__(
        self, model_path: Optional[str] = None, tokenizer_name: Optional[str] = None
    ):
        # Try to use local tokenizer model first
        if model_path:
            assert os.path.isfile(model_path), f"Model path does not exist: {model_path}"
            resolved_model_path = model_path
        else:
            # Try to find local tokenizer model in the project
            current_dir = Path(__file__).parent
            project_root = current_dir.parent  # Go up to project root
            local_tokenizer_path = project_root / "tokenizer" / "tokenizer.model"

            if local_tokenizer_path.exists():
                resolved_model_path = str(local_tokenizer_path)
                logger.info(f"Using local tokenizer model: {resolved_model_path}")
            elif tokenizer_name:
                resolved_model_path = self.download_tokenizer(tokenizer_name)
            else:
                # Default fallback - check a few common locations
                fallback_paths = [
                    Path.cwd() / "tokenizer" / "tokenizer.model",
                    Path.cwd() / "data" / "tokenizer.model",
                    Path.home() / ".gemini" / "tokenizer.model"
                ]

                found_path = None
                for path in fallback_paths:
                    if path.exists():
                        found_path = str(path)
                        break

                if found_path:
                    resolved_model_path = found_path
                    logger.info(f"Using fallback tokenizer model: {resolved_model_path}")
                else:
                    # Last resort - download default
                    resolved_model_path = self.download_tokenizer("hf-internal-testing/llama-tokenizer")

        self.sp_model = SentencePieceProcessor(model_file=resolved_model_path)
        logger.info(f"Reloaded SentencePiece model from {resolved_model_path}")

        # Initialize token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        # Initialize special token IDs for modalities
        # Try to get modality tokens, but don't fail if they don't exist
        try:
            self.modality_tokens = {
                "image": (
                    self.sp_model.piece_to_id("<img>"),
                    self.sp_model.piece_to_id("</img>"),
                ),
                "audio": (
                    self.sp_model.piece_to_id("<audio>"),
                    self.sp_model.piece_to_id("</audio>"),
                ),
            }
        except Exception as e:
            logger.warning(f"Could not load modality tokens: {e}")
            self.modality_tokens = {
                "image": (-1, -1),
                "audio": (-1, -1),
            }

    @staticmethod
    def download_tokenizer(tokenizer_name: str) -> str:
        """Downloads the SentencePiece model file from HuggingFace Hub.

        Args:
            tokenizer_name (str): Name of the tokenizer to download

        Raises:
            ValueError: If tokenizer name is not available
            Exception: If download fails

        Returns:
            str: Path to downloaded tokenizer model file
        """
        if tokenizer_name not in PRETRAINED_VOCAB_FILES_MAP["vocab_file"]:
            raise ValueError(f"Tokenizer {tokenizer_name} is not available.")

        model_url = PRETRAINED_VOCAB_FILES_MAP["vocab_file"][tokenizer_name]

        # Create data directory in current working directory
        data_dir = Path.cwd() / "data"
        data_dir.mkdir(exist_ok=True)
        model_path = data_dir / "tokenizer.model"

        # Check if already downloaded
        if model_path.exists():
            logger.info(f"Using cached tokenizer model: {model_path}")
            return str(model_path)

        # Download the tokenizer model file
        logger.info(f"Downloading tokenizer model from {model_url}")
        try:
            response = requests.get(model_url, timeout=30)
            response.raise_for_status()

            with open(model_path, "wb") as file:
                file.write(response.content)
            logger.info(f"Downloaded SentencePiece model to {model_path}")

        except requests.RequestException as e:
            raise Exception(f"Failed to download model from {model_url}: {e}")

        return str(model_path)

    def encode(
        self, s: str, modality: str = "text", bos: bool = True, eos: bool = True
    ) -> List[int]:
        """Encodes a string into a list of token IDs.

        Args:
            s (str): Input string to encode
            modality (str): Modality type ('text', 'image', 'audio')
            bos (bool, optional): Whether to add beginning of sequence token. Defaults to True.
            eos (bool, optional): Whether to add end of sequence token. Defaults to True.

        Returns:
            List[int]: List of token IDs
        """
        assert isinstance(s, str), "Input must be a string"

        try:
            tokens = self.sp_model.encode(s)
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            # Fallback: return empty list or basic encoding
            tokens = []

        # Prepend start and append end modality tokens if available
        if modality in self.modality_tokens:
            modality_start_id, modality_end_id = self.modality_tokens[modality]
            if modality_start_id != -1 and modality_end_id != -1:
                tokens = [modality_start_id] + tokens + [modality_end_id]

        # Add BOS/EOS tokens if required
        if bos and self.bos_id != -1:
            tokens = [self.bos_id] + tokens
        if eos and self.eos_id != -1:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of token IDs into a string.

        Args:
            tokens (List[int]): List of token IDs to decode

        Returns:
            str: Decoded string
        """
        try:
            # Remove modality tokens before decoding
            filtered_tokens = []
            for token in tokens:
                # Skip special modality tokens
                is_modality_token = False
                for start_id, end_id in self.modality_tokens.values():
                    if token in (start_id, end_id) and start_id != -1 and end_id != -1:
                        is_modality_token = True
                        break

                if not is_modality_token:
                    filtered_tokens.append(token)

            return self.sp_model.decode(filtered_tokens)

        except Exception as e:
            logger.error(f"Failed to decode tokens: {e}")
            return ""

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.n_words

    def get_special_tokens(self) -> dict:
        """Get special token IDs."""
        return {
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "unk_id": self.sp_model.unk_id(),
        }

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into pieces (for debugging)."""
        try:
            return self.sp_model.encode_as_pieces(text)
        except Exception as e:
            logger.error(f"Failed to tokenize text: {e}")
            return []

    def detokenize(self, pieces: List[str]) -> str:
        """Detokenize pieces back to text (for debugging)."""
        try:
            return self.sp_model.decode_pieces(pieces)
        except Exception as e:
            logger.error(f"Failed to detokenize pieces: {e}")
            return ""