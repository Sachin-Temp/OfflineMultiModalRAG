"""
VRAMManager — Singleton that tracks GPU memory usage and enforces
the 3.5GB VRAM ceiling. Every model load must go through this manager.
Implements LRU eviction of non-LLM models when ceiling is approached.
"""

import threading
from collections import OrderedDict
from loguru import logger
from config.settings import VRAM_CEILING_GB, VRAM_MODEL_SIZES


class VRAMManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # OrderedDict acts as LRU cache: most recently used at end
        self._loaded: OrderedDict[str, float] = OrderedDict()
        self._model_refs: dict[str, object] = {}  # model_name -> actual model object
        self._evict_callbacks: dict[str, callable] = {}  # model_name -> unload fn
        self.used_gb: float = 0.0
        self.ceiling_gb: float = VRAM_CEILING_GB
        logger.info(f"VRAMManager initialized. Ceiling: {self.ceiling_gb}GB")

    def register_evict_callback(self, model_name: str, callback: callable):
        """
        Register a function to call when model_name needs to be evicted.
        The callback is responsible for actually unloading the model from GPU.
        Example: vram_manager.register_evict_callback("clip", lambda: unload_clip())
        """
        self._evict_callbacks[model_name] = callback

    def acquire(self, model_name: str, priority: str = "normal") -> bool:
        """
        Request VRAM allocation for a model.
        - Evicts LRU non-LLM models if needed to stay under ceiling.
        - LLM models (llm_3b, llm_1b) are never evicted by this manager.
        - Returns True if acquired successfully, False if impossible.
        """
        with self._lock:
            if model_name in self._loaded:
                # Already loaded — move to end (most recently used)
                self._loaded.move_to_end(model_name)
                logger.debug(f"VRAMManager: {model_name} already loaded, refreshed LRU.")
                return True

            needed_gb = VRAM_MODEL_SIZES.get(model_name, 0.5)
            logger.info(f"VRAMManager: Requesting {needed_gb}GB for {model_name}. "
                        f"Current usage: {self.used_gb:.2f}GB / {self.ceiling_gb}GB")

            # Evict LRU non-LLM models until we have room
            while self.used_gb + needed_gb > self.ceiling_gb:
                evicted = self._evict_lru_non_llm()
                if not evicted:
                    logger.error(
                        f"VRAMManager: Cannot acquire {needed_gb}GB for {model_name}. "
                        f"No evictable models remain. Used: {self.used_gb:.2f}GB"
                    )
                    return False

            self._loaded[model_name] = needed_gb
            self.used_gb += needed_gb
            logger.info(f"VRAMManager: Acquired {needed_gb}GB for {model_name}. "
                        f"Total used: {self.used_gb:.2f}GB")
            return True

    def release(self, model_name: str):
        """
        Explicitly release a model's VRAM allocation.
        Does NOT unload the model — caller is responsible for that.
        """
        with self._lock:
            if model_name in self._loaded:
                freed = self._loaded.pop(model_name)
                self.used_gb = max(0.0, self.used_gb - freed)
                logger.info(f"VRAMManager: Released {freed}GB from {model_name}. "
                            f"Total used: {self.used_gb:.2f}GB")
            else:
                logger.warning(f"VRAMManager: release() called for untracked model {model_name}")

    def _evict_lru_non_llm(self) -> bool:
        """
        Evict the least recently used model that is NOT an LLM.
        Returns True if a model was evicted, False if none available.
        """
        LLM_MODELS = {"llm_3b", "llm_1b"}
        for model_name in list(self._loaded.keys()):  # oldest first
            if model_name not in LLM_MODELS:
                freed = self._loaded.pop(model_name)
                self.used_gb = max(0.0, self.used_gb - freed)
                logger.info(f"VRAMManager: LRU evicted {model_name}, freed {freed}GB. "
                            f"Total used: {self.used_gb:.2f}GB")
                # Call registered eviction callback if any
                if model_name in self._evict_callbacks:
                    try:
                        self._evict_callbacks[model_name]()
                    except Exception as e:
                        logger.error(f"VRAMManager: Eviction callback for {model_name} failed: {e}")
                return True
        return False

    def status(self) -> dict:
        """Return current VRAM usage summary."""
        return {
            "used_gb": round(self.used_gb, 3),
            "ceiling_gb": self.ceiling_gb,
            "available_gb": round(self.ceiling_gb - self.used_gb, 3),
            "loaded_models": list(self._loaded.keys()),
        }


# Module-level singleton accessor
vram_manager = VRAMManager()
