"""
Model loader for inference API.
Loads QNetwork weights once at startup.
"""

import os
import torch
import numpy as np
from src.agent.q_network import QNetwork


class ModelLoader:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = None
        self.model_path = None

    def _clean_path(self, path: str) -> str:
        """Fix Windows Git Bash path conversion issue."""
        # Git Bash converts /app to C:/Program Files/Git/app
        # This fixes it back to /app
        if "Program Files/Git" in path:
            path = path.split("Git")[-1]
            path = path.replace("\\", "/")
        return path

    def load(self, model_path: str, n_actions: int = 6):
        """Load model weights from checkpoint."""
        # Clean path first
        model_path = self._clean_path(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.n_actions  = n_actions
        self.model_path = model_path
        self.model      = QNetwork(n_actions).to(self.device)

        ckpt = torch.load(model_path, map_location=self.device)

        # Handle both full checkpoint and raw state_dict
        if "q_net" in ckpt:
            self.model.load_state_dict(ckpt["q_net"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()
        print(f"Model loaded from: {model_path}")
        print(f"Device: {self.device} | Actions: {n_actions}")

    def predict(self, state: list) -> tuple:
        """
        Run inference on a (4, 84, 84) state.

        Args:
            state: Nested list of shape (4, 84, 84)
        Returns:
            (action: int, q_values: list of floats)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        state_np = np.array(state, dtype=np.float32)        # (4, 84, 84)
        state_t  = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)  # (1,4,84,84)

        with torch.no_grad():
            q_values = self.model(state_t)

        q_np   = q_values.cpu().numpy()[0]
        action = int(np.argmax(q_np))
        return action, q_np.tolist()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


# Global singleton
model_loader = ModelLoader()