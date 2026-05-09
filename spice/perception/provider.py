from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from spice.decision.general import GenericObservation


class PerceptionProvider(ABC):
    provider_id: str

    def start(self, callback: Callable[[GenericObservation], None]) -> None:
        raise NotImplementedError("start() is reserved for future foreground/daemon providers.")

    def stop(self) -> None:
        raise NotImplementedError("stop() is reserved for future foreground/daemon providers.")

    @abstractmethod
    def poll(self) -> list[GenericObservation]:
        """Pull observations once."""
