"""Pipeline execution mixin for data loaders.

Provides pipeline-based loading for loaders that need multi-step processing.
"""

from __future__ import annotations

from typing import Iterable, Callable, Any
from abc import abstractmethod

from colorama import Fore, Style


class PipelineMixin:
    """Mixin for loaders that use multi-step pipeline processing.

    Use this for complex loaders that need:
    - Multiple sequential processing steps
    - Progress reporting
    - Error handling per step
    - Step-by-step debugging

    Usage:
        class MyLoader(UniverseLoader, PipelineMixin):
            def _load_pipeline(self):
                return [
                    ('Step 1', self.step1_method, [], {}),
                    ('Step 2', self.step2_method, [], {'param': value}),
                ]

            def _load_raw(self):
                return self._execute_pipeline()
    """

    # Must be set by the class using this mixin
    MODALITY: str

    @abstractmethod
    def _load_pipeline(self) -> Iterable[tuple[str, Callable, list, dict[str, Any]]]:
        """Define the pipeline steps.

        Returns:
            List of tuples: (step_name, function, args, kwargs)
        """
        ...

    def _execute_pipeline(self, progress: bool = True, **pipeline_kwargs: Any) -> Any:
        """Execute the pipeline and return the final result.

        Args:
            progress: Whether to print progress messages (default: True)
            **pipeline_kwargs: Additional parameters passed to _load_pipeline()

        Returns:
            Result from the final pipeline step
        """
        pipeline = self._load_pipeline(**pipeline_kwargs)
        result = None

        for name, func, kwargs in pipeline:
            try:
                if result is not None:
                    result = func(result, **kwargs)
                else:
                    result = func(**kwargs)
                if progress:
                    self._log_step_success(name)
            except Exception as e:
                self._log_step_failure(name, e)
                raise

        # Return the result of the last step
        # final_step_name = [x[0] for x in pipeline][-1]
        return result

    def _log_step_success(self, step_name: str) -> None:
        """Print success message for a pipeline step."""
        # Calculate padding for alignment
        max_len = len('Consolidate Intersections')  # Longest typical step name
        padding = max_len - len(step_name) + 4

        modality = getattr(self, 'MODALITY', 'Loader').title()
        print(f'{modality} -- {step_name} {"-" * padding}> {Fore.GREEN}Complete{Style.RESET_ALL}')

    def _log_step_failure(self, step_name: str, error: Exception) -> None:
        """Print failure message for a pipeline step."""
        max_len = len('Consolidate Intersections')
        padding = max_len - len(step_name) + 4

        modality = getattr(self, 'MODALITY', 'Loader').title()
        print(f'{modality} -- {step_name} {"-" * padding}> {Fore.RED}Failed{Style.RESET_ALL}: {error}')
