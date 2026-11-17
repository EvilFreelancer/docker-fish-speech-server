import unittest
from types import SimpleNamespace

from fish_speech_api.utils.cudagraphs import mark_compiled_step_starts


class CudaGraphStepMarkerTest(unittest.TestCase):
    def setUp(self):
        self.steps = []
        self.original_calls = []

        def compile_function(function, **kwargs):
            self.original_calls.append((function, kwargs))
            return function

        self.torch = SimpleNamespace(
            compile=compile_function,
            compiler=SimpleNamespace(cudagraph_mark_step_begin=lambda: self.steps.append("step")),
        )
        self.original_compile = self.torch.compile

    def test_marks_every_compiled_function_call(self):
        with mark_compiled_step_starts(self.torch):
            compiled = self.torch.compile(lambda value: value + 1, fullgraph=True)
            self.assertEqual(2, compiled(1))
            self.assertEqual(3, compiled(2))
            self.assertEqual(["step", "step"], self.steps)
            self.assertEqual(1, len(self.original_calls))
            self.assertTrue(self.original_calls[0][1]["fullgraph"])

        self.assertIs(self.original_compile, self.torch.compile)

    def test_restores_compile_after_exception(self):
        with self.assertRaisesRegex(ValueError, "inference failed"):
            with mark_compiled_step_starts(self.torch):
                raise ValueError("inference failed")

        self.assertIs(self.original_compile, self.torch.compile)

    def test_requires_cudagraph_marker(self):
        torch_without_marker = SimpleNamespace(compile=self.original_compile, compiler=SimpleNamespace())

        with self.assertRaisesRegex(RuntimeError, "cudagraph_mark_step_begin"):
            with mark_compiled_step_starts(torch_without_marker):
                pass


if __name__ == "__main__":
    unittest.main()
