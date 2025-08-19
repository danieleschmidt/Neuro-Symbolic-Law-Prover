
# Compatibility fix for typing.Complex
import typing
if not hasattr(typing, 'Complex'):
    typing.Complex = complex
