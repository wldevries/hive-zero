"""CNN-era cell-permutation symmetry tests — obsoleted by the token rewrite.

The token transformer applies D6 augmentation by permuting per-token
`(q, r)` positions (via the 12 axial transforms in
`engine_zero.d6_axial_transforms`), not by permuting cells in a
`(C, G, G)` tensor. Token identities and policy targets stay invariant
under augmentation, so the elaborate cell-index remapping that this file
used to exercise no longer applies.

Equivalent property tests for the token side live in
`tests/test_transformer_training.py::test_d6_augmentation_permutes_token_positions`.
"""
import pytest

pytest.skip(
    "CNN cell-permutation D6 augmentation removed in token rewrite; "
    "see test_transformer_training.py for the token equivalent.",
    allow_module_level=True,
)
