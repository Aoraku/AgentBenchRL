"""Installable game plugins discovered by AgentBenchRLFrame.

This is a top-level package (not ``rlbench.games``) on purpose: games are
plugins that live alongside the framework rather than inside it. Each
subpackage ``games/<name>/`` exposes a single ``PLUGIN`` object from its
``plugin`` module, which ``rlbench.registry`` finds by scanning this package
at runtime. SnakeGo (``games/snakego``) is the bundled reference game.
"""
