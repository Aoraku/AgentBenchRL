# Third-Party Data and Code Boundary

The MIT license in this repository applies to the AgentBenchRLFrame framework,
including its modified SnakeGo port, tests, configuration, and reporting code.

`src/games/snakego` is a modified port derived from AgentBench/THUAC 2022
SnakeGo controller semantics and code under the AgentBench MIT license. The
upstream repository is `https://github.com/Aoraku/AgentBench.git`; the
controller corpus is recorded at commit
`b581bca3ba3d2d7d58a2f8c6bbddd060fc7fdc87`, and its root MIT notice originates
at commit `b17a1fe7d39a0a82eeca4da80a2a30c6db663f03`. The exact upstream notice is
reproduced in `LICENSES/AgentBench-MIT.txt`. AgentBenchRLFrame implementation
lineage starts at `5c1dbfe3a9c7fd453e3d462b601a43c1fe3bbbfa`; the SnakeGo port enters that
lineage at `7caffd83721e69717c3797a530f974f7bd1adae2`.

The historical upstream URL may require authorization. The
immutable public verification source is
`provenance/agentbench-snakego-controller/`: it contains the seven exact
official controller files relevant to this port, their original paths and
SHA-256 digests, the source and license commits, and the AgentBench MIT notice.
`RIGHTS.md` records the Qingle copyright and commit-author evidence supporting
publication of this official-controller snapshot under those MIT terms.

This repository contains the permitted modified SnakeGo port described above.
It does not redistribute the upstream AgentBench corpus tree, contestant
source code, contestant archives, or a full contestant manifest.
Population blueprints contain only benchmark identifiers, content hashes, and
neutral expected paths. A user may supply an external AgentBench data root for
optional human-agent builds and evaluations.

External AgentBench data and contestant programs remain separate works. Their
use and redistribution are governed by their respective terms, if any. This
notice makes no claim about permissions for third-party materials.

The AgentBench MIT license and the repository's root MIT license do not cover,
relicense, or imply permission for contestant submissions.

The frozen Task 10 evaluation evidence names eight public competition handles
and archive basenames solely for attribution and reproducibility of recorded
match facts. Those identifiers do not include archive bytes or source trees
and do not grant or imply a license to the associated contestant programs.
