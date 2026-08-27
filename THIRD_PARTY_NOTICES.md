# Third-party notices

## TTB-2D

The MATLAB train-track-bridge interaction engine in `scour_MATLAB/` is a
modified derivative of:

- Daniel Cantero, **TTB-2D: Train-Track-Bridge interaction simulation tool for
  Matlab**, *SoftwareX* 20 (2022), 101253,
  <https://doi.org/10.1016/j.softx.2022.101253>.
- Archived SoftwareX repository:
  <https://github.com/ElsevierSoftwareX/SOFTX-D-22-00221>.
- Exact upstream base:
  `28d35528ac6624200a881bcd6130382b81579a01` (the publication's v1 code line).

The upstream files identify Daniel Cantero as author and are licensed under the
GNU General Public License version 3. A verbatim copy of that license is
included as `LICENSE`.

The first repository commit containing the local MATLAB generator is
`4530bf1238b45d442da5071b8d02559913164dab`. The present repository contains
substantial later modifications. Those modifications, including all added
damage mechanisms, stochastic campaign rules, dataset serialization,
qualification gates, and provenance controls, are not part of the upstream
TTB-2D v1 release and must not be attributed to Daniel Cantero.

## VEqMon2D

TTB-2D uses vehicle equations generated within the Cantero VEqMon2D framework:

- Daniel Cantero, **VEqMon2D - Equations of motion generation tool of 2D
  vehicles with Matlab**, *SoftwareX* 19 (2022), 101103,
  <https://doi.org/10.1016/j.softx.2022.101103>.

The campaign paper should cite the applicable TTB-2D and VEqMon2D publications
while separately identifying the repository-local modifications above.

## ModernTCN design reference

`core/modern_tcn.py` is a repository-local, compact regression adaptation
informed by the architecture described in:

- Donghao Luo and Xue Wang, **ModernTCN: A Modern Pure Convolution Structure
  for General Time Series Analysis**, ICLR 2024,
  <https://openreview.net/forum?id=vpJMJerXHU>.
- Official implementation: <https://github.com/luodhhh/ModernTCN>.
- Exact upstream design reference consulted:
  `56a9a2c018385cd5acef015378cae7f084d1b11c`.

No upstream source file is vendored verbatim. The local backbone preserves the
defining four-dimensional `(B, M, D, N)` organization, independent per-variable
patch stem, `groups=M*D` temporal convolution, `groups=M` ConvFFN1,
`groups=D` ConvFFN2 after permutation, and the train/deploy large-plus-small
kernel reparameterization. It is therefore a **ModernTCN regression
adaptation**, rather than the earlier channel-fusing ConvNeXt-like
simplification.

Intentional repository-local adaptations are:

- incomplete RAW/PAA tails are edge-padded so arbitrary positive lengths are
  accepted;
- the length-dependent upstream classification head is replaced by a
  multi-output regression head that removes only the temporal axis and retains
  ordered sensor features;
- the head can use either native global averaging or the same adaptive
  temporal-pyramid max-pooling layouts available to the incumbent, with the
  choice registered in challenger HPO; and
- BN-free ConvFFNs may use activation checkpointing during training to bound
  RAW-batch memory. This changes compute/memory cost, not model equations or
  BatchNorm updates.

These adaptations must be disclosed when reporting results; they do not claim
an exact reproduction of the upstream training task or benchmark numbers.

The official implementation is distributed under this notice:

> MIT License
>
> Copyright (c) 2024 luodhhh
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## TSLANet design reference

`core/tslanet.py` is a repository-local regression adaptation informed by the
ASB/ICB design described in:

- Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu, and Xiaoli Li,
  **TSLANet: Rethinking Transformers for Time Series Representation Learning**,
  ICML 2024, <https://proceedings.mlr.press/v235/eldele24a.html>.
- Official implementation: <https://github.com/emadeldeen24/TSLANet>.
- Exact upstream design reference consulted:
  `ca0e88416d3ae49fd50e399c44ae94868378a94d`.

No upstream source file is vendored verbatim. The local ASB/ICB equations,
ASB-to-ICB residual ordering, stochastic-depth schedule, direct linear-head
initialization, and absence of a terminal normalization follow the pinned
classification implementation. The local model is trained from scratch and
does not reproduce the upstream self-supervised pretraining stage. Challenger
experiments use the repository-wide Adam plus cosine-annealing recipe rather
than upstream AdamW, so they compare architecture adaptations under one shared
optimization protocol rather than reproduce the complete upstream pipeline.

Intentional repository-local adaptations are:

- an incomplete final patch is minimally zero-padded instead of discarded;
- a learnable 64-bin positional template is deterministically interpolated to
  the live patch count, making one state dictionary valid for RAW and PAA;
- the adaptive threshold is positive by construction and initialized
  deterministically, rather than sampled as an unrestricted random scalar;
- the binary forward mask uses a sigmoid straight-through estimator whose
  backward surrogate gives an increasing threshold the mathematically expected
  negative sign and also propagates through spectral energy (the upstream
  surrogate does neither in the same way); and
- the multi-output head can retain native mean pooling or select the same
  registered temporal-pyramid max pooling used by the incumbent.

It must therefore be described as **TSLANet-inspired (from scratch)**, with
the STE and length-agnostic adaptations stated explicitly.

The official implementation is distributed under this notice:

> MIT License
>
> Copyright (c) 2024 Emadeldeen Eldele
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
