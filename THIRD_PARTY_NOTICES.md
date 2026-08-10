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
