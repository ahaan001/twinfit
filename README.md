# Twinfit

Phone video of a real mechanism in, a verified MuJoCo simulation twin out.

Built during the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026).

## How it works

Twinfit films a mechanism in front of a printed ChArUco board, proposes its parts with Muse Wonder, segments and tracks them with SAM 3.1, turns the masks into metric motion with classical geometry, fits joint axes and a kinematic tree, identifies physically consistent dynamics, and issues a certificate that says how far the resulting MuJoCo model can be trusted. A Muse Wonder agent repairs runs that fail certification. Details: `docs/ARCHITECTURE.md`.

TBD at integration.

## Meta models used

TBD at integration.

## Results

<!-- RESULTS:BEGIN -->
TBD at integration.
<!-- RESULTS:END -->

## Quickstart

```
uv sync --all-extras --dev
uv run twinfit doctor
uv run twinfit synth --suite synth_small
uv run twinfit run ~/twinfit-data/synth/synth_small/pendulum_1r__s0 --impl stub
```

TBD at integration.

## Capturing your own data

Print the PDFs from `uv run twinfit printables`, then follow `docs/TEST_DATA_PROTOCOL.md`.

## Limitations

TBD at integration.

## License

MIT, see `LICENSE`. Third-party licenses: `THIRD_PARTY.md`.
