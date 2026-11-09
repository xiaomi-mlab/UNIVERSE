# Third-Party Runtime Dependencies

This directory keeps the runtime repositories used by UNIVERSE inference:

```text
third_party/navsim/
third_party/nuscenes-devkit/
```

`navsim/` is the bundled NavSIM runtime used by the NavSIM launcher.

`nuscenes-devkit/` is not vendored in this repository. Clone the upstream
nuScenes devkit repository into `third_party/nuscenes-devkit` locally:

```bash
git clone https://github.com/nutonomy/nuscenes-devkit.git third_party/nuscenes-devkit
```

The UNIVERSE launchers add `third_party/` and
`third_party/nuscenes-devkit/python-sdk` to `PYTHONPATH`.
