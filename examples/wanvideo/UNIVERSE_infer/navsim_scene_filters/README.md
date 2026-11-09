# NavSIM Scene Filters

These YAML files provide NavSIM scene filters for UNIVERSE evaluation.

Use one with:

```bash
SCENE_FILTER_YAML="${REPO_ROOT}/examples/wanvideo/UNIVERSE_infer/navsim_scene_filters/navtest_2000.yaml" \
bash examples/wanvideo/UNIVERSE_infer/scripts/eval_navsim_v1.sh
```

By default the wrapper script uses the YAML only for `log_names` and `tokens`, preserving CLI frame and horizon
settings. Set `SCENE_FILTER_YAML_FILTER_ONLY=0` to also use the YAML history/future/frame filter values.
