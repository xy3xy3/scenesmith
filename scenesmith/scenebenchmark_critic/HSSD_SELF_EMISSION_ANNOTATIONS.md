# HSSD Self-Emission Annotations

Date: 2026-07-15

This portable layer adds a `self_emission` record to every one of the 10,963
HSSD assets bundled on the `yz` branch. It is intended for SceneSmith lighting
priors, critic metadata, and deterministic asset queries.

## Coverage

| emission class | assets |
|---|---:|
| `luminaire` | 998 |
| `emissive_display` | 63 |
| `flame` | 129 |
| `none` | 9,773 |
| total | 10,963 |

Positive total: 1,190.

The full GLB audit read 10,861 real HSSD models; 102 geometry paths were absent,
and zero readable GLBs declared native `emissiveFactor`, `emissiveTexture`, or
non-default `KHR_materials_emissive_strength`. The positive labels are therefore
explicit semantic capability estimates, not claims that the original material
already emits light in Blender or glTF renderers.

## Fields and units

Each annotation separates:

- `is_self_emissive`: whether the asset can emit visible light in its active
  state.
- `emission_class`: luminaire, extended display, flame, or none.
- `detection`: category evidence and native material audit evidence.
- `photometry.electrical_power_w`: active electrical input power estimate.
- `photometry.luminous_flux_lm`: total visible flux.
- `photometry.luminous_intensity_cd`: average directional intensity for a
  point/cone model.
- `photometry.luminance_cd_m2`: extended display luminance.
- `estimate_scope`, `confidence`, and `provenance`: prevent category priors
  from being mistaken for asset-specific measurements.

For a uniform idealized distribution, intensity is calculated from IES/NIST
photometric definitions:

```text
I_v = Phi_v / Omega
Omega_cone = 2 pi (1 - cos(theta / 2))
```

Displays use luminance rather than a fabricated point intensity. Projector
intensity stays null when the lens solid angle is unavailable.

## SceneSmith access

Full merged retrieval naturally includes `record["self_emission"]`:

```python
store = AssetLibraryAnnotationStore()
record = store.require(hssd_id)
emission = record["self_emission"]
```

The lightweight path avoids loading optional affordance/clearance enrichment:

```python
emission = store.get_self_emission_annotations(hssd_id)
lights = store.search_self_emission(
    is_self_emissive=True,
    emission_class="luminaire",
    limit=100,
)
```

Command line:

```bash
python scripts/query_asset_library_annotations.py \
  <hssd_id> --self-emission-only
```

## Provenance and limitations

The versioned category profiles and source URLs are bundled at
`asset_annotation_data/self_emission_profiles.json`. They reference NIST and
IES unit definitions, the US Department of Energy LED/TV guidance, and
representative Philips, Dell, and Epson specifications.

Values are category-typical priors. They do not infer switch state, dimmer
setting, installed bulb, screen content, shade transmission, flame size, or an
asset-specific IES/LDT distribution. Scene generation should instantiate light
sources only when the scene state says the object is on, and should retain the
annotation confidence when sampling around the category value.

