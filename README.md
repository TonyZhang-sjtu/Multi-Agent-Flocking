# Layer 0 Flocking Simulator

This directory contains the Layer 0 foundation for the multi-agent flocking project: a lightweight 2-D double-integrator simulator, obstacle structures, simple validation controllers, metrics, and visualizations.

## Conda Environment

```bash
cd /opt/data/private/Multi-Agent/flocking
conda create -n mas_flocking python=3.10
conda activate mas_flocking
pip install -r requirements.txt
```

In this workspace, conda is available at `/opt/conda/bin/conda`, so non-interactive commands can use:

```bash
/opt/conda/bin/conda run -n mas_flocking python -m mas_flocking.main
```

## Run Demo

```bash
python -m mas_flocking.main
```

For a faster smoke test without GIF generation:

```bash
python -m mas_flocking.main --n-steps 200 --skip-animation
```

Outputs are written to:

- `outputs/figures/`
- `outputs/animations/`
- `outputs/logs/`



## Layer 1: Alpha-Agent Free-Space Flocking

Layer 1 implements Olfati-Saber Algorithm 1 interaction terms only: alpha-agent distance regulation plus velocity consensus. It intentionally does not include gamma-agent target navigation; migration belongs to Layer 2.

```bash
python -m mas_flocking.layer1_free_flocking --n-steps 1000 --skip-animation
```

To generate the Layer 1 GIF as well:

```bash
python -m mas_flocking.layer1_free_flocking --n-steps 1000
```

Layer 1 outputs are written under:

- `outputs/figures/layer1/`
- `outputs/animations/layer1/`
- `outputs/logs/layer1/`

Note: Algorithm 1 can fragment for generic initial states, as discussed in the original paper. This demo is meant to validate alpha-lattice formation mechanics and velocity consensus before adding the gamma-agent migration term in Layer 2.

## Layer 2: Gamma-Agent Target Navigation

Layer 2 implements Olfati-Saber Algorithm 2 in free space:

```text
u = u_alpha + u_gamma
```

The alpha-agent term maintains local spacing and velocity consensus. The gamma-agent term pulls the flock toward a virtual target state. By default, the static target matches the Layer 0 map goal:

```text
q_r = [18.0, 6.0], p_r = [0.0, 0.0]
```

Run the target-navigation demo:

```bash
python -m mas_flocking.layer2_target_navigation --n-steps 1200 --skip-animation
```

Generate the Layer 2 GIF as well:

```bash
python -m mas_flocking.layer2_target_navigation --n-steps 1200
```

Layer 2 outputs are written under:

- `outputs/figures/layer2/`
- `outputs/animations/layer2/`
- `outputs/logs/layer2/`

The module also accepts `--goal-vx` and `--goal-vy` for a dynamic gamma-agent, but the default experiment keeps the target static to match the Layer 0 setup.

## Layer 3: Static Beta-Agent Obstacle Avoidance

Layer 3 implements the static-obstacle part of Olfati-Saber Algorithm 3:

```text
u = u_alpha + u_beta + u_gamma
```

The default scenario reuses the Layer 0 obstacle positions, but freezes both obstacles as static circular obstacles. The beta-agent velocity defaults to the projected form, which damps normal motion into the obstacle while preserving tangential motion for smoother split/rejoin behavior.
The demo uses a shorter beta influence radius than the parameter dataclass default to avoid a local minimum in the Layer 0 geometry where the target lies directly behind the central obstacle.

```bash
python -m mas_flocking.layer3_static_obstacles --n-steps 1600 --skip-animation
```

Generate the Layer 3 GIF:

```bash
python -m mas_flocking.layer3_static_obstacles --n-steps 1600
```

Useful ablations:

```bash
python -m mas_flocking.layer3_static_obstacles --disable-beta --n-steps 1600 --skip-animation
python -m mas_flocking.layer3_static_obstacles --beta-velocity-mode zero --n-steps 1600 --skip-animation
python -m mas_flocking.layer3_static_obstacles --scenario narrow_passage --n-steps 1600 --skip-animation
```

Layer 3 outputs are written under:

- `outputs/figures/layer3/`
- `outputs/animations/layer3/`
- `outputs/logs/layer3/`

Layer 3 does not implement dynamic IAPF; moving-obstacle prediction and inhibiting velocity are reserved for Layer 4.

## Layer 4: Dynamic IAPF Obstacle Avoidance

Layer 4 adds a Shao-inspired dynamic IAPF term on top of Layer 3:

```text
u = u_alpha + u_beta + u_gamma + u_dyn
```

This layer uses closest-approach prediction, dynamic risk weighting, and an optional tangential bypass velocity. It is an engineering reproduction of the dynamic-obstacle idea in the pipeline, not a line-by-line reproduction of every formula in the IEEE TCE paper.

Run the default Layer3-matched scenario. It uses the same initial obstacle geometry as Layer 3; the upper obstacle is dynamic in Layer 4 so the IAPF term has a moving obstacle to predict.

```bash
python -m mas_flocking.layer4_dynamic_iapf --n-steps 1800 --skip-animation
```

Compare methods:

```bash
python -m mas_flocking.layer4_dynamic_iapf --method static_beta --n-steps 1800 --skip-animation
python -m mas_flocking.layer4_dynamic_iapf --method no_tangent --n-steps 1800 --skip-animation
python -m mas_flocking.layer4_dynamic_iapf --method no_avoidance --n-steps 1800 --skip-animation
python -m mas_flocking.layer4_dynamic_iapf --scenario multi_dynamic --n-steps 1800 --skip-animation
```

Generate the Layer 4 GIF:

```bash
python -m mas_flocking.layer4_dynamic_iapf --n-steps 1800
```

Layer 4 outputs are written under:

- `outputs/figures/layer4/`
- `outputs/animations/layer4/`
- `outputs/logs/layer4/`

## Run Tests

```bash
python -m unittest discover -s tests
```
