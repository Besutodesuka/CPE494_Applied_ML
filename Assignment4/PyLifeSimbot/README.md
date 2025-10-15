# PyLifeSimbot - Artificial Life Simulation

A genetic algorithm-based robot simulation where autonomous agents evolve survival strategies through continuous natural selection. Unlike traditional GA implementations that use fixed generations, this project simulates a living ecosystem where robots compete for food and energy until they die.

## Project Overview

This is an evolutionary robotics simulation where:
- 20 robots navigate an environment with obstacles and limited food
- Each robot has a chromosome encoding fuzzy logic rules for navigation
- Robots with better survival skills live longer and have more chances to reproduce
- When a robot dies, it's replaced by offspring from successful survivors
- Over time, the population evolves better food-seeking and obstacle-avoidance behaviors

Think of it as "robot natural selection" happening in real-time.

## Requirements

- Python 3.x
- For Windows: `pip install -r requirements_windows.txt`
- For macOS: `pip3 install -r requirements_macos.txt`

Key dependencies:
- `kivy` - simulation visualization
- `matplotlib` - learning curve plotting
- `numpy` - numerical operations

## Quick Start

```bash
python assignment4.py
```

The simulation will run for 100,000 ticks. You'll see:
- A visual window showing robots (circles) navigating around obstacles (blue squares)
- Food items (pink squares) that respawn when eaten
- Robot colors change based on energy level (yellow = low, green/blue = high)
- Console output showing when robots die
- Learning curve plots generated every 1000 ticks

## How It Works

### Robot Chromosome

Each robot has 10 fuzzy rules, with each rule containing 11 genes (110 total):

```
Rule = [S0, S1, S2, S3, S4, S5, S6, S7, Smell, Turn, Move]
```

- **S0-S7**: IR sensor readings (8 directions around the robot)
  - Values 0-255, mod 5 → 0=ignore, 1=near, 2=far
- **Smell**: Food direction sensor
  - Values 0-255, mod 6 → 1=left, 2=center, 3=right
- **Turn**: Rotation angle (-90° to +90°)
- **Move**: Movement speed (-10 to +10)

The fuzzy logic controller evaluates all rules and combines their outputs to determine movement.

### Energy Mechanics

Every robot starts with **300 energy**. Energy changes based on actions:

| Event | Energy Change | Reason |
|-------|--------------|---------|
| Each tick | -1 | Base metabolism |
| Stationary (lazy) | -40 | Penalty for not moving |
| Sharp turn/reverse | -15 | Inefficient movement |
| Collision | -30 | Hitting obstacles |
| Eating food | +300 | Reward for finding food |

When energy hits 0, the robot dies and is replaced via genetic operations.

### Evolutionary Algorithm

**Selection**: Tournament selection from top 25%
- Only robots with positive energy can be parents
- Sorted by current energy level
- Random pick from the best quartile

**Crossover**: Uniform crossover
- Each gene has 50% chance from parent1 or parent2
- Creates diverse offspring without position bias

**Mutation**: 1% per gene
- Random genes flip to new values (0-255)
- Maintains genetic diversity

**Random injection**: 1% chance
- Occasionally create completely random robots
- Prevents premature convergence

### Death and Rebirth Cycle

```python
if robot.energy < 0:
    1. Log the death (tick number)
    2. Select two parent robots from survivors
    3. Generate offspring via crossover
    4. Apply mutation
    5. Replace dead robot with offspring
    6. Reset energy to 300
```

This happens continuously throughout the simulation - no generational boundaries.

## Output Files

### best_robot.csv
The chromosome of the robot with highest energy at the end. Format:
```
gene0, gene1, gene2, ..., gene10  # Rule 0
gene0, gene1, gene2, ..., gene10  # Rule 1
...
```

### Learning Curves
- `learning_curve_tick_XXXX.png` - snapshots every 1000 ticks
- `learning_curve_final.png` - final plot after 100k ticks

The bar chart shows death counts per 1000-tick interval. A downward trend indicates successful learning - robots are living longer as evolution progresses.

## Tuning Parameters

You can adjust these in the code:

**Environment** (lines 399-405):
```python
num_robots = 20          # Population size
num_objectives = 4       # Number of food items
max_tick = 100000        # Simulation length
```

**Energy penalties** (lines 108-124):
```python
lazy_penalty = 40
headache_penalty = 15
hit_penalty = 30
eat_reward = 300
```

**GA parameters** (lines 194, 168):
```python
mutation_rate = 0.01
random_robot_rate = 0.01
```

**Plotting** (line 20):
```python
interval_size = 1000     # Death counting window
```

## What to Expect

**Early simulation (0-10k ticks)**:
- High death rate (~2000+ per interval)
- Random, inefficient movement
- Frequent collisions
- Many robots die quickly

**Mid simulation (10k-50k ticks)**:
- Death rate decreases steadily
- Better obstacle avoidance emerges
- Some robots find food efficiently
- Population becomes more competent

**Late simulation (50k-100k ticks)**:
- Death rate stabilizes (~200-300 per interval)
- Dominant strategies emerge
- Most robots can navigate and find food
- Occasional deaths from bad luck or edge cases

## Debugging Tips

**Robots all die immediately?**
- Check energy penalties aren't too harsh
- Verify food is spawning (`num_objectives=4`)
- Make sure eat reward is sufficient (+300)

**No evolution happening?**
- Verify mutation rate isn't zero
- Check that selection isn't completely random
- Make sure crossover is actually mixing genes

**Simulation too slow?**
- Reduce `maxfps` in Config (line 16)
- Increase `interval` parameter (line 405)
- Disable plotting during run

**Learning curve looks flat?**
- Might need more ticks (increase max_tick)
- Energy rewards/penalties may need balancing
- Check if all robots are dying at same rate (no selection pressure)

## Implementation Notes

This was built for AML Assignment 4. The main additions to the base PySimbot framework:

1. Energy system with penalties/rewards
2. Death detection and replacement
3. Tournament selection from survivors
4. Uniform crossover implementation
5. Mutation with configurable rate
6. Death tracking across simulation
7. Automated learning curve plotting

The fuzzy logic controller was provided - we added the evolutionary layer on top.

## License

Educational project for CPE494 Applied Machine Learning.
