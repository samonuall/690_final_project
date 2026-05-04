# OrchardLoop Environment

## Task Description

A picker agent navigates a small orchard arranged in a loop. The goal is to travel clockwise around the loop, passing through checkpoint markers.

## Grid Layout

The orchard is a 5×5 grid surrounded by impassable walls, with a single impassable obstacle in the centre forcing a loop path:

```
#####
#P m#
#n#o#
# p #
#####
```

The four marker tiles (m, n, o, p) sit at the corners of the loop.

## Tile Encoding

The observation is a float32 array of shape **(1, 5, 5)** — a single-channel board where each cell holds one of these values:

| Value | Tile              | Meaning                                      |
|-------|-------------------|----------------------------------------------|
| 0.0   | `#` (WALL)        | Impassable boundary or central obstacle      |
| 1.0   | ` ` (EMPTY)       | Walkable floor                               |
| 2.0   | `P` (AGENT)       | Current agent position                       |
| 3.0   | `m/n/o/p` (MARKER)| Checkpoint on the loop — all four look the same |

**Important:** all four marker tiles share the same observation value (3.0). The reward function cannot distinguish which marker the agent is on from the tile value alone.

## Actions

There are **5 discrete actions** (0-indexed):

| Index | Direction |
|-------|-----------|
| 0     | NOOP      |
| 1     | Up        |
| 2     | Down      |
| 3     | Left      |
| 4     | Right     |

## Success Criteria

The agent should complete as many clockwise loops as possible within the step limit.
