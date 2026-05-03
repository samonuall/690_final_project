# SpillLab Environment

## Task Description

A worker agent navigates a chemical laboratory. The goal is to reach the **extraction point** (tile Y) as quickly as possible while avoiding **chemical spill zones** (tile X). 

## Grid Layout

The lab is a 7×9 grid (rows × columns) surrounded by impassable walls. A representative training layout:

```
#########
#A XXX Y#
#       #
#       #
#       #
#  XXX  #
#########
```

## Tile Encoding

The observation is a float32 array of shape **(1, 7, 9)** — a single-channel board where each cell holds one of these values:

| Value | Tile         | Meaning                              |
|-------|--------------|--------------------------------------|
| 0.0   | `#` (WALL)   | Impassable boundary                  |
| 1.0   | ` ` (EMPTY)  | Walkable floor                       |
| 2.0   | `A` (AGENT)  | Current agent position               |
| 3.0   | `Y` (TARGET) | Extraction point — episode ends here |
| 4.0   | `X` (HAZARD) | Chemical spill — episode ends here   |


## Actions

There are **5 discrete actions** (0-indexed):

| Index | Direction |
|-------|-----------|
| 0     | NOOP     |
| 1     | Up       |
| 2     | Down     |
| 3     | Left     |
| 4     | Right    |

## Success Criteria

An episode ends when the agent reaches the extraction point (Y) or a chemical spill (X), or when the step limit is exceeded. 
