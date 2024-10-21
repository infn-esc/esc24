### Options:
```
--help
```
Display help message.
```
--seed <value>
```
Set the random seed.

```
--weights <empty> <predator> <prey>
```
Set the integer weights for cell states.

```
--width <value>
```
Set the grid width (default: 200).

```
--height <value>
```
Set the grid height (default: 200).

```
--verify <file>
```
Verify the grid against a reference file.

### Simulation Rules:

- An empty cell becomes a prey if there are more than two preys surrounding it.
- A prey cell becomes empty if there is a single predator surrounding it and its level is higher than prey's level minus 10.
- A prey cell becomes a predator with level equal to max predator level + 1 if there are more than two predators and its level is smaller than the sum of the levels of the predators surrounding it.
- A prey cell becomes empty if there are no empty spaces surrounding it or if there are more than three preys surrounding it.
- A prey cell's level is increased by one if it survives starvation.
- A predator cell becomes empty if there are no preys surrounding it, or if all preys have levels higher than or equal to the predator's level.

### Example

Run the simulation with default parameters:

```bash
./game_of_life
```

Run the simulation with a specific seed and grid size:

```bash
./game_of_life --seed 42 --width 300 --height 300
```

The `--verify` option can be used to compare the final grid against a reference file.