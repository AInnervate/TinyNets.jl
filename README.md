# Pruning

This package allows pruning Flux models into sparse arrays.

## Iterative Pruning

An iterative pruning strategy first prunes the trained model and then retrains it (which is called *fine-tuning*) until the stop condition is achieved.  
A pruning schedule defines the sequence of pruning and fine tuning steps.

You can create an iterative scheduler using
```
schedule = [
    (PruneByPercentage(0.50), TuneByEpochs(1)),
    (PruneByPercentage(0.75), TuneByEpochs(3)),
    (PruneByPercentage(0.90), TuneByEpochs(5))
]
```
Then, you can run the schedule with
```
sparsemodel = scheduledpruning(trainedmodel, schedule, lossfunction, optiser, data)
```
A working example, training a model in the MNIST dataset, can be found in the [examples folder](/examples).

## Implemented Features

### Pruning

- Random pruning by percentage
- Random pruning by quantity
- Magnitude pruning by percentage
- Magnitude pruning by quantity

### Iterative Schedules

- Schedule by epochs
- Schedule by absolute loss value
- Schedule by loss difference from the previous epoch
