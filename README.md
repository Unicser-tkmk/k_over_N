# Code Introduction: Evaluating Packet Loss Impact on Distributed Machine Learning

This script is designed to analyze the influence of packet loss on distributed machine learning (DML) tasks using a parameter server (PS) approach for data parallelism. The code trains a ResNet-101 model on the ImageNet dataset (or a subset, like Imagenette), implementing custom gradient manipulation strategies to simulate packet loss during gradient aggregation and worker communication. 

The main features of this script include:

- Simulation of packet loss by dropping or zeroing out a fraction of gradients during training.
- Customizable training parameters to study various scenarios of packet fragmentation and communication failure.

## Key Parameters for Testing

The script supports the following configurable parameters through command-line arguments:

| Parameter              | Description                                                  | Default Value |
| ---------------------- | ------------------------------------------------------------ | ------------- |
| `--num_epochs`         | Number of epochs to train the model.                         | 400           |
| `--batch_size`         | Batch size for training.                                     | 256           |
| `--learning_rate`      | Learning rate for the optimizer.                             | 0.01          |
| `--accumulation_steps` | Number of gradient accumulation steps.                       | 10            |
| `--loss_rate`          | Fraction of gradients to drop or zero out (simulate packet loss). | 0.0           |
| `--mtu`                | Maximum Transmission Unit for gradient fragmentation.        | 300           |
| `--use_zero`           | Whether to zero out dropped gradients.                       | `False`       |
| `--use_avg`            | Whether to average selected gradients instead of summing.    | `False`       |
| `--lab_round`          | Number of training rounds to repeat the experiment.          | 5             |
| `--worker_drop`        | Simulates gradient loss at the worker side (worker randomly drops a few packets). | `False`       |

## Output

Results, including metrics and model checkpoints, are saved in directories structured according to the parameter configurations. Metrics (e.g., loss and accuracy) are visualized and saved as plots.

This setup allows for a systematic evaluation of how various packet loss scenarios affect the convergence and performance of distributed training tasks. 
