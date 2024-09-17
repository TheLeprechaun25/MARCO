
import torch
import numpy as np


def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems


def get_distances_from_coords(coords):
    # coords.shape: (batch, problem, 2)
    batch_size, problem_size, _ = coords.shape

    x = coords[:, :, [0]]
    y = coords[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    x_t = x.transpose(1, 2)
    y_t = y.transpose(1, 2)
    # x_t, y_t shape: (batch, 1, problem)

    x_diff = x - x_t
    y_diff = y - y_t
    # x_diff, y_diff shape: (batch, problem, problem)

    dist = torch.sqrt(x_diff ** 2 + y_diff ** 2)
    # dist shape: (batch, problem, problem)
    return dist


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


def rotate_tensor(coords, angle):
    """ Rotate the input tensor by the given angle. """
    theta = torch.tensor([
        [torch.cos(angle), -torch.sin(angle)],
        [torch.sin(angle), torch.cos(angle)]
    ]).to(coords.device)

    return torch.matmul(coords, theta.T)


def normalize_within_unit_square(coords):
    """ Normalize coordinates to be within the unit square [0,1] x [0,1]. """
    min_vals, _ = torch.min(coords, dim=1, keepdim=True)
    max_vals, _ = torch.max(coords, dim=1, keepdim=True)
    return (coords - min_vals) / (max_vals - min_vals)





def tsp_coordinate_augmentations(coords, num_rotations):
    """
    Apply rotational augmentations to TSP city coordinates.
    Args:
    - coords (Tensor): A 3D tensor of shape (batch, num_cities, 2) representing city coordinates.
    - num_rotations (int): Number of rotational augmentations to apply (1 for 90 degrees, 2 for 180 degrees, etc.).

    Returns:
    - List[Tensor]: A list of tensors, each containing augmented coordinates.
    """
    augmented_coords = []


    # Angle for each rotation
    angle_step = 2 * torch.pi / num_rotations

    # Apply rotations
    for i in range(num_rotations):
        angle = torch.tensor(i * angle_step)
        rotated_coords = rotate_tensor(coords, angle)
        normalized_coords = normalize_within_unit_square(rotated_coords)

        augmented_coords.append(normalized_coords)


        """import matplotlib.pyplot as plt
        x = coords[0, :, 0].cpu().numpy()
        y = coords[0, :, 1].cpu().numpy()
        x2 = rotated_coords[0, :, 0].cpu().numpy()
        y2 = rotated_coords[0, :, 1].cpu().numpy()
        x3 = normalized_coords[0, :, 0].cpu().numpy()
        y3 = normalized_coords[0, :, 1].cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, marker='o', color='blue', label='Cities')
        plt.scatter(x2, y2, marker='o', color='red', label='Rotated Cities')
        plt.scatter(x3, y3, marker='o', color='green', label='Normalized Cities')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.legend()
        plt.show()"""

    # Convert list of tensors to a single tensor
    augmented_coords = torch.cat(augmented_coords, dim=0)

    return augmented_coords
