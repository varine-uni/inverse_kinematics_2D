import numpy as np
import matplotlib.pyplot as plt

# Forward Kinematics Function (2D case)
def forward_kinematics(theta1, theta2, l1, l2):
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return x1, y1, x2, y2

# Jacobian Matrix Function (2D case)
def jacobian(theta1, theta2, l1, l2):
    J11 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    J12 = -l2 * np.sin(theta1 + theta2)
    J21 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    J22 = l2 * np.cos(theta1 + theta2)
    return np.array([[J11, J12], [J21, J22]])

# Inverse Kinematics Function
def inverse_kinematics(x, y, l1, l2, initial_angles=[0, 0], max_iterations=100, tolerance=1e-6):
    theta = np.array(initial_angles)
    for _ in range(max_iterations):
        end_effector = np.array(forward_kinematics(theta[0], theta[1], l1, l2)[:2])
        error = np.array([x, y]) - end_effector
        if np.linalg.norm(error) < tolerance:
            break
        delta_theta = np.linalg.solve(jacobian(theta[0], theta[1], l1, l2)[:2, :], error)
        theta += delta_theta
    return theta

# Visualization for Two-Segment Arm
def visualize_two_segment_arm(theta1, theta2, l1, l2):
    x1, y1, x2, y2 = forward_kinematics(theta1, theta2, l1, l2)
    
    plt.plot([0, x1], [0, y1], 'bo-')  # Connect to fixed base
    plt.plot([x1, x2], [y1, y2], 'bo-')  # First segment
    plt.plot(x2, y2, 'ro')  # End effector
    plt.xlim(-l1 - l2, l1 + l2)
    plt.ylim(-l1 - l2, l1 + l2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

l1 = 2  # length of the first segment
l2 = 2  # length of the second segment
target_x = 2
target_y = 1

initial_angles = [0.5, 0.5]  # Initial guess for joint angles
joint_angles = inverse_kinematics(target_x, target_y, l1, l2, initial_angles)

print("Joint Angles:", joint_angles)

visualize_two_segment_arm(joint_angles[0], joint_angles[1], l1, l2)