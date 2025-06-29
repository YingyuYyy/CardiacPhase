"""
This script contains the latent motion trajectory analysis functions. Example of using them can be found in the provided notebook. 
Author: Yingyu Yang 
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, savgol_filter
from scipy.signal import butter, filtfilt
from time import time 
import torch 
import os 
from tqdm import tqdm


def highpass_filter(signal, fs, cutoff=0.5, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered


def detect_baseline_wander(signal_data, sampling_rate, cutoff_freq=0.5):
    # Apply FFT
    fft = np.fft.fft(signal_data)
    frequencies = np.fft.fftfreq(len(signal_data), 1/sampling_rate)
    
    # Calculate power in low frequency band
    low_freq_mask = np.abs(frequencies) < cutoff_freq
    low_freq_power = np.sum(np.abs(fft[low_freq_mask])**2)
    total_power = np.sum(np.abs(fft)**2)
    
    # If low frequency power is above threshold, baseline wander is present
    ratio = low_freq_power / total_power
    return ratio > 0.1  # Threshold can be adjusted based on your needs


def compute_main_orientation_and_extrema(trajectory, fps, theta_threshold_degrees=30, 
                                        ransac_iterations=100, window_length = 8, polyorder = 2, 
                                        edge_events=True,
                                        visualize=False):
    """
    Compute the main motion orientation and oscillation extreme points of a 2D trajectory.
    
    Parameters:
    - trajectory: numpy array of shape (T, 2) representing the 2D points.
    - fps: frame per second, used for baseline wander correction.
    - theta_threshold_degrees: Angular threshold for inliers (degrees).
    - ransac_iterations: Number of RANSAC iterations.
    - window_length: window size for savgol_filter smoothing 
    - polyorder: order of savgol_filter smoothing 
    - edge_events: whether to detect edge extremity 
    
    Returns:
    - group1: indices of points belonging to the first group. 
    - group2: indices of points belonging to the second group.
    - endpoint1: the position of one end point of the main axis 
    - endpoint2: the position of the second end point of the main axis 
    - trajectory_projected: the 1D trajectory after projecting the 2D positions to the main axis 
    - direction: vector of the main axis, used for projection computing 
    """
    # --- Step 1: Compute main orientation using RANSAC & PCA ---
    displacements = np.diff(trajectory, axis=0)
    magnitudes = np.linalg.norm(displacements, axis=1)
    nonzero_mask = magnitudes > 1e-8
    displacements = displacements[nonzero_mask]
    magnitudes = magnitudes[nonzero_mask]
    
    if len(displacements) == 0:
        raise ValueError("All displacements are zero.")
    
    unit_vectors = displacements / magnitudes[:, np.newaxis]
    theta_threshold = np.deg2rad(theta_threshold_degrees)
    cos_threshold = np.cos(theta_threshold)
    
    best_num_inliers = -1
    best_inliers = None
    
    for _ in range(ransac_iterations):
        idx = np.random.randint(len(unit_vectors))
        hypothesis = unit_vectors[idx]
        dots = np.abs(np.dot(unit_vectors, hypothesis))
        inliers = dots >= cos_threshold
        num_inliers = np.sum(inliers)
        
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
    
    if best_num_inliers == 0:
        raise ValueError("No inliers found.")
    
    inlier_displacements = displacements[best_inliers]
    pca = PCA(n_components=1)
    pca.fit(inlier_displacements)
    main_orientation = pca.components_[0]
    angle = np.arctan2(main_orientation[1], main_orientation[0])
    
    # --- Step 2: Project trajectory onto main axis to find extrema ---
    mean_point = np.mean(trajectory, axis=0)
    direction = np.array([np.cos(angle), np.sin(angle)])
    trajectory_projected = np.dot(trajectory - mean_point, direction)
    
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    smoothed_x = savgol_filter(x, window_length, polyorder)
    smoothed_y = savgol_filter(y, window_length, polyorder)

    mean = np.mean(trajectory, axis=0)
    v = direction
    t_min, t_max = np.min(trajectory_projected), np.max(trajectory_projected)
    #print(v, t_min, t_max)
    endpoint1 = mean + t_min * v  # Endpoint 1 (min projection)
    endpoint2 = mean + t_max * v  # Endpoint 2 (max projection)

    # 4. Detect peaks and valleys in the projections (direction changes)
    smoothed_proj = savgol_filter(trajectory_projected, window_length, polyorder)
    
    wander_flag = detect_baseline_wander(trajectory_projected, fps)
    # baseline wandering detection, if detected, we apply a high pass filter. 
    if wander_flag:
        # to avoid edge effect and better preserve the edge points 
        new_traj = np.pad(smoothed_proj, (10, 10), 'reflect')
        filtered_proj = highpass_filter(new_traj, fps)[10:-10]
    else:
        filtered_proj = smoothed_proj
    prominence_threshold = 0.3 * (np.max(filtered_proj) - np.min(filtered_proj))  # Adjust based on data

    if edge_events:
        '''
        EchoNet-Dynamic only provide one ground truth ED/ES for each sequence, some of the GT indices are the first/last frame of the video.
        Peak detection functions are bad at detecting extremites that happen at the edge positions. 
        We concatenate one value in the begining to make sure that we won't miss such edge extremites. 
        '''
        peak_input = np.concatenate(([min(filtered_proj)],filtered_proj,[min(filtered_proj)]))
        valley_input = np.concatenate(([max(filtered_proj)],filtered_proj,[max(filtered_proj)]))
    peaks, _ = find_peaks(peak_input, prominence=prominence_threshold)
    valleys, _ = find_peaks(-valley_input, prominence=prominence_threshold)
    if edge_events:
        peaks = peaks - 1 
        valleys = valleys - 1

    # 5. Group direction changes by proximity to endpoints
    # (ED and ES may be opposite, need to be checked using validtion data, in my trained model, ED are in valleys)
    group1_indices = valleys  # Closer to endpoint1 #ED 
    group2_indices = peaks  # Closer to endpoint2 #ES

    # Convert to numpy arrays
    group1 = np.array(group1_indices)
    group2 = np.array(group2_indices)

    if visualize:
        # Plot results
        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.scatter(x, y, c='gray', alpha=0.3, label='Original trajectory position')
        plt.plot(smoothed_x, smoothed_y, 'b-', lw=1, alpha=0.6, label='Smoothed Path')
        try:
            plt.scatter(x[group1], y[group1], c='green', s=50, marker='s', edgecolor='black', label='End-diastole')
        except:
            pass
        try:
            plt.scatter(x[group2], y[group2], c='orange', s=50, marker='s', edgecolor='black', label='End-systole')
        except:
            pass
        plt.plot([endpoint1[0], endpoint2[0]], [endpoint1[1], endpoint2[1]], 
                'r--', linewidth=2, label='Principal Direction')
        # Annotate mean and direction vector
        plt.scatter(mean[0], mean[1], c='red', s=80, marker='X', label='Mean Position')
        plt.quiver(mean[0], mean[1], v[0], v[1], color='red', scale=10, 
                width=0.005, label='Direction Vector')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.title('Direction Changes Grouped by Endpoint Proximity')
        plt.subplot(212)
        plt.plot(trajectory_projected, label='original')
        plt.plot(smoothed_proj, label='smoothed')
        plt.plot(filtered_proj, label='filtered')
        try:
            plt.scatter(group1, filtered_proj[group1], label='ED')
        except:
            pass
        try:
            plt.scatter(group2, filtered_proj[group2], label='ES')
        except:
            pass
        plt.legend()
        plt.show()
        print("Group 1 (near Endpoint 1) indices:", group1)
        print("Group 2 (near Endpoint 2) indices:", group2)

    return group1, group2, endpoint1, endpoint2, trajectory_projected, direction