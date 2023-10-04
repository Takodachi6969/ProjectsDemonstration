#!/usr/bin/env python
# coding: utf-8

# In[8]:
import numpy as np
import scipy.signal
import matplotlib.pylab as plt
import matplotlib.animation
import IPython.display
import time
import copy, re, itertools, json, csv
import re
import psutil
import pandas as pd
from skimage.metrics import structural_similarity
from skimage import transform
from skimage.measure import find_contours
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from scipy.ndimage import binary_fill_holes
from scipy import stats
from fractions import Fraction
from sklearn.metrics import normalized_mutual_info_score as nmi
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import binary_closing
import functools
from tqdm import tqdm


def get_cell_by_name(name, data):
    for item in data:
        if item['name'] == name:
            return item['cells']
    return None

def get_params(name, data):
    # loop through each entry in the data and check if its name matches
    for entry in data:
        if entry['name'] == name:
            # if a match is found, return the params
            return entry['params']
    # if no match is found, return None
    return None

def rle2arr(st):
		rle_groups = re.findall('(\d*)([p-y]?[.boA-X$])', st.rstrip('!'))  # [(2 yO)(1 $)(1 yO)]
		code_list = sum([[c] * (1 if n=='' else int(n)) for n,c in rle_groups], [])  # [yO yO $ yO]
		code_arr = [l.split(',') for l in ','.join(code_list).split('$')]  # [[yO yO] [yO]]
		V = [ [0 if c in ['.','b'] else 255 if c=='o' else ord(c)-ord('A')+1 if len(c)==1 else (ord(c[0])-ord('p'))*24+(ord(c[1])-ord('A')+25) for c in row if c!='' ] for row in code_arr]  # [[255 255] [255]]
		# lines = st.rstrip('!').split('$')
		# rle = [re.findall('(\d*)([p-y]?[.boA-X])', row) for row in lines]
		# code = [ sum([[c] * (1 if n=='' else int(n)) for n,c in row], []) for row in rle]
		# V = [ [0 if c in ['.','b'] else 255 if c=='o' else ord(c)-ord('A')+1 if len(c)==1 else (ord(c[0])-ord('p'))*24+(ord(c[1])-ord('A')+25) for c in row ] for row in code]
		maxlen = len(max(V, key=len))
		A = np.array([row + [0] * (maxlen - len(row)) for row in V])/255  # [[1 1] [1 0]]
		# print(sum(sum(r) for r in V))
		return A
    
# decorator function
def memory_usage(func):
    total_mem = 0

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal total_mem
        process = psutil.Process()
        mem_before = process.memory_info().rss
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss
        mem_diff = mem_after - mem_before
        total_mem += mem_diff
        return result

    def display_memory_usage():
        nonlocal total_mem
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        mem_unit = 0
        mem_usage = total_mem
        while mem_usage > 1024 and mem_unit < len(units) - 1:
            mem_usage /= 1024
            mem_unit += 1
        print(f"Total memory usage: {mem_usage:.2f} {units[mem_unit]}")

    wrapper.display_memory_usage = display_memory_usage

    return wrapper

from fractions import Fraction

def turn_to_float(b):
    fractions_list = b.split(',')
    floats_list = []  
    for fraction in fractions_list:
        if '/' in fraction:  
            numerator, denominator = fraction.split('/') 
            numerator = float(numerator) 
            denominator = float(denominator)  
            float_value = numerator / denominator  
            floats_list.append(float_value)
        else:
            floats_list.append(float(fraction))
    return floats_list


def soup_or_stable(arr):
    global num_positive
    subgrids = [arr[i:i+4, j:j+4] for i in range(0, 256, 4) for j in range(0, 256, 4)]
    pixel_sums = [subgrid.sum() for subgrid in subgrids]
    num_positive = sum([1 for ps in pixel_sums if ps > 0.2])
    if num_positive > 500:
        return 'soup'
    else:
        return 'stable'
    
def Image_Player(A):
    plt.figure(figsize=(6, 6))
    plt.imshow(A, cmap='gray', interpolation='none')
    plt.title('Final Image')
    plt.axis('off')
    plt.show()
    
from scipy.stats import entropy

def normalize(animal_configuration):
    animal_configuration = np.array(animal_configuration)
    return (animal_configuration - np.min(animal_configuration)) / (np.max(animal_configuration) - np.min(animal_configuration))

def create_histogram(animal_configuration, num_bins=10):
    hist, _ = np.histogram(animal_configuration, bins=num_bins, range=(0, 1))
    return hist

def probability_distribution(hist):
    N_total = np.sum(hist)
    return hist / N_total

def information_entropy(prob_distribution):
    H = entropy(prob_distribution)
    return H

bell = lambda x, m, s: np.exp(-((x-m)/s)**2 / 2)

def perturbation(n, percentage, number):
    result = []
    step_size = percentage / (number - 1)
    for i in range(number):
        perturb = -percentage + (i * step_size * 2)
        val = n * (1 + perturb / 100)
        result.append(val)
    return result

def tolerance_cal(good_parameters, separation_number):
    y = (len(good_parameters)/ (separation_number * separation_number))
    return y

def perturb_pattern(A, p):
    noise = np.random.rand(*A.shape)
    perturbed_A = np.where(noise < p, 1 - A, A)
    return perturbed_A

def perturb_patternalt(A, p, cells):
    cells_shape = cells.shape
    cells_center = tuple(slice((A.shape[i] - cells_shape[i]) // 2, (A.shape[i] + cells_shape[i]) // 2) for i in range(cells.ndim))
    noise = np.random.rand(*cells_shape)
    perturbed_cells = np.where(noise < p, 1 - cells, cells)
    perturbed_A = A.copy()
    perturbed_A[cells_center] = perturbed_cells
    return perturbed_A

from sklearn.metrics import normalized_mutual_info_score as nmi
from skimage.metrics import structural_similarity as ssim

def binarize_image(image, threshold=0.5):
    return (image > threshold).astype(int)

def similarity(A, B):
    mse = np.mean((A - B) ** 2)
    ssim_index = ssim(A, B, data_range=1)
    return mse, ssim_index

def measure_recovery(A, fK, T, m, s, p, steps, C, growth):
    global recovered_A
    perturbed_A = perturb_patternalt(A, p, C)
    recovered_A = run_simulation(perturbed_A, fK, T, m, s, steps, growth)
    mse, ssim_index = similarity(A, recovered_A)
    return mse, ssim_index

def calculate_robustness(A, fK, T, m, s, steps, p_values, C, growth):
    recovery_measures = [measure_recovery(A, fK, T, m, s, p, steps, C, growth) for p in p_values]
    mse_auc = np.trapz([mse for mse, _ in recovery_measures], p_values)
    ssim_auc = np.trapz([ssim for _, ssim in recovery_measures], p_values)
    return mse_auc, ssim_auc
def run_simulation(A, fK, T, m, s, steps, growth):
    for i in range(steps):
        center = center_of_mass(A)
        shift = np.array(A.shape) // 2 - center
        A = shift_array(A, shift.astype(int))
        U = np.real(np.fft.ifft2(fK * np.fft.fft2(A)))
        A = np.clip(A + 1/float(T) * growth(U, m, s), 0, 1)
        A = shift_array(A, -shift.astype(int))
    return A

def calculate_size(arr):
    return np.count_nonzero(arr)

def calculate_symmetry(arr, num_angles=360):
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    coefficients = [rotate_and_compare(arr, angle) for angle in angles]
    max_coefficient = np.max(coefficients)
    return max_coefficient

from skimage.morphology import binary_closing

def calculate_area_and_perimeter(numpy_array):
    filled_array = binary_fill_holes(numpy_array)
    contours = find_contours(filled_array, 0.5)

    total_area = np.count_nonzero(filled_array) - np.count_nonzero(numpy_array)
    total_perimeter = 0

    for contour in contours:
        total_perimeter += np.sum(np.sqrt(np.sum((contour[:-1] - contour[1:])**2, axis=1)))

    return total_area, total_perimeter


def calculate_circularity(numpy_array):
    area, perimeter = calculate_area_and_perimeter(numpy_array)
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity

def dice_coefficient(a, b):
    intersection = np.count_nonzero(np.logical_and(a, b))
    return 2 * intersection / (np.count_nonzero(a) + np.count_nonzero(b))

def rotate_and_compare(arr, angle):
    rotated_arr = transform.rotate(arr, angle, preserve_range=True)
    flip_array = np.flip(rotated_arr, axis=0)
    return dice_coefficient(rotated_arr, flip_array)

def animate_lenia(A, fK, T, m, s,growth, steps):
    no_cells_frame = None
    for i in range(steps):
        center = center_of_mass(A)
        shift = np.array(A.shape) // 2 - center
        A = shift_array(A, shift.astype(int))

        U = np.real(np.fft.ifft2(fK * np.fft.fft2(A)))
        A = np.clip(A + 1/float(T) * growth(U, m, s), 0, 1)

        A = shift_array(A, -shift.astype(int))

        if np.sum(A) == 0 and no_cells_frame is None:
            no_cells_frame = i
            break
    return A, no_cells_frame
    
    
def check_and_resize(C, size):
    scale_factor = 1
    if C.shape[0] > size or C.shape[1] > size:
        scale_factor = min(size / C.shape[0], size / C.shape[1])
        C = scipy.ndimage.zoom(C, scale_factor, order=0)
    return C, scale_factor

def tolerance(name, data):
    def growth(U, m, s):
        return bell(U, m, s) * 2 - 1

    size = 256
    mid = size // 2
    # name = 'Orbium unicaudatus'
    print(name)
    params = get_params(name, data)
    if params is None:
        raise ValueError(f"No params found for animal '{name}'")
    if params:
        R = params['R']
        T = params['T']
        b = params['b']
        m = params['m']
        s = params['s']
    b = np.asarray(turn_to_float(b))

    m_values = perturbation(m, 20, 10)
    s_values = perturbation(s, 20, 10)
    results = {}

    for m, s in tqdm(itertools.product(m_values, s_values),
                           total=len(m_values)*len(s_values),
                           bar_format='{l_bar}{bar:50}{r_bar}'):

        A = np.zeros([size, size])
        C = np.asarray(rle2arr(get_cell_by_name(name, data)))
        C, scale_factor = check_and_resize(C, size)
        R *= scale_factor

        cx = (size - C.shape[0]) // 2
        cy = (size - C.shape[1]) // 2

        A[cx:cx+C.shape[0], cy:cy+C.shape[1]] = C

        initial_frame = A.copy()

        D = np.linalg.norm(np.ogrid[-mid:mid, -mid:mid]) / R * len(b)
        K = (D<len(b)) * b[np.minimum(D.astype(int),len(b)-1)] * bell(D % 1, 0.5, 0.15)
        fK = np.fft.fft2(np.fft.fftshift(K / np.sum(K)))

        final_state, no_cells_frame = animate_lenia(A, fK, T, m, s,growth, steps=300)

        results[(m, s, R, T)] = (final_state, no_cells_frame)

    good_params = []
    for params, (final_state, no_cells_frame) in results.items():
        if no_cells_frame is None and soup_or_stable(final_state) == 'stable':
            good_params.append(params)
    tol = tolerance_cal(good_params, 10)
    return tol, results, good_params

def center_of_mass(A):
    indices = np.indices(A.shape).reshape(2, -1)
    mass = np.sum(A)
    if mass == 0:
        return np.array(A.shape) // 2
    center = np.sum(indices * A.reshape(1, -1), axis=1) / mass
    return center

def shift_array(A, shift):
    shifted_A = np.roll(A, shift, axis=(0, 1))
    return shifted_A

def check_extinction_frame(animal_name):
    with open('Complete_information.csv', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header row
        for row in reader:
            if row[0] == animal_name:
                extinction_frame = row[1]
                if extinction_frame == '-1':
                    return 'A'
                else:
                    return 'B'


# In[ ]:




