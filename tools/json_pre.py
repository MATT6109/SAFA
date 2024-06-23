
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import glob

def interpolate_keypoints(keypoints):
    x = keypoints[0::3]
    y = keypoints[1::3]
    c = keypoints[2::3]
    
    valid_indices_x = [i for i, val in enumerate(x) if val != 0]
    valid_indices_y = [i for i, val in enumerate(y) if val != 0]
    valid_indices_c = [i for i, val in enumerate(c) if val != 0]
    
    if len(valid_indices_x) < 2 or len(valid_indices_y) < 2 or len(valid_indices_c) < 2:
        return keypoints

    interp_func_x = interp1d(valid_indices_x, [x[i] for i in valid_indices_x], bounds_error=False, fill_value="extrapolate")
    interp_func_y = interp1d(valid_indices_y, [y[i] for i in valid_indices_y], bounds_error=False, fill_value="extrapolate")
    interp_func_c = interp1d(valid_indices_c, [c[i] for i in valid_indices_c], bounds_error=False, fill_value="extrapolate")

    x_new = [interp_func_x(i) if i not in valid_indices_x else x[i] for i in range(len(x))]
    y_new = [interp_func_y(i) if i not in valid_indices_y else y[i] for i in range(len(y))]
    c_new = [interp_func_c(i) if i not in valid_indices_c else c[i] for i in range(len(c))]

    interpolated_keypoints = []
    for i in range(len(x)):
        interpolated_keypoints.extend([x_new[i], y_new[i], c_new[i]])
    
    return interpolated_keypoints

def extract_frame_number(image_id):
    return int(image_id.split('.')[0])

def remove_duplicate_frames(data):
    data.sort(key=lambda x: (extract_frame_number(x['image_id']), -x['score']))
    unique_data = []
    seen_frames = set()

    for entry in data:
        frame_number = extract_frame_number(entry['image_id'])
        if frame_number not in seen_frames:
            unique_data.append(entry)
            seen_frames.add(frame_number)

    return unique_data

def interpolate_missing_frames(data):
    frame_numbers = sorted([extract_frame_number(entry['image_id']) for entry in data])
    all_frames = list(range(frame_numbers[0], frame_numbers[-1] + 1))
    missing_frames = set(all_frames) - set(frame_numbers)
    
    if not missing_frames:
        return data

    df = pd.DataFrame(data)
    df['frame_number'] = df['image_id'].apply(extract_frame_number)
    df.set_index('frame_number', inplace=True)
    
    for frame in missing_frames:
        prev_frame = max(f for f in frame_numbers if f < frame)
        next_frame = min(f for f in frame_numbers if f > frame)
        
        prev_keypoints = np.array(df.loc[prev_frame, 'keypoints'])
        next_keypoints = np.array(df.loc[next_frame, 'keypoints'])
        
        interpolated_keypoints = (prev_keypoints + next_keypoints) / 2  # Simple linear interpolation

        new_entry = {
            'image_id': f"{frame}.jpg",
            'category_id': int(df.loc[prev_frame, 'category_id']),
            'keypoints': interpolate_keypoints(interpolated_keypoints.tolist()),
            'score': float((df.loc[prev_frame, 'score'] + df.loc[next_frame, 'score']) / 2),
            'box': list((np.array(df.loc[prev_frame, 'box']) + np.array(df.loc[next_frame, 'box'])) / 2),
            'idx': df.loc[prev_frame, 'idx']  # assuming idx is a list and should be copied directly
        }
        data.append(new_entry)
    
    return sorted(data, key=lambda x: extract_frame_number(x['image_id']))

input_dir = 'C:/Users/haozh/RVI-26/'
output_dir = 'C:/Users/haozh/RVI-26inter/'

os.makedirs(output_dir, exist_ok=True)

for file_path in glob.glob(os.path.join(input_dir, '*.json')):
    with open(file_path, 'r') as f:
        data = json.load(f)

    data = remove_duplicate_frames(data)

    data = interpolate_missing_frames(data)

    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)

    data = json.loads(json.dumps(data, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))

    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f'Processed {file_name} and saved to {output_path}')


def interpolate_keypoints(keypoints):
    # Separate x, y coordinates and confidence scores
    x = keypoints[0::3]
    y = keypoints[1::3]
    
    valid_indices_x = [i for i, val in enumerate(x) if val != 0]
    valid_indices_y = [i for i, val in enumerate(y) if val != 0]
    
    if len(valid_indices_x) < 2 or len(valid_indices_y) < 2:
        # If there are less than 2 valid points, interpolation isn't possible
        return keypoints

    interp_func_x = interp1d(valid_indices_x, [x[i] for i in valid_indices_x], bounds_error=False, fill_value="extrapolate")
    interp_func_y = interp1d(valid_indices_y, [y[i] for i in valid_indices_y], bounds_error=False, fill_value="extrapolate")

    x_new = [interp_func_x(i) if i not in valid_indices_x else x[i] for i in range(len(x))]
    y_new = [interp_func_y(i) if i not in valid_indices_y else y[i] for i in range(len(y))]

    interpolated_keypoints = []
    for i in range(len(x)):
        interpolated_keypoints.extend([x_new[i], y_new[i]])
    
    return interpolated_keypoints

input_dir = '/RVI-26inter/'
output_dir = '/RVI-26inter_noscore/'

os.makedirs(output_dir, exist_ok=True)

for file_path in glob.glob(os.path.join(input_dir, '*.json')):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        entry['keypoints'] = interpolate_keypoints(entry['keypoints'])

    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)
   
    print('duplicated removed and interpolation complete')
    print(f'Processed {file_name} and saved to {output_path}')


def interpolate_keypoints(keypoints):
    x = keypoints[0::2]
    y = keypoints[1::2]
    
    valid_indices_x = [i for i, val in enumerate(x) if val != 0]
    valid_indices_y = [i for i, val in enumerate(y) if val != 0]
    
    if len(valid_indices_x) < 2 or len(valid_indices_y) < 2:
        # If there are less than 2 valid points, interpolation isn't possible
        return keypoints

    interp_func_x = interp1d(valid_indices_x, [x[i] for i in valid_indices_x], bounds_error=False, fill_value="extrapolate")
    interp_func_y = interp1d(valid_indices_y, [y[i] for i in valid_indices_y], bounds_error=False, fill_value="extrapolate")

    x_new = [interp_func_x(i) if i not in valid_indices_x else x[i] for i in range(len(x))]
    y_new = [interp_func_y(i) if i not in valid_indices_y else y[i] for i in range(len(y))]

    interpolated_keypoints = []
    for i in range(len(x)):
        interpolated_keypoints.extend([x_new[i], y_new[i]])
    
    return interpolated_keypoints

def rolling_median_with_edges(df, column, window):
    median_filtered = df[column].rolling(window=window, center=True, min_periods=1).median()
    median_filtered.iloc[0] = df[column].iloc[0]
    median_filtered.iloc[-1] = df[column].iloc[-1]
    return median_filtered

def rolling_mean_with_edges(df, column, window):
    mean_filtered = df[column].rolling(window=window, center=True, min_periods=1).mean()
    mean_filtered.iloc[:window//2] = df[column].iloc[:window//2]  
    mean_filtered.iloc[-window//2:] = df[column].iloc[-window//2:]  
    return mean_filtered

input_dir = '/RVI-26inter_noscore/'
output_dir = '/RVI-26out/'

os.makedirs(output_dir, exist_ok=True)

for file_path in glob.glob(os.path.join(input_dir, '*.json')):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        entry['keypoints'] = interpolate_keypoints(entry['keypoints'])

    df = pd.json_normalize(data)

    keypoints_columns_x = [f'keypoint_{i}_x' for i in range(26)]
    keypoints_columns_y = [f'keypoint_{i}_y' for i in range(26)]
    
    for i in range(26):
        df[f'keypoint_{i}_x'] = df['keypoints'].apply(lambda x: x[2 * i] if len(x) > 2 * i else np.nan)
        df[f'keypoint_{i}_y'] = df['keypoints'].apply(lambda x: x[2 * i + 1] if len(x) > 2 * i + 1 else np.nan)

    window_size = int(30 * 0.5)  # Assuming 30 FPS, adjust as needed
    for column in keypoints_columns_x + keypoints_columns_y:
        df[column] = rolling_median_with_edges(df, column, window_size)

    smoothing_window = int(30 * 0.5)  
    for column in keypoints_columns_x + keypoints_columns_y:
        df[column] = rolling_mean_with_edges(df, column, smoothing_window)

    for i, entry in enumerate(data):
        keypoints = [df.iloc[i][f'keypoint_{j}_{axis}'] for j in range(26) for axis in ['x', 'y']]
        entry['keypoints'] = keypoints

    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f'Processed {file_name} and saved to {output_path}')

