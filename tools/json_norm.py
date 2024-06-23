import json
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed

def extract_keypoints(data):
    frames = []
    keypoints = []

    for entry in data:
        frame = entry['image_id']
        keypoints_list = entry['keypoints']
        frames.extend([frame] * 26)
        for i in range(26):
            x = keypoints_list[2 * i]
            y = keypoints_list[2 * i + 1]
            keypoints.append((f'keypoint_{i}', x, y))

    df = pd.DataFrame(keypoints, columns=['bp', 'x', 'y'])
    df['frame'] = frames
    return df

def compute_centers_and_angles(df):
    upper_body_keypoints = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18}
    
    def get_center_and_angle(group, joint1, joint2):
        x1, y1 = group.loc[group['bp'] == f'keypoint_{joint1}', ['x', 'y']].values[0]
        x2, y2 = group.loc[group['bp'] == f'keypoint_{joint2}', ['x', 'y']].values[0]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        angle = np.arctan2(y2 - y1, x2 - x1)
        return center_x, center_y, angle

    results = []

    for frame, group in df.groupby('frame'):
        if set(group['bp']).issuperset({f'keypoint_{i}' for i in range(26)}):
            shoulder_center_x, shoulder_center_y, shoulder_angle = get_center_and_angle(group, 5, 6)
            hip_center_x, hip_center_y, hip_angle = get_center_and_angle(group, 11, 12)
            ref_dist = np.sqrt((shoulder_center_y - hip_center_y) ** 2 + (shoulder_center_x - hip_center_x) ** 2)
            group['refx'] = shoulder_center_x
            group['refy'] = shoulder_center_y
            group['ref_dist'] = ref_dist
            group['ref_angle'] = shoulder_angle
            group['upper'] = group['bp'].apply(lambda x: 1 if int(x.split('_')[1]) in upper_body_keypoints else 0)
            group.loc[group['upper'] == 0, ['refx', 'refy', 'ref_angle']] = hip_center_x, hip_center_y, hip_angle
            results.append(group)

    return pd.concat(results)

def normalise_skeletons(df):
    df['x_rotate'] = df['refx'] + np.cos(df['ref_angle']) * (df['x'] - df['refx']) - np.sin(df['ref_angle']) * (df['y'] - df['refy'])
    df['y_rotate'] = df['refy'] + np.sin(df['ref_angle']) * (df['x'] - df['refx']) + np.cos(df['ref_angle']) * (df['y'] - df['refy'])

    df['x_rotate'] = (df['x_rotate'] - df['refx']) / df['ref_dist']
    df['y_rotate'] = (df['y_rotate'] - df['refy']) / df['ref_dist']
    df['x'] = df['x_rotate']
    df['y'] = df['y_rotate']

    df.loc[df['upper'] == 0, 'y'] = df.loc[df['upper'] == 0, 'y'] + 1
    return df

def df_to_json(df, original_data):
    data = []
    for entry in original_data:
        frame = entry['image_id']
        frame_data = df[df['frame'] == frame]
        keypoints = []
        for i in range(26):
            x = frame_data.loc[frame_data['bp'] == f'keypoint_{i}', 'x'].values[0]
            y = frame_data.loc[frame_data['bp'] == f'keypoint_{i}', 'y'].values[0]
            keypoints.extend([x, y])
        entry['keypoints'] = keypoints
        data.append(entry)
    return data

input_dir = '/RVI-26out/'
output_dir = '/RVI-26outnorm/'
os.makedirs(output_dir, exist_ok=True)

def process_file(filename):
    input_file_path = os.path.join(input_dir, filename)
    output_file_path = os.path.join(output_dir, filename)
    
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    
    df = extract_keypoints(data)
    
    df = compute_centers_and_angles(df)
    
    df = normalise_skeletons(df)
    
    processed_data = df_to_json(df, data)
    
    with open(output_file_path, 'w') as f:
        json.dump(processed_data, f)
    
    print(f'Processed data saved to {output_file_path}')

Parallel(n_jobs=-1)(delayed(process_file)(filename) for filename in os.listdir(input_dir) if filename.endswith('.json'))
