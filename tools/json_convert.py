import os
import json

# Define the directory containing the JSON files
input_dir = './data/38A'
output_dir = './data/38A_out' 
# Function to convert input JSON format to the desired output format
def convert_json(input_data):
    output_data = {
        "data": [],
        "label": "",
        "label_index": 0
    }

    for item in input_data:
        frame_index = int(item["image_id"].split(".")[0])
        pose = item["keypoints"][:52]
        
        category_id = item["category_id"]
        
        frame_data = {
            "frame_index": frame_index,
            "skeleton": [
                {
                    "pose": pose
                }
            ]
        }
        
        output_data["data"].append(frame_data)
        output_data["label"] = category_id
        output_data["label_index"] = category_id
    
    return output_data

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        input_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        # Read the input JSON file
        with open(input_path, 'r') as infile:
            input_data = json.load(infile)
        
        # Convert the JSON data
        output_data = convert_json(input_data)
        
        # Write the converted JSON to the same file
        with open(out_path, 'w') as outfile:
            json.dump(output_data, outfile, indent=4)

print("Conversion completed for all JSON files in the directory.")
