# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Import Necessary Packages

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from PIL import Image
import os

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Define inputs and outputs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get recipe inputs
# 1. Labelled data with annotations
labels = dataiku.Dataset("vid_output")
labels_df = labels.get_dataframe()

# 2. Folder where labelled images are found
image_folder = dataiku.Folder("vid_fps_s3")

# Get recipe outputs
# 1. Folder where images & labels are stored to comply with YOLO format
output_folder = dataiku.Folder("yolo_converted")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Check if output folder is in filesystem (to make training with YOLO in Dataiku work)
if output_folder.get_info()['type'] != 'Filesystem':
    raise(Exception("The output folder used is not a Filesystem connection"))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Convert annotations to one txt per image
# 
# The format to follow in the txt file is as such: [Github Comment](https://github.com/ultralytics/yolov5/discussions/7370#discussioncomment-2542801)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Function to format (using centers & normalise) the labels generated from Labelling recipe
def normalise_bbox(path_to_image, label):
    height = 0
    width = 0

    with image_folder.get_download_stream(path_to_image) as stream:
        img = Image.open(stream)
        width, height = img.size

    output = []

    if type(label) != list:
        label = eval(label)

    for obj in label:
        normalised_obj = {}
#         obj = eval(obj)
        normalised_obj['category'] = obj['category']
        normalised_obj['bbox'] = [(obj['bbox'][0] + obj['bbox'][2]/2)/width, ## x-center
                                  (obj['bbox'][1] + obj['bbox'][3]/2)/height, ## y-center
                                  obj['bbox'][2]/width, ## width
                                  obj['bbox'][3]/height] ## height
        output.append(normalised_obj)

    return output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Apply the function on the input dataframe
labels_df['normalised_label'] = labels_df.apply(lambda x: normalise_bbox(x['path'], x['label']), axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Extract the unique categories of the object detection task
categories = set()

for labels in labels_df['normalised_label']:
    for obj in labels:
        categories.add(obj['category'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create reference dictionary for use when creating/writing the txt files containing labels for images
reference_dict = {}

for index, value in enumerate(categories):
    reference_dict[value] = index

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Text Block for use in yaml file
classes = '''names:
'''

for index, value in enumerate(categories):
    classes +=f"""    {index}: {value}
"""

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Store into folder with images & labels at same folder sub-directory

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Split the DataFrame into an 80/20 split (train/val)
split_index = int(len(labels_df) * 0.8)
df1 = labels_df.sample(n=split_index, random_state=47)  # Select 80% of the rows for train set
df2 = labels_df.drop(df1.index)  # Select the remaining 20% of the rows for validation set

# Iterate over train set
for index, row in df1.iterrows():
    # Define working filename
    filename = os.path.basename(row['path']).split(sep='.')[0]

    # Create txt file
    with open("file.txt", "w") as file:
        for obj in row['normalised_label']:
            class_index = reference_dict[obj['category']]
            x_center = obj['bbox'][0]
            y_center = obj['bbox'][1]
            width = obj['bbox'][2]
            height = obj['bbox'][3]
            file.write(f"{class_index} {x_center} {y_center} {width} {height} \n")

    # Upload image and text file
    output_folder.upload_stream(f"/train/images/{filename}.jpg", image_folder.get_download_stream(row['path']))
    output_folder.upload_file(f"/train/labels/{filename}.txt", "file.txt")

    # Remove txt file
    os.remove("file.txt")

# Perform the same with validation set
for index, row in df2.iterrows():
    # Define working filename
    filename = os.path.basename(row['path']).split(sep='.')[0]

    # Create txt file
    with open("file.txt", "w") as file:
        for obj in row['normalised_label']:
            class_index = reference_dict[obj['category']]
            x_center = obj['bbox'][0]
            y_center = obj['bbox'][1]
            width = obj['bbox'][2]
            height = obj['bbox'][3]
            file.write(f"{class_index} {x_center} {y_center} {width} {height} \n")

    # Upload image and text file
    output_folder.upload_stream(f"/valid/images/{filename}.jpg", image_folder.get_download_stream(row['path']))
    output_folder.upload_file(f"/valid/labels/{filename}.txt", "file.txt")

    os.remove("file.txt")

# Create yaml config file for YOLO retraining
with open("file.yaml", "w") as file:
    file.write(f"""
# train: train/images
# val: valid/images

# nc: {len(categories)}

# # Classes
# names: {categories}

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: {output_folder.get_path()}  # dataset root dir
train: train/images  # train images (relative to 'path') 4 images
val: valid/images  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Classes
{classes}
    """)

output_folder.upload_file(f"config.yaml", "file.yaml")

os.remove("file.yaml")
