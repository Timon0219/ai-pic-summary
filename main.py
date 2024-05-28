import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import streamlit as st
import pandas as pd

# Define descriptive phrases for different layers of interpretation
basic_objects = ["dog", "car", "tree", "person", "bicycle"]
interactions = ["playing fetch", "driving", "walking", "riding a bicycle"]
thematic_contexts = ["leisure time in a park", "morning commute", "outdoor exercise"]

# Initialize the model and processor
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

def analyze_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None

    # Prepare text prompts
    text_prompts = basic_objects + interactions + thematic_contexts
    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)

    # Get model outputs
    outputs = model(**inputs)

    # Calculate similarity
    logits_per_image = outputs.logits_per_image.softmax(dim=1).cpu().detach().numpy().flatten()
    confidence_scores = {prompt: float(score) for prompt, score in zip(text_prompts, logits_per_image)}

    # Organize results
    result = {
        "image_name": os.path.basename(image_path),
        "confidence_scores": confidence_scores,
        "summary": generate_summary(confidence_scores)
    }
    return result

def generate_summary(confidence_scores):
    # Sort by confidence
    sorted_scores = sorted(confidence_scores.items(), key=lambda item: item[1], reverse=True)

    # Pick the highest confidence scores from each category
    basic = [item for item in sorted_scores if item[0] in basic_objects]
    interact = [item for item in sorted_scores if item[0] in interactions]
    thematic = [item for item in sorted_scores if item[0] in thematic_contexts]

    # Create summary text
    summary = f"This image likely contains {basic[0][0]} " if basic else ""
    summary += f"and shows {interact[0][0]} " if interact else ""
    summary += f"in a context of {thematic[0][0]}." if thematic else ""
    return summary.strip()

def process_images(directory):
    results = []
    for image_file in os.listdir(directory):
        image_path = os.path.join(directory, image_file)
        result = analyze_image(image_path)
        if result:
            results.append(result)
            with open(f"{os.path.splitext(image_file)[0]}.json", 'w') as f:
                json.dump(result, f, indent=4)
    return results

# Streamlit UI
st.set_page_config(layout="wide")  # Set the layout to wide

# Custom CSS to extend the width of the content
st.markdown("""
    <style>
        .reportview-container .main .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Image Scene Understanding with CLIP")

image_directory = st.text_input("Image Directory", "images")

if os.path.isdir(image_directory):
    st.write("Analyzing images...")
    results = process_images(image_directory)

    if results:
        for result in results:
            # Display the image and table in the first row
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Display the image
                image_path = os.path.join(image_directory, result['image_name'])
                st.image(image_path, caption=result['image_name'], use_column_width=True)
            
            with col2:
                # Display confidence scores as a table
                confidence_df = pd.DataFrame(list(result['confidence_scores'].items()), columns=['Description', 'Confidence'])
                st.dataframe(confidence_df)
            
            # Display the summary and chart in the second row
            col3, col4 = st.columns([1, 3])
            
            with col3:
                # Display the summary
                st.write(f"**Summary**: {result['summary']}")
            
            with col4:
                # Display confidence scores as a bar chart
                st.bar_chart(confidence_df.set_index('Description'))
            
            st.write("---")
else:
    st.error(f"Directory '{image_directory}' does not exist.")
