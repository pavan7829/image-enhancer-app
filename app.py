import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from huggingface_hub import from_pretrained_keras

# --- 1. SETUP AGENTIC AI (Google SDK / Gemini) ---
# Safely load the API key from Streamlit's hidden secrets
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    api_key_loaded = True
except Exception as e:
    api_key_loaded = False
    st.warning("Google API Key not found in secrets. Make sure you add it in the Streamlit Cloud dashboard!")

def generate_agentic_report(psnr_score):
    """Uses Gemini to act as an AI Agent analyzing the model's output."""
    if not api_key_loaded:
        return f"**Fallback Analysis:** The model achieved a PSNR of {psnr_score:.2f} dB. Add your API key to enable the Gemini Agent."
        
    try:
        agent = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        You are a Senior Data Scientist evaluating a Deep Learning Image Enhancement model 
        (Google's MAXIM architecture) during a hackathon. 
        
        The model just processed a low-light image and achieved a PSNR (Peak Signal-to-Noise Ratio) score of {psnr_score:.2f} dB.
        
        Write a short, professional, and highly technical paragraph summarizing this result for the judging panel. 
        Explain what this PSNR score means in the context of image restoration (e.g., above 20 dB is decent, 
        closer to 25-30 dB is excellent for low-light datasets like LOL). 
        """
        response = agent.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Agentic AI Error: {str(e)}"

# --- 2. LOAD HUGGING FACE SOTA MODEL ---
@st.cache_resource 
def load_maxim_model():
    """Downloads and loads the pre-trained Google MAXIM model from Hugging Face."""
    return from_pretrained_keras("google/maxim-s2-enhancement-lol")

try:
    maxim_model = load_maxim_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load Hugging Face model. Error: {e}")

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="AI Image Enhancer")
st.title("🌓 GenAI Low Light Image Enhancer")
st.markdown("Powered by Google MAXIM (Hugging Face) & Gemini Agentic AI Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Inputs")
    low_light_file = st.file_uploader("Upload Low Light Image", type=['png', 'jpg', 'jpeg'])
    ground_truth_file = st.file_uploader("Upload Ground Truth (Optional - to calculate PSNR)", type=['png', 'jpg', 'jpeg'])

if low_light_file is not None and model_loaded:
    # Read and preprocess the low light image
    img = Image.open(low_light_file).convert('RGB')
    
    # Resize to 256x256 using TensorFlow
    img_tensor = tf.convert_to_tensor(np.array(img))
    img_resized = tf.image.resize(img_tensor, (256, 256))
    
    # Normalize to [0, 1] and add batch dimension
    img_input = tf.expand_dims(img_resized / 255.0, axis=0) 

    # --- INFERENCE ---
    with st.spinner('MAXIM Neural Network is enhancing the image (this may take 10-20 seconds on cloud CPU)...'):
        prediction = maxim_model.predict(img_input)[0]
    
    # Format output for display
    enhanced_img_array = tf.clip_by_value(prediction * 255.0, 0, 255).numpy().astype('uint8')

    with col2:
        st.subheader("2. Enhancement Results")
        st.image(
            [np.array(img_resized, dtype='uint8'), enhanced_img_array], 
            caption=['Original Dark Image', 'SOTA Enhanced Image'], 
            width=350
        )

        # --- PSNR & AGENTIC WORKFLOW ---
        if ground_truth_file is not None:
            gt_img = Image.open(ground_truth_file).convert('RGB')
            gt_tensor = tf.image.resize(tf.convert_to_tensor(np.array(gt_img)), (256, 256))
            gt_normalized = gt_tensor / 255.0
            
            # Calculate PSNR
            psnr_val = tf.image.psnr(gt_normalized, prediction, max_val=1.0).numpy()
            
            st.markdown("---")
            st.metric(label="Calculated PSNR", value=f"{psnr_val:.2f} dB")
            
            st.subheader("Agentic AI Evaluation")
            if st.button("Generate Technical Report"):
                with st.spinner("Gemini Agent is analyzing the metrics..."):
                    report = generate_agentic_report(psnr_val)
                    st.info(report)
        else:
            st.info("Upload a Ground Truth image to calculate PSNR and trigger the Agentic AI Report.")