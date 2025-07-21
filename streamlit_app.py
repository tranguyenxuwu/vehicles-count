from inference_utils import predict_image, predict_video
from generate_heatmap import generate_depth_heatmap_image, generate_depth_heatmap_video
import streamlit as st
from ultralytics import YOLO
import torch
import tempfile
import os
from pathlib import Path
import base64

st.title("YOLOv11 Object Detection Demo")

if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None

uploaded_model = st.file_uploader("Upload YOLOv11 model (.pt)", type=["pt"])
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

conf = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

# Add depth heatmap options
st.subheader("Background Perspective Depth Heatmap")
enable_depth_heatmap = st.checkbox("Enable Background Perspective Depth Heatmap", value=False, 
                                  help="Generate depth visualization based on perspective (vertical position) showing scene structure")

if st.button("Run Predict"):
    if uploaded_model is not None and uploaded_file is not None:
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model_path = None
        input_path = None
        try:
            # Step 1: Save uploaded files
            status_text.text("📁 Saving uploaded files...")
            progress_bar.progress(10)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
                tmp_model.write(uploaded_model.read())
                model_path = tmp_model.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_input:
                tmp_input.write(uploaded_file.read())
                input_path = tmp_input.name

            # Step 2: Load model
            status_text.text("🤖 Loading YOLO model...")
            progress_bar.progress(30)
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = YOLO(model_path)
            model.to(device)

            if input_path.lower().endswith((".jpg", ".jpeg", ".png")):
                # Step 3: Image inference
                st.session_state.output_video_path = None
                
                if enable_depth_heatmap:
                    status_text.text("� Running depth heatmap analysis...")
                    progress_bar.progress(60)
                    
                    depth_img_path, depth_info = generate_depth_heatmap_image(
                        model, input_path, conf=conf
                    )
                    
                    status_text.text("✅ Displaying depth heatmap results...")
                    progress_bar.progress(90)
                    
                    st.image(depth_img_path, caption="Background Perspective Depth Heatmap Detection Result")
                    
                    # Display depth information
                    st.subheader("Background Perspective Depth Analysis")
                    if depth_info:
                        depth_df_data = []
                        for obj_name, info in depth_info.items():
                            depth_df_data.append({
                                "Object": obj_name.rsplit('_', 1)[0],
                                "Confidence": f"{info['confidence']:.2f}",
                                "Perspective Distance": f"{info['depth']:.1f}m"
                            })
                        st.dataframe(depth_df_data)
                        
                        # Show statistics
                        depths = [info['depth'] for info in depth_info.values()]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Closest Object", f"{min(depths):.1f}m")
                        with col2:
                            st.metric("Farthest Object", f"{max(depths):.1f}m")
                        with col3:
                            st.metric("Average Distance", f"{sum(depths)/len(depths):.1f}m")
                        
                        st.info("🌈 Background perspective depth shows scene structure based on vertical position - top areas are farther, bottom areas are closer")
                    
                    if os.path.exists(depth_img_path):
                        os.remove(depth_img_path)
                else:
                    status_text.text("🖼️ Running image inference...")
                    progress_bar.progress(60)
                    
                    img, counts, out_path = predict_image(model, input_path, conf=conf)
                    
                    # Step 4: Display results
                    status_text.text("✅ Displaying results...")
                    progress_bar.progress(90)
                    
                    st.image(out_path, caption="Detection Result")
                    st.write("Object counts:", counts)
                    
                    if os.path.exists(out_path):
                        os.remove(out_path)
                
                # Step 5: Complete
                progress_bar.progress(100)
                status_text.text("🎉 Image processing completed!")
                    
            elif input_path.lower().endswith((".mp4", ".avi", ".mov")):
                # Step 3: Video setup
                status_text.text("🎬 Preparing video inference...")
                progress_bar.progress(50)
                
                # Get video info for progress tracking
                import cv2
                cap = cv2.VideoCapture(input_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                if enable_depth_heatmap:
                    # Create nested progress bars for video processing
                    video_progress_container = st.container()
                    with video_progress_container:
                        st.write("Depth Heatmap Video Processing:")
                        video_progress = st.progress(0)
                        frame_info = st.empty()
                    
                    # Step 4: Video inference with depth heatmap
                    status_text.text(f"🎥 Processing background perspective depth heatmap video ({total_frames} frames)...")
                    
                    out_path = generate_depth_heatmap_video(
                        model, input_path, conf=conf
                    )
                    
                    # Update progress manually since depth heatmap doesn't have callback yet
                    progress_bar.progress(85)
                    video_progress.progress(100)
                    frame_info.text(f"Completed all {total_frames} frames")
                    
                    # Step 5: Loading video for display
                    status_text.text("📺 Loading depth heatmap video...")
                    progress_bar.progress(90)
                    
                    st.session_state.output_video_path = out_path
                    with open(out_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
                    
                    st.success(f"🎉 Background perspective depth heatmap video shows estimated distances with color coding!")
                    st.info("🌈 Background perspective depth heatmap shows scene structure based on vertical position - top areas are farther, bottom areas are closer")
                    
                else:
                    # Create nested progress bars for video processing
                    video_progress_container = st.container()
                    with video_progress_container:
                        st.write("Video Processing Progress:")
                        video_progress = st.progress(0)
                        frame_info = st.empty()
                    
                    # Progress callback function
                    def update_video_progress(progress, current_frame, total):
                        video_progress.progress(progress)
                        frame_info.text(f"Processing frame {current_frame}/{total} ({progress*100:.1f}%)")
                        # Update main progress bar (60% to 85% range for video processing)
                        main_progress = 60 + (progress * 25)
                        progress_bar.progress(int(main_progress))
                    
                    # Step 4: Video inference with progress updates
                    status_text.text(f"🎥 Processing video ({total_frames} frames)...")
                    
                    out_path = predict_video(model, input_path, conf=conf, progress_callback=update_video_progress)
                    
                    # Step 5: Loading video for display
                    status_text.text("📺 Loading processed video...")
                    progress_bar.progress(90)
                    
                    st.session_state.output_video_path = out_path
                    with open(out_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
                
                # Step 6: Complete
                progress_bar.progress(100)
                status_text.text("🎉 Video processing completed!")

        except Exception as e:
            status_text.text(f"❌ Error during inference: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Clean up temporary files
            if model_path and os.path.exists(model_path):
                os.remove(model_path)
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
    else:
        st.warning("Please upload both a model and an image/video file.")