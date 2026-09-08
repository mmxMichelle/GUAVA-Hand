import os
import time
import gradio as gr
from pathlib import Path
from functools import partial
import subprocess

# --- Constants ---
OUTPUT_DIR = 'outputs/app'
OUTNAME = 'render'
DEVICES = '0'
EHM_TRACKER_DIR = 'EHM-Tracker' # Define the tracker directory

TRACKED_IMG_DIR = 'assets/example/tracked_image'
TRACKED_VID_DIR = 'assets/example/tracked_video'

# --- Core Functions ---

def run_cmd(command, current_dir=None):
    """Executes a shell command and streams its output."""
    print(f"▶️ Executing command:\n{command}", flush=True)
    print(f"▶️ Working directory: {current_dir or os.getcwd()}", flush=True)
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=current_dir,
        bufsize=1
    )

    # Stream stdout
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
        process.stdout.close()

    return_code = process.wait()

    # Handle errors
    if return_code != 0:
        print(f"‼️ Command failed with return code {return_code}", flush=True)
        raise subprocess.CalledProcessError(return_code, command)

    print(f"✅ Command executed successfully.", flush=True)


def master_check_status(source_selection, source_upload, driven_selection, driven_upload):
    """
    Checks the processing status based on the combination of gallery and uploaded inputs.
    """
    src_name, dst_name = None, None

    if source_upload:
        src_name = os.path.splitext(os.path.basename(source_upload))[0]
    elif source_selection:
        src_name = source_selection['caption']
    else:
        return "Please provide a source to check.", None

    if driven_upload:
        dst_name = os.path.splitext(os.path.basename(driven_upload))[0]
    elif driven_selection:
        dst_name = driven_selection['caption']
    else:
        return "Please provide a driving video to check.", None

    output_file = os.path.join(OUTPUT_DIR, f'{OUTNAME}_cross_act', src_name, f'{src_name}_{dst_name}', f'{src_name}_{dst_name}_video.mp4')
    print('Try to find => ' + output_file)

    if not os.path.exists(output_file):
        return "Still processing... You can check progress again later. ⏳", None

    return "Processing completed successfully! 🎉", output_file


def run_master_process(source_selection, source_upload, driven_selection, driven_upload, progress=gr.Progress()):
    """
    A master function to handle all combinations of gallery/upload for source and driven inputs.
    """
    print("\n--- run_master_process called ---")
    print(f"source_selection (from gallery): {source_selection}")
    print(f"source_upload (from upload):   {source_upload}")
    print(f"driven_selection (from gallery): {driven_selection}")
    print(f"driven_upload (from upload):   {driven_upload}")
    print("---------------------------------\n")
    
    try:
        # --- 1. Input Validation ---
        has_source = source_selection is not None or source_upload is not None
        has_driven = driven_selection is not None or driven_upload is not None
        if not has_source or not has_driven:
            return "Error: Please provide both a source and a driving input.", None
        
        progress(0.01, desc="Preparing...")
        
        # --- 2. Setup Paths and Commands ---
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        output_dir = os.path.join(current_dir, OUTPUT_DIR)
        tracker_dir = os.path.join(current_dir, EHM_TRACKER_DIR)
        
        cmd_in_basedir = partial(run_cmd, current_dir=current_dir)
        cmd_in_tracker = partial(run_cmd, current_dir=tracker_dir)

        tracked_source_image_dir = os.path.join(output_dir, 'tracked_source_image')
        tracked_driven_video_dir = os.path.join(output_dir, 'tracked_driven_video')
        os.makedirs(tracked_source_image_dir, exist_ok=True)
        os.makedirs(tracked_driven_video_dir, exist_ok=True)

        # --- 3. Resolve Source Input ---
        progress(0.05, desc="☕️ Processing source...")
        if source_upload:
            print("Processing uploaded source image...")
            source_image_fp = os.path.abspath(source_upload)
            src_name = os.path.splitext(os.path.basename(source_image_fp))[0]
            src_img_root = os.path.join(tracked_source_image_dir, src_name)
            
            if os.path.exists(os.path.join(src_img_root, 'optim_tracking_ehm.pkl')):
                print(f'🐶 Uploaded source image "{src_name}" has been processed before, skipping tracking.')
            else:
                cmd_in_tracker(f'python -m src.tracking_single_image -i "{source_image_fp}" -o "{tracked_source_image_dir}"')
        else: # A gallery item was selected
            src_name = source_selection['caption']
            src_img_root = os.path.join(current_dir, TRACKED_IMG_DIR, src_name)
            print(f"Using pre-tracked source from gallery: {src_name}")
            if not os.path.exists(src_img_root):
                return f"Error: Unable to find source character data path {src_img_root}", None
        
        progress(0.2, desc="✅ Source processed.")

        # --- 4. Resolve Driven Input ---
        progress(0.25, desc="☕️ Processing driven video (can take a while)...")
        if driven_upload:
            print("Processing uploaded driven video...")
            driven_video_fp = os.path.abspath(driven_upload)
            dst_name = os.path.splitext(os.path.basename(driven_video_fp))[0]
            dcv_vid_root = os.path.join(tracked_driven_video_dir, dst_name)

            if os.path.exists(os.path.join(dcv_vid_root, 'optim_tracking_ehm.pkl')):
                 print(f'🐶 Uploaded driven video "{dst_name}" has been processed before, skipping tracking.')
            else:
                cmd_in_tracker(f'python tracking_video.py -i "{driven_video_fp}" -o "{tracked_driven_video_dir}" --check_hand_score 0.0 -p 0,1 -n 1 -v 0')
        else: # A gallery item was selected
            dst_name = driven_selection['caption']
            dcv_vid_root = os.path.join(current_dir, TRACKED_VID_DIR, dst_name)
            print(f"Using pre-tracked driven video from gallery: {dst_name}")
            if not os.path.exists(dcv_vid_root):
                 return f"Error: Unable to find driven video data path {dcv_vid_root}", None
        
        progress(0.65, desc="✅ Driven video processed. Starting final generation...")

        # --- 5. Generate Final Avatar ---
        print('⚡️ Initiating GUAVA generation results, please wait...')
        output_file = os.path.join(output_dir, f'{OUTNAME}_cross_act', src_name, f'{src_name}_{dst_name}', f'{src_name}_{dst_name}_video.mp4')
        
        if os.path.exists(output_file):
            print(f'🐶 The result already exists, skipping generation....')
        else:
            command = (
                f'PYTHONPATH=. python main/test.py -d {DEVICES} -n {OUTNAME} -m assets/GUAVA'
                f' --source_data_path "{src_img_root}"'
                f' --data_path "{dcv_vid_root}"'
                f' --save_path "{output_dir}"'
                f' --skip_self_act --render_cross_act'
            )
            cmd_in_basedir(command)
            print(f'Completion! The results are saved in {output_dir}/{OUTNAME}_cross_act')
            
        progress(1.0, desc="🎉 Complete!")
        return "🎉 Processing complete!", output_file

    except Exception as e:
        return f"An error occurred: {str(e)}", None


# --- History and Gallery Functions ---
def get_history_videos():
    """Get all previously generated videos and format them for a gr.Gallery."""
    results = []
    base_dir = f"{OUTPUT_DIR}/{OUTNAME}_cross_act"
    if not os.path.exists(base_dir):
        return []
        
    for source_dir in sorted(os.listdir(base_dir)):
        source_path = os.path.join(base_dir, source_dir)
        if os.path.isdir(source_path):
            for driven_dir in sorted(os.listdir(source_path)):
                driven_path = os.path.join(source_path, driven_dir)
                if os.path.isdir(driven_path):
                    videos = list(Path(driven_path).glob("*.mp4"))
                    for video in videos:
                        label = f"Source: {source_dir}\nDriven: {driven_dir[len(source_dir)+1:]}"
                        results.append((str(video), label))
    
    results.sort(key=lambda x: os.path.getctime(x[0]), reverse=True)
    return results

def prepare_gallery_data(base_dir):
    """Prepares image-label pairs for the gr.Gallery component."""
    gallery_list = []
    if not os.path.exists(base_dir):
        print(f"Warning: Gallery directory not found at {base_dir}")
        return gallery_list
        
    for item_name in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item_name)
        if os.path.isdir(item_path):
            preview_path = os.path.join(item_path, 'preview.png')
            if os.path.exists(preview_path):
                gallery_list.append((preview_path, item_name))
    return gallery_list

# --- Gradio Interface ---

with gr.Blocks(title="GUAVA Avatar Generator", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("assets/Docs/guava.png", show_label=False, height=100, container=False, interactive=False)
        with gr.Column(scale=5):
            gr.Markdown("""
            # GUAVA: Generalizable Upper Body 3D Gaussian Avatar
            Generate an animated avatar by selecting a pre-processed character and driving video, or by uploading your own. 
            """)
            
    with gr.Row():#equal_height=True
        with gr.Column(scale=3):
            gr.Markdown("## Create GUAVA Avatar")
            source_gallery_data = prepare_gallery_data(TRACKED_IMG_DIR)
            video_gallery_data = prepare_gallery_data(TRACKED_VID_DIR)
            selected_source = gr.State(None)
            selected_video = gr.State(None)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1. Choose Source Character")
                    with gr.Tabs() as source_tabs:
                        with gr.TabItem("From Gallery"):
                            source_gallery = gr.Gallery(value=source_gallery_data, label="Source Characters", columns=3, height="auto", object_fit="contain")
                        with gr.TabItem("Upload Image"):
                            upload_source_image = gr.Image(label="Source Image", type="filepath", height=400)
                with gr.Column():
                    gr.Markdown("### 2. Choose Driving Video")
                    with gr.Tabs() as driven_tabs:
                        with gr.TabItem("From Gallery"):
                            video_gallery = gr.Gallery(value=video_gallery_data, label="Driving Videos", columns=3, height="auto", object_fit="contain")
                        with gr.TabItem("Upload Video"):
                            upload_driven_video = gr.Video(label="Driving Video", height=400)
            
            gr.Markdown("**Note:** Processing custom uploads takes significantly longer due to tracking steps. A 10-second video can take over 6 minutes to process.")
            
            with gr.Row():
                generate_btn = gr.Button("Generate Animation", variant="primary", size="lg")
                check_btn = gr.Button("Check Progress 🔄", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("## GUAVA Generation")
            output_message = gr.Textbox(label="Status", value="Welcome! Select your inputs and click Generate.", lines=2)
            output_video = gr.Video(label="Generated Animation", width=500)

    with gr.Accordion("📜 Generation History", open=False):
        history_gallery = gr.Gallery(label="Previously Generated Videos", show_label=False, columns=4, object_fit="contain", height="auto")
        view_history_btn = gr.Button("Refresh History")
    
    # --- Event Handlers ---

    # --- CHANGE START: Made gallery selection handler an explicit function for clarity and debugging ---
    def on_gallery_select(evt: gr.SelectData):
        """Fires when a user selects an item from a gallery. Prints and returns the selection."""
        if evt.value:
            print(f"✅ Gallery item selected: {evt.value['caption']}")
            return evt.value
        return None

    source_gallery.select(on_gallery_select, None, selected_source, show_progress="hidden")
    video_gallery.select(on_gallery_select, None, selected_video, show_progress="hidden")
    # --- CHANGE END ---

    # --- CHANGE START: Reworked tab switching logic to be more explicit with state handling ---
    def handle_source_tab_change(selected_tab_index, current_selection):
        if selected_tab_index == 1: 
            return None, gr.update()  
        else: 
            return current_selection, None  

    def handle_driven_tab_change(selected_tab_index, current_selection):
        if selected_tab_index == 1:
            return None, gr.update()  
        else: 
            return current_selection, None  

    source_tabs.select(
        fn=handle_source_tab_change,
        inputs=[source_tabs, selected_source],  # Pass current state in
        outputs=[selected_source, upload_source_image],
        show_progress="hidden"
    )
    driven_tabs.select(
        fn=handle_driven_tab_change,
        inputs=[driven_tabs, selected_video],  # Pass current state in
        outputs=[selected_video, upload_driven_video],
        show_progress="hidden"
    )
    # --- CHANGE END ---

    generate_btn.click(
        fn=run_master_process,
        inputs=[selected_source, upload_source_image, selected_video, upload_driven_video],
        outputs=[output_message, output_video]
    )
    check_btn.click(
        fn=master_check_status,
        inputs=[selected_source, upload_source_image, selected_video, upload_driven_video],
        outputs=[output_message, output_video]
    )
    view_history_btn.click(fn=get_history_videos, outputs=history_gallery)
    demo.load(fn=get_history_videos, outputs=history_gallery)

# --- Launch the App ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Whether to share the app.")
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    allowed_paths = [".", OUTPUT_DIR, TRACKED_IMG_DIR, TRACKED_VID_DIR, EHM_TRACKER_DIR]
    
    demo.launch(
        server_name="0.0.0.0", 
        allowed_paths=allowed_paths, 
        share=args.share,
    )