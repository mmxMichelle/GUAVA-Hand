export PYTHONPATH='.'
base_dir="assets/example/tracked_video"

for subdir in "$base_dir"/*/ ; do
    if [ -d "$subdir" ]; then
         CUDA_VISIBLE_DEVICES=0 python main/test.py -d '0' -m assets/GUAVA \
        -s outputs/example --data_path "$subdir" --skip_self_act --render_static_novel_views 
    fi
done

