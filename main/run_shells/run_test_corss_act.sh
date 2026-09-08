#cross-reenactment: source_dir->id  target_dir->motion
export PYTHONPATH='.'
source_dir="assets/example/tracked_image"
target_dir="assets/example/tracked_video"

for target_subdir in "$target_dir"/*/ ; do 
        CUDA_VISIBLE_DEVICES=0 python main/test.py  \
            -d '0' \
            -m assets/GUAVA \
            -s outputs/example\
            --data_path "$target_subdir" \
            --source_data_path $source_dir/NTFbJBzjlts__047 \
            --skip_self_act \
            --render_cross_act 
    done