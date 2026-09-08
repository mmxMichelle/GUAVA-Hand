
import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from insightface.app  import FaceAnalysis

class FaceComparator:
    def __init__(
        self,
        ctx_id=0,
        model_name="buffalo_l",
        root="~/.insightface",
        providers=None,
        allowed_modules=None,
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider"]

        self.app = FaceAnalysis(
            name=model_name,
            root=root,
            allowed_modules=allowed_modules,
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id)

    def get_features(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Invalid image: {img_path}")
            faces = self.app.get(img)
            return sorted(faces, key=lambda x:x.bbox[0])[0].normed_embedding  if faces else None
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None

    def process_render_folder(self, source_feat, render_folder):
        total_sim = 0.0
        valid_count = 0
        for img_file in tqdm(os.listdir(render_folder),  desc=f"Processing {os.path.basename(render_folder)}"):
            if not img_file.lower().endswith(('.png',  '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(render_folder,  img_file)
            feat = self.get_features(img_path)
            if feat is not None:
                total_sim += np.dot(source_feat,  feat.T)
                valid_count += 1
        return total_sim, valid_count

def save_json(save_path, total_sim, total_images):
    data = {
        "face_similarity_avg": float(total_sim / total_images) if total_images > 0 else 0.0,
        "face_similarity_total": float(total_sim),
        "total_images": total_images
    }
    with open(save_path, 'w') as f:
        json.dump(data,  f, indent=4)

def process_source(comparator:FaceComparator, source_path, render_root):
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    target_folder = os.path.join(render_root,  base_name)

    if not os.path.exists(target_folder):
        print(f"⚠️ Target folder not found: {target_folder}")
        return 0.0, 0

    source_feat = comparator.get_features(source_path)
    if source_feat is None:
        print(f"⚠️ Failed to extract features from: {source_path}")
        return 0.0, 0

    total_sim = 0.0
    total_images = 0
    subdir_results = []

    for subdir in os.listdir(target_folder):
        subdir_path = os.path.join(target_folder,  subdir)
        render_folder = os.path.join(subdir_path,  'render')

        if not os.path.isdir(render_folder):
            continue

        sub_sim, sub_count = comparator.process_render_folder(source_feat,  render_folder)
        if sub_count > 0:
            subdir_result = {
                "subfolder": subdir,
                "face_similarity_avg": sub_sim / sub_count,
                "face_similarity_total": sub_sim,
                "image_count": sub_count
            }
            subdir_results.append(subdir_result)
            save_json(os.path.join(subdir_path,  'result.json'),  sub_sim, sub_count)
            total_sim += sub_sim
            total_images += sub_count

    if total_images > 0:
        save_json(os.path.join(target_folder,  'result.json'),  total_sim, total_images)
        return total_sim, total_images
    return 0.0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-level Face Comparison Pipeline')
    parser.add_argument('--source_folder',  type=str, required=True, help='Path to source images')
    parser.add_argument('--render_folder',  type=str, required=True, help='Root path for render outputs')
    args = parser.parse_args()

    comparator = FaceComparator()
    global_stats = {"total_sim": 0.0, "total_images": 0}

    for src_file in tqdm(os.listdir(args.source_folder),  desc="Processing source images"):
        if not src_file.lower().endswith(('.png',  '.jpg', '.jpeg')):
            continue

        src_path = os.path.join(args.source_folder,  src_file)
        sub_sim, sub_count = process_source(comparator, src_path, args.render_folder)

        global_stats["total_sim"] += sub_sim
        global_stats["total_images"] += sub_count

    if global_stats["total_images"] > 0:
        final_result = {
            "face_similarity_avg": global_stats["total_sim"] / global_stats["total_images"],
            "face_similarity_total": global_stats["total_sim"],
            "total_images": global_stats["total_images"]
        }
        with open(os.path.join(args.render_folder,  'result.json'),  'w') as f:
            json.dump(final_result,  f, indent=4)
        print(f"\n✅ Final results saved to {os.path.join(args.render_folder,  'result.json')}")