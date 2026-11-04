import os, json, csv, pathlib

def load_yaml(yaml_path):
    # Very simple parser for the generated yaml content (no yaml lib)
    data = {}
    with open(yaml_path, 'r') as f:
        for line in f:
            line=line.strip()
            if line.startswith('train:'):
                data['train']=line.split(':',1)[1].strip()
            if line.startswith('val:'):
                data['val']=line.split(':',1)[1].strip()
    return data

def count_labels(labels_dir):
    class_counts = {}
    total_instances = 0
    for label_file in pathlib.Path(labels_dir).glob('*.txt'):
        with open(label_file) as lf:
            for line in lf:
                parts=line.strip().split()
                if not parts:
                    continue
                class_id=int(parts[0])
                class_counts[class_id]=class_counts.get(class_id,0)+1
                total_instances+=1
    return class_counts, total_instances

def main():
    dataset_root = pathlib.Path('/home/hans/Documents/nckh/data/smartspaces_micro')
    yaml_path = dataset_root / 'smartspaces_micro.yaml'
    if not yaml_path.is_file():
        print('Dataset yaml not found, abort')
        return
    splits = load_yaml(yaml_path)
    out_rows=[]
    for split_name, rel_path in splits.items():
        if split_name in ('train','val'):
            images_dir = dataset_root / 'images' / split_name
            labels_dir = dataset_root / 'labels' / split_name
            img_cnt = len(list(images_dir.glob('*.jpg')))
            class_counts, total_instances = count_labels(labels_dir)
            out_rows.append({
                'split':split_name,
                'num_images':img_cnt,
                'num_instances':total_instances,
                'class_counts':class_counts
            })
    # Write CSV
    csv_path = pathlib.Path('audit/dataset_statistics.csv')
    with open(csv_path,'w',newline='') as csvfile:
        fieldnames=['split','num_images','num_instances','worker','forklift','pallet_truck','amr']
        writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({
                'split':row['split'],
                'num_images':row['num_images'],
                'num_instances':row['num_instances'],
                'worker':row['class_counts'].get(0,0),
                'forklift':row['class_counts'].get(1,0),
                'pallet_truck':row['class_counts'].get(2,0),
                'amr':row['class_counts'].get(3,0),
            })
    # Sampling info (hard‑coded from build_micro_dataset.py)
    with open('audit/sampling_check.txt','w') as f:
        f.write('subsample_interval = 60 frames (≈ 0.5 FPS for 30 FPS source)\n')
        f.write('max_frames_per_camera: train=150, val=50\n')
    print('Dataset audit completed')

if __name__=='__main__':
    main()

